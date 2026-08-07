"""Tensor-wise interpolation of two frozen AudioCLIP checkpoints.

Implements Eq. 1 of the paper: for each of the 499 corresponding parameter tensors,

    theta_p = alpha_p * theta_un_p + (1 - alpha_p) * theta_reg_p,   alpha_p = sigmoid(beta_p)

The invariant every caller relies on: alpha multiplies ``theta_un``, the plastic
endpoint, and ``1 - alpha`` multiplies ``theta_reg``, the stability endpoint. The two
slots are roles, not methods: sequential fine-tuning usually occupies the un slot, but
EWC occupies it in the ewc_lwf pair. Only the 499 ``beta`` logits are trainable; both
endpoints stay frozen and are never written back.

The arithmetic itself lives in :mod:`alphawise.fusion.blend`, which knows nothing about
AudioCLIP; this module adds what is AudioCLIP-specific, the endpoint construction and
the 499-coefficient contract. Fitting the coefficients lives in
:mod:`alphawise.fusion.fit`; fixing them at a constant (the WiSE-FT baseline) is the
``fusion.alpha`` path of the same driver, with :mod:`alphawise.fusion.wiseft` as its
one-call form. All of them route through here, so there is exactly one implementation
of the blend.
"""

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call

from ..models.factory import (
    COEFFICIENTS_PER_BRANCH,
    NUM_COEFFICIENTS,
    branch_of,
    build_audioclip,
)
from .blend import blend, sigmoid_alphas

#: ParameterDict keys cannot contain "."; the reference implementation substituted "#",
#: and shipped ``learned_alphas.pth`` files use that encoding, so it is load-bearing.
_KEY_SEP = "#"


def _encode(key: str) -> str:
    return key.replace(".", _KEY_SEP)


def _decode(key: str) -> str:
    return key.replace(_KEY_SEP, ".")


class TensorWiseInterpolation(nn.Module):
    """A differentiable blend of two frozen checkpoints, one coefficient per tensor.

    ``endpoint_un`` is the plastic endpoint, the one alpha multiplies; ``endpoint_reg``
    is the stability endpoint, weighted by ``1 - alpha``. The slots are roles, not
    methods; which trained model sits in which slot is the caller's statement about
    which one should pull the blend toward new behaviour.
    """

    def __init__(self, endpoint_un, endpoint_reg, *, init_alpha: float = 0.5, device=None):
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError(f"init_alpha must lie in (0, 1), got {init_alpha}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.theta_un = build_audioclip(endpoint_un).to(self.device)
        self.theta_reg = build_audioclip(endpoint_reg).to(self.device)
        self._check_endpoints_correspond()

        # functional_call needs a module to carry the graph; it never holds blended values.
        self.base = copy.deepcopy(self.theta_reg).to(self.device)
        for module in (self.base, self.theta_un, self.theta_reg):
            module.requires_grad_(False)

        logit = math.log(init_alpha / (1.0 - init_alpha))
        self.beta = nn.ParameterDict({
            _encode(name): nn.Parameter(torch.tensor(logit, device=self.device))
            for name, _ in self.theta_reg.named_parameters()
        })

    def _check_endpoints_correspond(self) -> None:
        un = {k: tuple(v.shape) for k, v in self.theta_un.named_parameters()}
        reg = {k: tuple(v.shape) for k, v in self.theta_reg.named_parameters()}
        if un != reg:
            only_un = sorted(set(un) - set(reg))
            only_reg = sorted(set(reg) - set(un))
            mismatched = sorted(k for k in set(un) & set(reg) if un[k] != reg[k])
            raise ValueError(
                "endpoints are not architecturally identical; "
                f"only in theta_un: {only_un[:5]}, only in theta_reg: {only_reg[:5]}, "
                f"shape mismatches: {mismatched[:5]}"
            )
        counts = {branch: 0 for branch in COEFFICIENTS_PER_BRANCH}
        for key in reg:
            counts[branch_of(key)] += 1
        if len(reg) != NUM_COEFFICIENTS or counts != COEFFICIENTS_PER_BRANCH:
            raise ValueError(
                f"expected {NUM_COEFFICIENTS} coefficients split {COEFFICIENTS_PER_BRANCH}, "
                f"got {len(reg)} split {counts}"
            )

    def alphas(self) -> dict[str, Tensor]:
        """The interpolation coefficients, keyed by checkpoint parameter name."""
        return sigmoid_alphas({_decode(name): logit for name, logit in self.beta.items()})

    def blended(self) -> dict[str, Tensor]:
        """Eq. 1 through :func:`alphawise.fusion.blend.blend`: the 499 blended tensors,
        plus the endpoints' buffers passed through from ``theta_reg`` unblended.
        Differentiable in ``beta``."""
        return blend(self.theta_un.state_dict(), self.theta_reg.state_dict(), self.alphas())

    def forward(self, audio, image, text, batch_indices):
        """Run the blend without ever materializing it into a module.

        ``functional_call`` substitutes the blended tensors into ``base`` for one forward
        pass, so the graph reaches ``beta`` while ``base`` keeps its own frozen values.
        Passing ``None`` for a modality drops the pairs that involve it, which is how the
        objective is restricted to a subset of the retrieval directions.
        """
        return functional_call(self.base, self.blended(), (audio, image, text, batch_indices))

    @torch.no_grad()
    def materialize(self) -> dict[str, Tensor]:
        """A standalone checkpoint: blended parameters plus the endpoints' buffers.

        Buffers pass through from ``theta_reg`` rather than being blended. The only
        buffer in this architecture is the audio window function, which is a fixed
        constant identical in both endpoints, so the passthrough is exact rather than
        an approximation.
        """
        return {name: tensor.detach().clone() for name, tensor in self.blended().items()}

    @torch.no_grad()
    def alpha_report(self) -> dict:
        """Per-branch coefficient statistics covering all 499 tensors.

        The reference implementation bucketed only transformer-block parameters, so its
        summaries silently described 483 of the 499 coefficients. Every key is counted here
        and the total is asserted, which is why these numbers differ from the historical
        ``alpha_summary.txt`` files.
        """
        per_branch: dict = {branch: [] for branch in COEFFICIENTS_PER_BRANCH}
        for name, alpha in self.alphas().items():
            per_branch[branch_of(name)].append(float(alpha))
        counted = sum(len(v) for v in per_branch.values())
        # A raise, not an assert: this guards a historical silent-wrong-number bug and
        # must survive python -O.
        if counted != NUM_COEFFICIENTS:
            raise RuntimeError(f"report covers {counted} of {NUM_COEFFICIENTS} coefficients")

        def describe(values):
            tensor = torch.tensor(values)
            return {
                "count": len(values),
                "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
            }

        report = {branch: describe(values) for branch, values in per_branch.items()}
        report["all"] = describe([a for values in per_branch.values() for a in values])
        return report
