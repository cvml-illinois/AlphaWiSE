"""Tensor-wise interpolation of two state dicts, with no model knowledge.

This is the portable half of AlphaWiSE. Everything the method contributes is the blend
below, and it consumes plain name-to-tensor mappings, so it applies to any two
checkpoints of the same architecture, from this repository or not. Nothing here imports
the data pipeline, the trainers, the evaluation harness, or AudioCLIP; the
AudioCLIP-specific wrapper that knows about the 499 coefficients and their branches is
:class:`alphawise.fusion.core.TensorWiseInterpolation`, which routes through these
functions rather than reimplementing them.

Role convention, from Eq. 1 of the paper: ``state_dict_a`` is ``theta_un``, the plastic
endpoint, the one alpha multiplies; ``state_dict_b`` is ``theta_reg``, the stability
endpoint, weighted by ``1 - alpha``. The slot is a role, not a method: whichever
checkpoint should pull the blend toward new behaviour goes first, even when that
checkpoint was itself trained with a regularizer.

Applying the method to your own model is two calls::

    import torch
    from alphawise.fusion import blend

    finetuned = torch.load("finetuned.pt")
    merged = blend(finetuned, torch.load("original.pt"),
                   alphas={name: 0.6 for name in finetuned})
    model.load_state_dict(merged)
"""

from __future__ import annotations

from typing import Mapping, Union

import torch
from torch import Tensor

Alpha = Union[float, Tensor]


def blend(state_dict_a: Mapping[str, Tensor], state_dict_b: Mapping[str, Tensor],
          alphas: Mapping[str, Alpha]) -> dict[str, Tensor]:
    """``alpha * a + (1 - alpha) * b`` per tensor, one coefficient per name in ``alphas``.

    Entries of ``state_dict_b`` that ``alphas`` does not name (buffers, usually) pass
    through unblended, so the result is a complete state dict for the shared
    architecture. Tensor coefficients keep their autograd graph, which is what lets the
    coefficients be fitted by gradient descent through this function.
    """
    for name in alphas:
        if name not in state_dict_a or name not in state_dict_b:
            raise KeyError(f"alphas names {name!r}, which is absent from a state dict")
        if state_dict_a[name].shape != state_dict_b[name].shape:
            raise ValueError(
                f"{name!r} differs in shape between the endpoints: "
                f"{tuple(state_dict_a[name].shape)} vs {tuple(state_dict_b[name].shape)}"
            )
    blended = {
        name: alpha * state_dict_a[name] + (1.0 - alpha) * state_dict_b[name]
        for name, alpha in alphas.items()
    }
    return {name: blended.get(name, value) for name, value in state_dict_b.items()}


def sigmoid_alphas(beta: Mapping[str, Alpha]) -> dict[str, Tensor]:
    """Bounded coefficients from unbounded logits: ``alpha = sigmoid(beta)`` per name.

    This is the paper's beta parameterization. Fitting happens in logit space, so the
    coefficients that reach :func:`blend` stay inside (0, 1) by construction.
    Differentiable when the logits are tensors.
    """
    return {name: torch.sigmoid(torch.as_tensor(value)) for name, value in beta.items()}
