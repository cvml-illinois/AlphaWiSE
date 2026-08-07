"""Fitting the interpolation coefficients, which is the whole of AlphaWiSE's training.

Both endpoints stay frozen. The only tensors that receive gradients are the 499 logits in
:class:`~alphawise.fusion.core.TensorWiseInterpolation`, optimized on the exemplar memory
plus a class-balanced current-phase draw (see :mod:`alphawise.data.fusion_memory`) so
that the blended model is good at the retrieval directions in ``fit.directions``.

Two settings exist to reproduce published numbers rather than to be principled, and both
default to the published behaviour:

``loss_normalization``
    The backbone divides its summed symmetric cross entropies by ``factorial(k)`` where
    ``k`` is the number of modality pairs present, so a joint objective is scaled by 6 and
    a single direction by 1. ``paper_eq3`` instead takes the plain mean the paper states.

``audio_ttf_term``
    The backbone adds an unnormalized time-frequency regularizer from the audio encoder,
    but only when audio is present, so an image-to-text objective silently has no such
    term while the other three do. ``legacy`` is the only implemented value; it names
    that behaviour rather than switching it, and anything else fails loudly.

    python -m alphawise.fusion.fit --config configs/fusion/normal_ewc.yaml \\
        --out-dir checkpoints/alphawise_n_ewc_forward
"""

from __future__ import annotations

import argparse
from math import factorial
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import Config, load, overrides_from_cli
from ..data.audioset import collate_batch
from ..data.fusion_memory import FusionMemoryAudioSet
from ..data.roots import DataRoots
from ..utils import dist
from ..utils.seed import seed_everything, worker_init_fn
from .core import TensorWiseInterpolation

#: A direction is symmetric: ``a2t`` optimizes audio to text and text to audio together.
DIRECTIONS = {
    "joint": ("a", "i", "t"),
    "a2t": ("a", "t"),
    "i2a": ("i", "a"),
    "i2t": ("i", "t"),
}


def modality_mask(directions: str) -> tuple[bool, bool, bool]:
    """Which of (audio, image, text) the objective sees."""
    if directions not in DIRECTIONS:
        raise ValueError(f"directions must be one of {sorted(DIRECTIONS)}, got {directions!r}")
    selected = DIRECTIONS[directions]
    return ("a" in selected, "i" in selected, "t" in selected)


def rescale_loss(loss: Tensor, directions: str, normalization: str) -> Tensor:
    """Convert the backbone's normalization into the requested one.

    Each present modality pair contributes two directed cross entropies, so ``k`` pairs
    give ``2k`` of them. The backbone returns ``sum / factorial(k)``, which happens to be
    their mean for the joint objective (``6/6``) but is twice their mean for a single pair
    (``2/1``). ``paper_eq3`` is the plain mean the paper states, so it rescales by
    ``factorial(k) / 2k``.
    """
    if normalization == "legacy_audioclip":
        return loss
    if normalization != "paper_eq3":
        raise ValueError(f"unknown loss_normalization {normalization!r}")
    pairs = 3 if sum(modality_mask(directions)) == 3 else 1
    return loss * factorial(pairs) / (2 * pairs)


class CoefficientFitter:
    """Optimizes the 499 logits of one phase's blend."""

    def __init__(self, model: TensorWiseInterpolation, *, directions: str = "joint",
                 lr: float = 1e-3, loss_normalization: str = "legacy_audioclip",
                 audio_ttf_term: str = "legacy"):
        if audio_ttf_term != "legacy":
            raise ValueError(
                "audio_ttf_term has exactly one implemented value, 'legacy': the "
                "backbone adds its time-frequency term whenever audio is present and "
                f"exposes no switch. Got {audio_ttf_term!r}."
            )
        self.model = model
        self.directions = directions
        self.loss_normalization = loss_normalization
        self.audio_ttf_term = audio_ttf_term
        self.optimizer = torch.optim.Adam(model.beta.parameters(), lr=lr)
        self.uses_audio, self.uses_image, self.uses_text = modality_mask(directions)

    def step(self, batch) -> float:
        """One optimizer step, returning the loss it was taken on.

        Modalities outside the objective are passed as ``None`` rather than masked after
        the fact, so the backbone never computes the logits for a pair the objective does
        not include.
        """
        audio, image, text = batch
        stack = lambda field: torch.stack(  # noqa: E731
            [t if torch.is_tensor(t) else torch.as_tensor(t) for t in field]
        ).to(self.model.device)

        audio = stack(audio) if self.uses_audio else None
        image = stack(image) if self.uses_image else None
        text = text if self.uses_text else None
        indices = torch.arange(len(batch[0]), device=self.model.device)

        self.optimizer.zero_grad(set_to_none=True)
        _out, loss = self.model(audio, image, text, indices)
        loss = rescale_loss(loss.mean(), self.directions, self.loss_normalization)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def run(self, loader: DataLoader, epochs: int) -> list[float]:
        """Returns the mean loss per epoch, so a caller can see whether fitting moved."""
        history = []
        for epoch in range(epochs):
            losses = [self.step(batch) for batch in loader]
            history.append(sum(losses) / max(1, len(losses)))
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}", flush=True)
        return history


def endpoint_path(root, phase: int, filename: str) -> Path:
    """The checkpoint of one endpoint at one phase, under the per-phase layout."""
    return Path(root) / f"phase_{phase}" / filename


def fit_phase(cfg: Config, phase: int, out_dir: Path, *, device=None) -> Path:
    """Fit one phase, or fix it at ``fusion.alpha``, and write checkpoint plus coefficients."""
    if dist.is_distributed():
        raise RuntimeError(
            "coefficient fitting is single process. The contrastive objective makes the "
            "batch size load bearing, and splitting it across ranks silently shrinks the "
            "negative count that the published coefficients were fitted with."
        )

    fixed_alpha = cfg["fusion.alpha"]
    fit = cfg["fusion.fit"]
    if fixed_alpha is not None and fit["epochs"]:
        raise ValueError(
            "fusion.alpha holds every coefficient at a constant, so there is nothing to "
            "fit; set fusion.fit.epochs to 0 or drop fusion.alpha"
        )
    if fixed_alpha is None and not fit["epochs"]:
        raise ValueError(
            "a zero-epoch fit is the fixed-coefficient baseline; request it explicitly "
            "through fusion.alpha instead of relying on the initialization"
        )

    source = cfg["fusion.source"]
    model = TensorWiseInterpolation(
        endpoint_path(source["un"]["root"], phase, source["filename"]),
        endpoint_path(source["reg"]["root"], phase, source["filename"]),
        init_alpha=fixed_alpha if fixed_alpha is not None else cfg["fusion.init_alpha"],
        device=device,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    class_order = cfg["continual.class_order"]

    if fit["epochs"]:
        seed = seed_everything(cfg["fusion.data.seed"])
        dataset = FusionMemoryAudioSet(
            DataRoots.from_config(cfg, "train"), phase,
            class_order=class_order,
            memory_size=cfg["fusion.data.memory_size"],
            balancing=cfg["fusion.data.balance_current"],
            # Must match the partition the endpoints were trained under, or the fitting
            # set covers different classes than the checkpoints being fused.
            classes_per_phase=cfg["continual.classes_per_phase"],
        )
        print(f"phase {phase}: {len(dataset.exemplars)} exemplars + "
              f"{len(dataset.current_phase)} current = {len(dataset)} samples", flush=True)
        workers = cfg["train.dataloader.num_workers"]
        loader = DataLoader(
            dataset, batch_size=fit["batch_size"], shuffle=True, drop_last=True,
            collate_fn=collate_batch, num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
            worker_init_fn=worker_init_fn(seed),
        )
        fitter = CoefficientFitter(
            model,
            directions=fit["directions"],
            lr=fit["optimizer"]["lr"],
            loss_normalization=fit["loss_normalization"],
            audio_ttf_term=fit["audio_ttf_term"],
        )
        fitter.run(loader, fit["epochs"])

    model.eval()
    fused = out_dir / cfg["fusion.output.model_filename"]
    torch.save(model.materialize(), fused)
    # The coefficient file is self-describing: alpha_profile and any external reader get
    # the endpoints, ordering and objective without the producing config. A fixed blend
    # was never fitted, so its objective and fitting-set size are honestly None.
    torch.save({
        "beta": model.beta.state_dict(),
        "alphas": {k: float(v) for k, v in model.alphas().items()},
        "report": model.alpha_report(),
        "endpoints": {"plastic": source["un"]["method"], "stable": source["reg"]["method"]},
        "class_order": class_order,
        "objective": fit["directions"] if fit["epochs"] else None,
        "memory_size": cfg["fusion.data.memory_size"] if fit["epochs"] else None,
    }, out_dir / cfg["fusion.output.alphas_filename"])
    print(f"phase {phase}: wrote {fused}", flush=True)
    return fused


def main(argv: list[str] | None = None) -> int:
    """Fit every phase named by the config or the command line, in order."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=[], required=True,
                        help="config overlay, repeatable; later overlays win")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="run directory; each phase writes into <out-dir>/phase_<n>")
    parser.add_argument("--phases", help="comma separated, defaults to the config's list")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load(*args.config, cli_overrides=overrides_from_cli(args.set))
    phases = ([int(p) for p in args.phases.split(",")] if args.phases
              else cfg["fusion.phases"])
    for phase in phases:
        fit_phase(cfg, phase, args.out_dir / f"phase_{phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
