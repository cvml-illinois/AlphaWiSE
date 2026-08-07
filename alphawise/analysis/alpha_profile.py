"""Figure 5: learned alpha by depth and parameter type.

Every in-block parameter carries two grouping keys, its depth position and its
type, so norm and bias adaptation is not blurred into the weight curves:

  Image/Text blocks: attention weights | MLP weights | LayerNorm scale+bias
                     | attention and MLP biases
  Audio blocks:      conv weights ("MLP / Conv W" slot) | norm scale+bias

The Audio row plots two curves rather than four because a bottleneck block has no
attention submodule and all of its convolutions are bias-free, so the other two
cells are genuinely empty rather than dropped.

Parameters outside any block (embeddings, the conv and FBSP stem, ln_pre/ln_post/
ln_final, visual.proj / text_projection / audio fc, the audio attention heads)
have no depth position; their per-branch mean is drawn as one detached square to
the right of each panel. The three ``logit_scale*`` temperatures are excluded.

Alpha semantics follow the learned-coefficient module: the stored values are
logits, ``alpha = sigmoid(logit)``, and the blend is
``alpha * fine_tuned + (1 - alpha) * stable``. The by-branch marginal that pools
all types within a block is the row-wise pool of :func:`load_alpha_profile`'s
``block_means``; it is not plotted separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from . import style
from ..naming import FUSION_PAIRS, METHOD_LABELS

BRANCHES = ("Audio", "Image", "Text")
PARAM_TYPES = ("Attention W", "MLP / Conv W", "Norm scale+bias", "Other biases")
BRANCH_PARAM_TYPES = {
    "Audio": ("MLP / Conv W", "Norm scale+bias"),
    "Image": PARAM_TYPES,
    "Text": PARAM_TYPES,
}
BLOCKS_PER_BRANCH = {"Audio": 16, "Image": 12, "Text": 12}

#: Residual-block order of the audio tower: layer1 (3) -> layer2 (4) -> layer3 (6) -> layer4 (3).
AUDIO_LAYER_OFFSET = {"layer1": 0, "layer2": 3, "layer3": 7, "layer4": 13}

TYPE_COLORS = dict(zip(PARAM_TYPES, style.CATEGORICAL))
TYPE_LABELS = {
    "Attention W": "attention weights",
    "MLP / Conv W": "MLP / conv weights",
    "Norm scale+bias": "norm scales+biases",
    "Other biases": "other biases",
}

ALPHAS_FILE = "learned_alphas.pth"
BASENAME = "alpha_type_depth_profile"
DEFAULT_PHASES = tuple(range(1, 8))
DEFAULT_DPI = 200

#: One coefficient per backbone parameter tensor. Mirrors
#: ``alphawise.models.factory.NUM_COEFFICIENTS``, which cannot be imported here
#: without pulling the torchvision-backed model stack into a figure module.
NUM_COEFFICIENTS = 499

_AUDIO_LAYER = re.compile(r"layer[1-4]")
#: Branch of a parameter that sits outside every residual block, by leading key part.
_OUTSIDE_BRANCH = {
    "visual": "Image",
    "audio": "Audio",
    "transformer": "Text",
    "token_embedding": "Text",
    "positional_embedding": "Text",
    "ln_final": "Text",
    "text_projection": "Text",
}

X_OUT = 1.18  # detached column holding the params with no depth position
X_SEP = 1.09  # dotted separator between the depth axis and that column


def is_excluded_key(key: str) -> bool:
    """The three learned temperatures carry no depth or type position."""
    return key.startswith("logit_scale")


def _block_position(parts: Sequence[str]) -> tuple[str, int] | None:
    if parts[0] == "visual" and len(parts) > 3 and parts[1] == "transformer":
        return "Image", int(parts[3])
    if parts[0] == "transformer" and parts[1] == "resblocks":
        return "Text", int(parts[2])
    if parts[0] == "audio" and _AUDIO_LAYER.fullmatch(parts[1]):
        return "Audio", AUDIO_LAYER_OFFSET[parts[1]] + int(parts[2])
    return None


def _param_type(key: str, parts: Sequence[str]) -> str:
    # The audio tower keeps its BatchNorm attribute names after the in-place swap to
    # GroupNorm, so the "bn" prefix still identifies a norm parameter.
    if parts[-2].startswith(("ln_", "bn")) or "downsample#1" in key:
        return "Norm scale+bias"
    if parts[-1] == "in_proj_weight" or ("attn" in parts and parts[-1] == "weight"):
        return "Attention W"
    if parts[-1].endswith("bias"):
        return "Other biases"
    return "MLP / Conv W"  # MLP c_fc/c_proj weights, audio conv and downsample conv


def classify_alpha_key(key: str) -> tuple[str, int | None, str | None]:
    """Map a '#'-joined parameter name to (branch, block index, parameter type).

    ``block index`` and ``parameter type`` are ``None`` together for parameters that
    sit outside every residual block. Unrecognised keys raise instead of falling
    through to a default branch, so an architecture change surfaces as an error
    rather than as a silently mislabelled curve.
    """
    if is_excluded_key(key):
        raise KeyError(f"{key} is excluded from the profile; filter with is_excluded_key()")
    parts = key.split("#")
    position = _block_position(parts)
    if position is not None:
        branch, block = position
        return branch, block, _param_type(key, parts)
    branch = _OUTSIDE_BRANCH.get(parts[0])
    if branch is None:
        raise ValueError(f"unclassifiable parameter key: {key}")
    return branch, None, None


@dataclass(frozen=True)
class AlphaProfile:
    """One method's learned alphas across phases, already through the sigmoid."""

    method: str
    phases: tuple[int, ...]
    block_means: dict[tuple[str, str], np.ndarray]  # (branch, type) -> [n_phases, n_blocks]
    outside_means: dict[str, np.ndarray]  # branch -> [n_phases]
    n_blocks: dict[str, int]


def _sigmoid(logit: float) -> float:
    """The numerically stable sigmoid, in plain floats so JSON files never need torch."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    scaled = math.exp(logit)
    return scaled / (1.0 + scaled)


def read_alphas(path: str | Path) -> dict[str, float]:
    """One coefficient file as ``{encoded parameter name: alpha}``, sigmoid applied once.

    Coefficients are stored as logits in three formats, and each takes exactly one
    sigmoid here:

    * a JSON file holding a ``"logits"`` mapping with dotted parameter names, the
      committed reference layout (``results/reference/coefficients/<pair>_phase<k>.json``);
    * a ``.pth`` whose ``"alphas"`` mapping holds logits with ``"#"`` separators, the
      historical shipped ``learned_alphas.pth`` layout;
    * a ``.pth`` holding the fitter's ``"beta"`` ParameterDict state of logit tensors.

    ``"beta"`` wins over ``"alphas"`` when both are present, because the release fitter
    writes both and its ``"alphas"`` are already through the sigmoid; reading them here
    would apply it twice, the defect that once drifted every published coefficient
    toward 0.5. Keys are normalised to the ``"#"`` encoding the shipped files use, so
    :func:`classify_alpha_key` sees one spelling whatever the source format.
    """
    path = Path(path)
    if path.suffix == ".json":
        logits = json.loads(path.read_text())["logits"]
    else:
        import torch  # only the .pth formats need it; the committed JSONs must not

        stored = torch.load(path, map_location="cpu", weights_only=False)
        if "beta" in stored:
            logits = stored["beta"]
        elif "alphas" in stored:
            logits = stored["alphas"]
        else:
            raise ValueError(
                f"{path}: neither 'beta' nor 'alphas' present; file holds {sorted(stored)}"
            )
    logits = {key.replace(".", "#"): float(value) for key, value in logits.items()}
    if len(logits) != NUM_COEFFICIENTS:
        raise ValueError(
            f"{path}: {len(logits)} coefficients, expected {NUM_COEFFICIENTS} (one per "
            "backbone parameter tensor); a truncated file would plot a silently "
            "shorter curve"
        )
    return {key: _sigmoid(logit) for key, logit in logits.items()}


def _bucket_phase(alphas: dict[str, float]):
    """Split one phase into (branch, type, block) -> alphas and branch -> alphas."""
    cells: dict[tuple[str, str, int], list[float]] = {}
    outside: dict[str, list[float]] = {}
    for key, alpha in alphas.items():
        if is_excluded_key(key):
            continue
        branch, block, param_type = classify_alpha_key(key)
        if block is None:
            outside.setdefault(branch, []).append(alpha)
        else:
            cells.setdefault((branch, param_type, block), []).append(alpha)
    return cells, outside


def _derive_n_blocks(cells) -> dict[str, int]:
    n_blocks = {}
    for branch, _, block in cells:
        n_blocks[branch] = max(n_blocks.get(branch, 0), block + 1)
    for branch, expected in BLOCKS_PER_BRANCH.items():
        if n_blocks.get(branch) != expected:
            raise ValueError(
                f"{branch} branch yielded {n_blocks.get(branch)} blocks, expected {expected}"
            )
    return n_blocks


def _block_curve(cells, branch: str, param_type: str, n_blocks: int, where: str) -> np.ndarray:
    """Per-block mean alpha in depth order, refusing a partially populated row."""
    missing = [i for i in range(n_blocks) if (branch, param_type, i) not in cells]
    if missing:
        raise ValueError(
            f"{where}: ({branch}, {param_type}) missing blocks {missing}; "
            "the figure would silently plot a shorter curve"
        )
    return np.array([np.mean(cells[(branch, param_type, i)]) for i in range(n_blocks)])


def _profile_from_alphas(method: str, phases: tuple[int, ...],
                         per_phase_alphas: Sequence[Mapping[str, float]]) -> AlphaProfile:
    """Bucket one already-read alpha mapping per phase into the plotted aggregates."""
    per_phase = [_bucket_phase(alphas) for alphas in per_phase_alphas]

    n_blocks = _derive_n_blocks(per_phase[0][0])
    block_means = {}
    for branch in BRANCHES:
        for param_type in BRANCH_PARAM_TYPES[branch]:
            block_means[(branch, param_type)] = np.stack([
                _block_curve(cells, branch, param_type, n_blocks[branch],
                             where=f"{method} phase {phase}")
                for phase, (cells, _) in zip(phases, per_phase)
            ])
    # Outside-block params are pooled over the whole branch first, then over phases,
    # unlike the in-block values which are pooled per block first.
    outside_means = {
        branch: np.array([np.mean(outside[branch]) for _, outside in per_phase])
        for branch in BRANCHES
    }
    return AlphaProfile(method, phases, block_means, outside_means, n_blocks)


def load_alpha_profile(run_dir: str | Path, phases: Iterable[int], method: str) -> AlphaProfile:
    """Read one ``learned_alphas.pth`` per phase of a fitted run and bucket its alphas."""
    run_dir = Path(run_dir)
    phases = tuple(phases)
    return _profile_from_alphas(
        method, phases,
        [read_alphas(run_dir / f"phase_{p}" / ALPHAS_FILE) for p in phases])


def load_coefficient_profile(coefficients_dir: str | Path, pair: str,
                             phases: Iterable[int]) -> AlphaProfile:
    """One fusion pair's committed coefficient files as a profile column.

    ``pair`` is a :data:`alphawise.naming.FUSION_PAIRS` key, the ``<plastic>_<stable>``
    stem shared with the fusion overlays. Three layouts are accepted per (pair, phase),
    tried in order, so both a curated directory and a raw fusion run tree work:
    ``<dir>/<pair>_phase<k>.json``, ``<dir>/<pair>_phase<k>.pth``, and
    ``<dir>/<pair>/phase_<k>/learned_alphas.pth``, the last being exactly what
    ``alphawise.fusion.fit`` writes when its run directory is placed (or symlinked)
    under ``<dir>`` with the pair's name. The column title is the fused method's
    display label, so the figure header cannot drift from the tables.
    """
    if pair not in FUSION_PAIRS:
        raise ValueError(f"unknown fusion pair {pair!r}; expected one of "
                         f"{sorted(FUSION_PAIRS)}")
    coefficients_dir = Path(coefficients_dir)
    phases = tuple(phases)

    def _locate(phase: int) -> Path:
        candidates = (coefficients_dir / f"{pair}_phase{phase}.json",
                      coefficients_dir / f"{pair}_phase{phase}.pth",
                      coefficients_dir / pair / f"phase_{phase}" / "learned_alphas.pth")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"no coefficients for pair {pair!r} phase {phase} under {coefficients_dir}; "
            f"looked for {', '.join(str(c.relative_to(coefficients_dir)) for c in candidates)}")

    return _profile_from_alphas(
        METHOD_LABELS[FUSION_PAIRS[pair]], phases,
        [read_alphas(_locate(p)) for p in phases])


def _prepare_panel(ax) -> None:
    ax.set_facecolor(style.SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=style.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color=style.AXIS, linewidth=0.8, zorder=1)  # equal blend
    ax.axvline(X_SEP, color=style.GRID, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xlim(-0.03, 1.28)
    ax.set_xticks([0, 0.5, 1.0, X_OUT])
    ax.set_xticklabels(["first", "mid", "last", "emb/\nproj"])
    ax.tick_params(length=0)


def _draw_branch(ax, profile: AlphaProfile, branch: str) -> list[tuple[float, str]]:
    """Depth curves plus the detached emb/proj marker; returns the curve end points."""
    ends = []
    n = profile.n_blocks[branch]
    x = np.arange(n) / (n - 1)
    for param_type in BRANCH_PARAM_TYPES[branch]:
        per_phase = profile.block_means[(branch, param_type)]
        mean, std = per_phase.mean(axis=0), per_phase.std(axis=0)
        color = TYPE_COLORS[param_type]
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=1.8, solid_capstyle="round",
                solid_joinstyle="round", zorder=3, marker="o", markersize=3.4,
                markerfacecolor=color, markeredgecolor=style.SURFACE, markeredgewidth=0.9)
        ends.append((float(mean[-1]), param_type))

    outside = profile.outside_means[branch]
    ax.errorbar([X_OUT], [outside.mean()], yerr=[outside.std()], fmt="s",
                color=style.MUTED, markersize=5, markerfacecolor=style.MUTED,
                markeredgecolor=style.SURFACE, markeredgewidth=1.0,
                elinewidth=1.2, capsize=2.5, zorder=3)
    return ends


def _label_curve_ends(ax, ends: Sequence[tuple[float, str]]) -> None:
    ends = sorted(ends)
    ys = [value for value, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.075)  # nudge apart when labels would collide
    for y, (_, param_type) in zip(ys, ends):
        ax.text(1.32, y, TYPE_LABELS[param_type], va="center", ha="left", fontsize=7.5,
                color=TYPE_COLORS[param_type], clip_on=False, zorder=4)


def _legend_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], color=TYPE_COLORS[t], linewidth=1.8, marker="o", markersize=3.4,
               markerfacecolor=TYPE_COLORS[t], markeredgecolor=style.SURFACE,
               markeredgewidth=0.9, label=TYPE_LABELS[t])
        for t in PARAM_TYPES
    ]
    handles.append(
        Line2D([], [], color=style.MUTED, linewidth=0, marker="s", markersize=5,
               markerfacecolor=style.MUTED, markeredgecolor=style.SURFACE,
               markeredgewidth=1.0, label="emb/proj (no depth)")
    )
    return handles


def plot_alpha_type_depth(
    profiles: Sequence[AlphaProfile],
    out_paths: Sequence[str | Path],
    *,
    dpi: int = 200,
) -> tuple[Path, ...]:
    """Rows are branches, columns are the given profiles in order."""
    if not profiles:
        raise ValueError("no profiles to plot")
    phases = profiles[0].phases
    with plt.rc_context(style.ANALYSIS_RC):
        fig, axes = plt.subplots(len(BRANCHES), len(profiles),
                                 figsize=(9.6, 8.2), sharey=True, dpi=dpi, squeeze=False)
        fig.patch.set_facecolor(style.SURFACE)

        for row, branch in enumerate(BRANCHES):
            for col, profile in enumerate(profiles):
                ax = axes[row][col]
                _prepare_panel(ax)
                ends = _draw_branch(ax, profile, branch)
                if col == len(profiles) - 1:
                    _label_curve_ends(ax, ends)
                if row == 0:
                    ax.set_title(profile.method, fontsize=10, color=style.INK, pad=6)
                if col == 0:
                    ax.set_ylabel(f"{branch}\n" + r"mean learned $\alpha$", fontsize=9)

        axes[0][0].set_ylim(0, 1)
        axes[0][0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        mid = len(profiles) // 2
        axes[0][mid].annotate("equal blend", xy=(1.02, 0.5), xytext=(0, 4), ha="right",
                              textcoords="offset points", fontsize=7.5, color=style.MUTED)
        axes[-1][mid].set_xlabel(
            "block position (relative depth; Audio 16 blocks, Image/Text 12 blocks)"
            "  |  emb/proj: params outside blocks",
            fontsize=8.5, labelpad=6)

        fig.suptitle(r"Learned $\alpha$ by depth and parameter type", fontsize=11,
                     x=0.02, y=0.995, ha="left", color=style.INK)
        fig.text(0.02, 0.966,
                 r"Per-block mean of learned $\alpha$ ($1$ = fine-tuned, $0$ = stable "
                 f"continual model), averaged over phases {phases[0]} to {phases[-1]}; "
                 r"band / error bar: $\pm$1 std across phases",
                 ha="left", fontsize=8.5, color=style.INK2)
        fig.legend(handles=_legend_handles(), loc="upper left", bbox_to_anchor=(0.02, 0.955),
                   ncol=5, frameon=False, fontsize=8, handlelength=1.6,
                   labelcolor=style.INK2, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.925))
        return style.save_figure(fig, out_paths, dpi=dpi)


def write_coefficients_csv(profiles: Sequence[AlphaProfile], path: str | Path) -> Path:
    """The plotted aggregates as machine-readable rows, one per (phase, curve point).

    In-block rows carry the per-phase per-block mean; outside-block rows carry the
    per-phase branch mean, spelt ``param_type == "outside"`` with an empty block. This
    is every number the figure draws (the bands are their std across phases), so a
    reference comparison never has to read pixels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "phase", "branch", "param_type", "block", "mean_alpha"])
        for profile in profiles:
            for (branch, param_type), per_phase in sorted(profile.block_means.items()):
                for phase_index, phase in enumerate(profile.phases):
                    for block in range(per_phase.shape[1]):
                        writer.writerow([profile.method, phase, branch, param_type, block,
                                         f"{per_phase[phase_index, block]:.6f}"])
            for branch, values in sorted(profile.outside_means.items()):
                for phase, value in zip(profile.phases, values):
                    writer.writerow([profile.method, phase, branch, "outside", "",
                                     f"{value:.6f}"])
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Render Figure 5 from the committed coefficient files, plus its CSV counterpart.

    ``--coefficients`` names a directory of ``<pair>_phase<k>.json`` files (the shipped
    ``results/reference/coefficients``) and ``--pairs`` orders the figure columns. The
    phase window and dpi come from the artifact config where it declares them
    (``artifact.profile.phases``, ``artifact.profile.dpi``). All columns are averaged
    over the same phase list, which the subtitle names, so the bands across columns are
    comparable. Three files land in ``--out-dir``: the profile as pdf and png, and
    ``alpha_coefficients.csv`` holding the plotted aggregates.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coefficients", type=Path, required=True,
                        help="directory of <pair>_phase<k>.json coefficient files")
    parser.add_argument("--pairs", default="normal_lwf,normal_ewc,normal_icarl",
                        help="comma list of fusion pairs, one figure column each, in order")
    parser.add_argument("--config", action="append", default=[],
                        help="config overlay, repeatable; may declare artifact.profile.*")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    from ..config import load as load_config
    cfg = load_config(*args.config)
    phases = tuple(int(p) for p in cfg.get("artifact.profile.phases") or DEFAULT_PHASES)
    dpi = int(cfg.get("artifact.profile.dpi") or DEFAULT_DPI)

    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    profiles = [load_coefficient_profile(args.coefficients, pair, phases) for pair in pairs]

    out_paths = style.output_paths(args.out_dir, BASENAME, ("pdf", "png"))
    written = list(plot_alpha_type_depth(profiles, out_paths, dpi=dpi))
    written.append(write_coefficients_csv(profiles,
                                          Path(args.out_dir) / "alpha_coefficients.csv"))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
