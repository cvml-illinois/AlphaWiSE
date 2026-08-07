"""Figure 4: cross-modal t-SNE, continual-learning baseline against AlphaWiSE.

Grid: rows are the two variants (the plain continual model, then the same model
under AlphaWiSE), columns are the three base methods. Every panel shows one
modality pair for one phase-7 checkpoint.

Settings the paper's caption leaves unstated, recovered from the run that produced
the figure and pinned here as module constants:

* five hand-picked classes, in the order that fixes their colour assignment
  (:data:`FIGURE_CLASSES` against :data:`CLASS_PALETTE`);
* :data:`SAMPLES_PER_CLASS` is a **cap**, not a count. On the evaluation pool the
  five classes hold 22 / 25 / 16 / 20 / 25 usable pairs, so the panel carries 108
  image points, not 125. The caption's "25 samples per class" is an upper bound;
* one background point per class that was not selected, drawn from the same
  modality as the first plotted one (522 grey markers on the release pool);
* perplexity 30 and t-SNE seed 0, with the sample draw pinned at seed 42.

Two properties worth keeping in mind when reading the figure. t-SNE runs **once
per panel**, over the concatenation of both modalities and the background, so the
six panels have independent, non-comparable two-dimensional layouts; and the
original run never seeded the per-class sample draw, so the released figure is
regenerated under seed 42 and its point placement differs from the published PDF.
For the same reason every projected coordinate is also written to
``tsne_points.csv`` beside the panels, so the layout a release actually shipped
stays recoverable without re-running the projection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from . import style

#: Class names as they appear in the ontology; the order fixes the colour mapping.
FIGURE_CLASSES = (
    "Video game music",
    "Snare drum",
    "Classical music",
    "Crying, sobbing",
    "Chicken, rooster",
)
SAMPLES_PER_CLASS = 25  # cap; the pool supplies 22 / 25 / 16 / 20 / 25
SUBSET_SEED = 42
PERPLEXITY = 30
TSNE_RANDOM_STATE = 0
ENCODE_BATCH = 32
FIG_DPI = 250

METHOD_COLUMNS = ("EWC", "iCaRL", "LwF")
VARIANT_ROWS = ("CL Method", "AlphaWiSE")

#: Modality pair filling every panel. The first entry is always a per-sample
#: modality, so it can also supply the background points.
PAIR_CONFIG = {
    "i2t": ("image", "text", "Image-to-Text"),
    "a2t": ("audio", "text", "Audio-to-Text"),
    "i2a": ("image", "audio", "Image-to-Audio"),
}

MODALITY_MARKER = {"image": "o", "audio": "^", "text": "X"}
MODALITY_SIZE = {"image": 28, "audio": 28, "text": 55}
MODALITY_ALPHA = {"image": 0.70, "audio": 0.70, "text": 0.95}
MODALITY_LABEL = {"image": "Image (circle)", "audio": "Audio (triangle)", "text": "Text (cross)"}
MODALITY_ZORDER = {"image": 1, "audio": 2, "text": 3}

BACKGROUND_COLOR = "0.6"


def effective_perplexity(n_points: int, perplexity: int = PERPLEXITY) -> int:
    """t-SNE needs perplexity below the sample count; the cap binds above 90 points."""
    return min(perplexity, max(5, n_points // 3))


def run_tsne(
    X: np.ndarray,
    *,
    perplexity: int = PERPLEXITY,
    random_state: int = TSNE_RANDOM_STATE,
) -> np.ndarray:
    """Rows of ``X`` projected to two dimensions, PCA init and cosine distance.

    Cosine rather than Euclidean, because these are joint-space embeddings that every
    retrieval metric in the pipeline also compares by inner product on the unit sphere.
    """
    from sklearn.manifold import TSNE  # imported late: only the rendering path needs it

    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity(len(X), perplexity),
        random_state=random_state,
        max_iter=1000,
        metric="cosine",
    ).fit_transform(X)


@dataclass(frozen=True)
class Panel:
    """One subplot's points, already projected to two dimensions."""

    points_a: np.ndarray
    labels_a: np.ndarray
    points_b: np.ndarray
    labels_b: np.ndarray
    background: np.ndarray | None = None


def project_panel(
    embeddings_a: np.ndarray,
    labels_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels_b: np.ndarray,
    background: np.ndarray | None = None,
    *,
    perplexity: int = PERPLEXITY,
    random_state: int = TSNE_RANDOM_STATE,
) -> Panel:
    """Project both modalities and the background in a single shared t-SNE run."""
    parts = [embeddings_a, embeddings_b]
    if background is not None and len(background):
        parts.append(background)
    projected = run_tsne(np.concatenate(parts, axis=0),
                         perplexity=perplexity, random_state=random_state)
    n_a, n_b = len(embeddings_a), len(embeddings_b)
    return Panel(
        points_a=projected[:n_a],
        labels_a=np.asarray(labels_a),
        points_b=projected[n_a:n_a + n_b],
        labels_b=np.asarray(labels_b),
        background=projected[n_a + n_b:] if len(parts) == 3 else None,
    )


#: Saturated hues only, in class order. The published render used a ``tab20`` resample
#: whose fourth slot was a light grey and vanished into the background crosses; every
#: entry here is chromatic, so no class can collide with :data:`BACKGROUND_COLOR`.
CLASS_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                 "#CC79A7", "#56B4E9", "#F0E442")


def class_colors(n_classes: int) -> list:
    """The first ``n_classes`` palette entries, so a class keeps one colour in every panel."""
    if n_classes > len(CLASS_PALETTE):
        raise ValueError(
            f"{n_classes} classes but only {len(CLASS_PALETTE)} palette entries; the "
            "palette is fixed rather than cycled so no two classes share a colour"
        )
    return list(CLASS_PALETTE[:n_classes])


def plot_pair_tsne(
    ax,
    panel: Panel,
    *,
    modality_a: str,
    modality_b: str,
    colors: Sequence,
    title: str = "",
) -> None:
    """Draw one panel: colour carries the class, marker carries the modality.

    The axes are left without ticks or spines deliberately. Each panel is an independent
    t-SNE run, so its coordinates mean nothing outside itself and a visible scale would
    invite comparisons between panels that the projection does not support.
    """
    if panel.background is not None and len(panel.background):
        # Drawn first so the highlighted classes sit on top of the label space.
        ax.scatter(panel.background[:, 0], panel.background[:, 1], c=BACKGROUND_COLOR,
                   marker="X", s=20, alpha=0.5, linewidths=0.0, zorder=0, rasterized=True)

    for modality, points, labels in (
        (modality_a, panel.points_a, panel.labels_a),
        (modality_b, panel.points_b, panel.labels_b),
    ):
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[colors[label] for label in labels],
            marker=MODALITY_MARKER[modality],
            s=MODALITY_SIZE[modality],
            alpha=MODALITY_ALPHA[modality],
            linewidths=0.3,
            edgecolors="white" if modality != "text" else "none",
            zorder=MODALITY_ZORDER[modality],
            rasterized=True,
        )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _modality_legend(fig, modality_a: str, modality_b: str) -> None:
    handles = [
        plt.scatter([], [], marker=MODALITY_MARKER[m], color="dimgray",
                    s=MODALITY_SIZE[m] * 2.2, label=MODALITY_LABEL[m])
        for m in (modality_a, modality_b)
    ]
    handles.append(
        plt.scatter([], [], marker="X", color=BACKGROUND_COLOR, s=MODALITY_SIZE["text"] * 2.2,
                    label=f"Other classes ({modality_a})")
    )
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=18,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.0))


def plot_tsne_grid(
    panels: Sequence[Sequence[Panel]],
    out_paths: Sequence[str | Path],
    *,
    pair: str = "it",
    n_classes: int = len(FIGURE_CLASSES),
    column_labels: Sequence[str] = METHOD_COLUMNS,
    row_labels: Sequence[str] = VARIANT_ROWS,
    dpi: int = FIG_DPI,
) -> tuple[Path, ...]:
    """``panels[row][col]`` laid out as rows of variants over columns of methods."""
    modality_a, modality_b, pair_label = PAIR_CONFIG[pair]
    colors = class_colors(n_classes)
    rc = dict(style.ANALYSIS_RC)
    rc.update({"font.family": "sans-serif", "font.size": 18, "axes.linewidth": 0.5})

    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(row_labels), len(column_labels),
                                 figsize=(15, 10), dpi=dpi, squeeze=False)
        glyphs = "   ".join(MODALITY_LABEL[m] for m in (modality_a, modality_b))
        fig.suptitle(
            f"CL method vs AlphaWiSE, {pair_label} cross-modal alignment\n"
            f"Colour = class label  |  {glyphs}",
            fontsize=22, y=1.00,
        )
        for row, row_panels in enumerate(panels):
            for col, panel in enumerate(row_panels):
                plot_pair_tsne(axes[row][col], panel, modality_a=modality_a,
                               modality_b=modality_b, colors=colors,
                               title=column_labels[col] if row == 0 else "")
        for row, label in enumerate(row_labels):
            axes[row][0].set_ylabel(label, fontsize=20, fontweight="bold", labelpad=8)
        _modality_legend(fig, modality_a, modality_b)
        fig.tight_layout(rect=(0, 0.05, 1, 0.98))
        return style.save_figure(fig, out_paths, dpi=dpi)


def write_points_csv(
    panels: Sequence[Sequence[Panel]],
    path: str | Path,
    *,
    pair: str,
    class_names: Sequence[str],
    column_labels: Sequence[str],
    row_labels: Sequence[str],
) -> Path:
    """Every projected point of every panel, one row each.

    Panels are independent t-SNE runs, so a point's identity needs the panel named:
    ``variant`` and ``method`` are the grid position, ``modality`` the marker, and
    ``class`` the colour (empty on the grey background points). This is what makes a
    shipped figure's layout recoverable after the fact.
    """
    modality_a, modality_b, _ = PAIR_CONFIG[pair]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "method", "modality", "class", "x", "y"])
        for variant, row_panels in zip(row_labels, panels):
            for method, panel in zip(column_labels, row_panels):
                for modality, points, labels in (
                    (modality_a, panel.points_a, panel.labels_a),
                    (modality_b, panel.points_b, panel.labels_b),
                ):
                    for point, label in zip(points, labels):
                        writer.writerow([variant, method, modality,
                                         class_names[int(label)],
                                         f"{point[0]:.6f}", f"{point[1]:.6f}"])
                if panel.background is not None:
                    for point in panel.background:
                        writer.writerow([variant, method, modality_a, "",
                                         f"{point[0]:.6f}", f"{point[1]:.6f}"])
    return path


@dataclass(frozen=True)
class TsneFigureConfig:
    """Everything the end-to-end Figure-4 render needs; no site path is defaulted."""

    img_root: Path
    wav_root: Path
    meta_csv: Path
    ont_path: Path
    columns: Sequence[tuple[str, Path, Path]]  # (method name, base ckpt, AlphaWiSE ckpt)
    out_paths: Sequence[Path]
    manifest: Path | None = None
    pair: str = "it"
    classes: Sequence[str] = FIGURE_CLASSES
    samples_per_class: int = SAMPLES_PER_CLASS
    seed: int = SUBSET_SEED
    perplexity: int = PERPLEXITY
    random_state: int = TSNE_RANDOM_STATE
    encode_batch: int = ENCODE_BATCH
    device: str = "cpu"
    dpi: int = FIG_DPI
    assets_dir: Path | None = None
    row_labels: Sequence[str] = VARIANT_ROWS


def _panel_from_embeddings(embeddings, modality_a: str, modality_b: str,
                           perplexity: int, random_state: int) -> Panel:
    """Adapter over the encoder's per-modality arrays and their aligned labels."""
    return project_panel(
        getattr(embeddings, modality_a), getattr(embeddings, f"{modality_a}_labels"),
        getattr(embeddings, modality_b), getattr(embeddings, f"{modality_b}_labels"),
        embeddings.background,
        perplexity=perplexity, random_state=random_state,
    )


def render_cl_vs_alphawise_grid(cfg: TsneFigureConfig) -> tuple[Path, ...]:
    """Build the subset once, then encode, project and plot each of the six checkpoints.

    Besides the figure files, the projected coordinates land in ``tsne_points.csv``
    next to the first output path; the returned tuple carries all of them.
    """
    from ..eval.encode import encode_subset, load_model
    from ..eval.subset import build_eval_subset

    modality_a, modality_b, _ = PAIR_CONFIG[cfg.pair]
    subset = build_eval_subset(
        cfg.img_root, cfg.wav_root, cfg.meta_csv, cfg.ont_path,
        samples_per_class=cfg.samples_per_class,
        selection="explicit",
        explicit_classes=cfg.classes,
        with_background=True,
        seed=cfg.seed,
        manifest=cfg.manifest,
    )
    modalities = frozenset({modality_a, modality_b})

    panels: list[list[Panel]] = [[] for _ in cfg.row_labels]
    for _name, *checkpoints in cfg.columns:
        for row, checkpoint in enumerate(checkpoints):
            model = load_model(Path(checkpoint), device=cfg.device, assets_dir=cfg.assets_dir)
            embeddings = encode_subset(model, subset, modalities=modalities,
                                       batch_size=cfg.encode_batch, encode_background=True,
                                       background_modality=modality_a)
            del model
            panels[row].append(_panel_from_embeddings(
                embeddings, modality_a, modality_b, cfg.perplexity, cfg.random_state))

    column_labels = [name for name, *_ in cfg.columns]
    written = plot_tsne_grid(panels, cfg.out_paths, pair=cfg.pair,
                             n_classes=len(subset.classes),
                             column_labels=column_labels,
                             row_labels=cfg.row_labels, dpi=cfg.dpi)
    points_csv = write_points_csv(
        panels, Path(cfg.out_paths[0]).parent / "tsne_points.csv",
        pair=cfg.pair, class_names=subset.classes,
        column_labels=column_labels, row_labels=cfg.row_labels)
    return (*written, points_csv)


def _parse_column(argument: str) -> tuple[str, Path, Path]:
    name, _, checkpoints = argument.partition("=")
    base, _, alphawise = checkpoints.partition(",")
    if not (name and base and alphawise):
        raise argparse.ArgumentTypeError(f"expected NAME=BASE_CKPT,ALPHAWISE_CKPT, got {argument!r}")
    return name, Path(base), Path(alphawise)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the t-SNE grid end to end, from the raw media and six checkpoints.

    Each ``--column NAME=BASE_CKPT,ALPHAWISE_CKPT`` contributes one figure column, in the
    order given, and its two checkpoints fill the two rows. Checkpoints are loaded one at a
    time and released, so the peak cost is one model, not six.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--img-root", type=Path, required=True)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--column", type=_parse_column, action="append", required=True,
                        metavar="NAME=BASE_CKPT,ALPHAWISE_CKPT")
    parser.add_argument("--out", type=Path, action="append", required=True,
                        help="output path; repeat for several formats")
    parser.add_argument("--manifest", type=Path,
                        help="replay a recorded subset instead of drawing one")
    parser.add_argument("--pair", choices=sorted(PAIR_CONFIG), default="i2t")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--classes", default=None,
                        help="comma-separated class names to highlight; defaults to the "
                             "published figure's five. Override when your ontology "
                             "differs from AudioSet's.")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.classes:
        kwargs["classes"] = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    written = render_cl_vs_alphawise_grid(TsneFigureConfig(
        img_root=args.img_root, wav_root=args.wav_root, meta_csv=args.meta_csv,
        ont_path=args.ontology, columns=args.column, out_paths=args.out,
        manifest=args.manifest, pair=args.pair, seed=args.seed, device=args.device,
        **kwargs,
    ))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
