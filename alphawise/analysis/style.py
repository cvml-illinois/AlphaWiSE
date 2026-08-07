"""One visual identity for every figure the paper ships.

Two rc profiles, because the paper mixes two figure families:

``ANALYSIS_RC``
    Sans, 9pt, near-white surface with warm-grey rules. Used by the diagnostic
    figures (learned-alpha depth profile, cross-modal t-SNE) that carry their own
    in-figure labelling.
``PAPER_RC``
    Serif, 10pt, ``pdf.fonttype 42``. Used by the results curves, which are placed
    at full text width so LaTeX never downscales them; any other body size would
    make in-figure text disagree with the surrounding paragraph.

Per-figure categorical palettes stay next to the figure that owns them, since each
entry there bundles colour with marker, dash pattern and z-order.
"""

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # figures are always written to disk, never displayed

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots 1-4 of the validated palette, in fixed order, never cycled.
CATEGORICAL = ("#2a78d6", "#1baf7a", "#eda100", "#008300")

ANALYSIS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK2,
}

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.columnspacing": 1.2,
    "legend.handletextpad": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "grid.linestyle": ":",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
}


def output_paths(directory: str | Path, basename: str, formats: Iterable[str]) -> tuple[Path, ...]:
    """``dir/basename.fmt`` for each requested format, for callers driven by a config."""
    directory = Path(directory)
    return tuple(directory / f"{basename}.{fmt.lstrip('.')}" for fmt in formats)


def save_figure(
    fig,
    out_paths: Sequence[str | Path],
    *,
    dpi: int | None = None,
    facecolor: str = SURFACE,
) -> tuple[Path, ...]:
    """Write one figure to every requested path; the suffix picks the format."""
    if not out_paths:
        raise ValueError("no output path given; figure functions never choose their own")
    written = []
    for path in out_paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=facecolor, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return tuple(written)
