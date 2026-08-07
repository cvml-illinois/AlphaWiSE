"""Figure 3: retention curves over the continual-learning phases.

One panel per retrieval direction, one line per method, drawn at full text width with
body-matched serif type so LaTeX places the figure without downscaling it. One figure
covers a single class order, the curves CSV's ``class_order`` column, which is chosen
by the caller since a curves file usually carries both.

The line styles carry the argument. Each fixed-alpha WiSE-FT curve is colour-paired
with the AlphaWiSE curve built from the same two endpoints, dotted with an open
marker against dashed with a filled one, so a pair reads as "one global alpha" versus
"one learned alpha per tensor". The continual baselines stay thin, solid and muted
behind both. The joint upper bound and the unregularised model are not drawn: neither
is a competing method, and including them compresses the range where the pairs differ.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style
from ..naming import (
    CLASS_ORDERS,
    DIRECTIONS,
    DIRECTION_LABELS_TEX,
    METHOD_LABELS,
    METRICS,
    METRIC_LABELS,
)

#: (direction, panel label) in the published panel order, spelt by naming so the
#: panels cannot drift from the Table 1 column groups.
TASKS = tuple((direction, DIRECTION_LABELS_TEX[direction]) for direction in DIRECTIONS)
PANEL_LETTERS = ("(a)", "(b)", "(c)")
YLIM = {"map": (0.05, 0.62), "r1": (0.00, 0.68)}
DASHES = (4.4, 1.6)
GAP_COLOR = "#A50F15"
LAST_PHASE = 7

#: (method key, legend label, colour, linestyle, linewidth, marker, markersize, family).
#: Keys and labels come from :mod:`alphawise.naming`, so the legend cannot drift from
#: Table 1's rows. Family drives z-order, emphasis and the legend row a method lands on.
CURVE_STYLES = tuple(
    (key, METHOD_LABELS[key], *styling)
    for key, styling in (
        ("lwf", ("#332288", "-", 1.2, "v", 3.0, "base")),
        ("ewc", ("#88CCEE", "-", 1.2, "^", 3.0, "base")),
        ("der", ("#44AA99", "-", 1.2, "x", 3.5, "base")),
        ("icarl", ("#117733", "-", 1.2, "+", 3.5, "base")),
        ("cclip", ("#AA4499", "-", 1.2, "*", 4.0, "base")),
        ("wiseft_n_ewc", ("#A50F15", ":", 1.5, "D", 3.2, "wiseft")),
        ("wiseft_n_lwf", ("#D55E00", ":", 1.5, "o", 3.2, "wiseft")),
        ("wiseft_n_icarl", ("#E69F00", ":", 1.5, "s", 3.2, "wiseft")),
        ("alphawise_n_ewc", ("#A50F15", "--", 2.2, "D", 4.5, "ours")),
        ("alphawise_n_lwf", ("#D55E00", "--", 2.2, "o", 4.5, "ours")),
        ("alphawise_n_icarl", ("#E69F00", "--", 2.2, "s", 4.5, "ours")),
    )
)
_FAMILY_ZORDER = {"ours": 6, "wiseft": 4, "base": 2}


def load_phase_curves(path: str | Path) -> dict[str, dict[str, dict[int, dict[str, float | None]]]]:
    """class order -> method -> phase -> ``a2t_avg_r1``-style column -> value.

    Reads the evaluator's raw curves CSV and collapses it through the same
    :func:`alphawise.analysis.tables.aggregate_phase_curves` the results table uses, so
    the figure and the table can never disagree about what a phase's score is. Class
    order stays part of a row's identity: keying on (method, phase) alone would let the
    forward and reverse rows for one phase overwrite each other.
    """
    from .tables import aggregate_phase_curves

    curves: dict[str, dict[str, dict[int, dict[str, float | None]]]] = {}
    for (method, class_order), phases in aggregate_phase_curves(path).items():
        methods = curves.setdefault(class_order, {})
        methods[method] = {phase: dict(columns) for phase, columns in phases.items()}
    return curves


def _line_kwargs(color, linestyle, linewidth, marker, markersize, family) -> dict:
    kwargs = dict(color=color, linewidth=linewidth, marker=marker, markersize=markersize)
    if family == "ours":
        kwargs.update(dashes=DASHES, markerfacecolor=color,
                      markeredgecolor="white", markeredgewidth=0.6)
    elif family == "wiseft":
        kwargs.update(linestyle=linestyle, markerfacecolor="white",
                      markeredgecolor=color, markeredgewidth=1.0)
    else:
        kwargs.update(linestyle=linestyle)
        if marker in ("x", "+"):
            kwargs.update(markeredgewidth=1.0)
    return kwargs


def _draw_panel(ax, curves, column: str) -> None:
    """One direction; annotates the final-phase gap between our best and the best baseline."""
    finals: dict[str, float] = {}
    for key, label, color, linestyle, linewidth, marker, markersize, family in CURVE_STYLES:
        points = curves.get(key)
        if not points:
            continue
        phases = sorted(p for p in points if points[p].get(column) is not None)
        if not phases:
            continue
        ax.plot(phases, [points[p][column] for p in phases], label=label,
                zorder=_FAMILY_ZORDER[family], alpha=1.0 if family != "base" else 0.9,
                **_line_kwargs(color, linestyle, linewidth, marker, markersize, family))
        final = points.get(LAST_PHASE, {}).get(column)
        if final is not None and family in ("ours", "base"):
            finals[family] = max(finals.get(family, final), final)

    if "ours" in finals and "base" in finals:
        gap = finals["ours"] - finals["base"]
        ax.annotate(f"{gap:+.3f}", xy=(LAST_PHASE + 0.15, sum(finals.values()) / 2),
                    fontsize=8, color=GAP_COLOR, ha="left", va="center")


def _legend_rows(curves) -> tuple[list, list, int]:
    """Proxy handles emitted column-major so each method family renders as its own row."""
    families: dict[str, list] = {"base": [], "wiseft": [], "ours": []}
    for key, label, color, linestyle, linewidth, marker, markersize, family in CURVE_STYLES:
        if key in curves:
            handle = Line2D([], [], **_line_kwargs(color, linestyle, linewidth,
                                                   marker, markersize, family))
            families[family].append((handle, label))
    rows = [row for row in (families["base"], families["wiseft"], families["ours"]) if row]
    ncol = max((len(row) for row in rows), default=1)
    spacer = (Line2D([], [], linestyle="none", marker="none"), "")
    for row in rows:
        row.extend([spacer] * (ncol - len(row)))

    handles, labels = [], []
    for index in range(ncol):
        for row in rows:
            handle, label = row[index]
            handles.append(handle)
            labels.append(label)
    return handles, labels, ncol


def plot_retention_curves(
    curves: dict,
    metric: str,
    out_paths: Sequence[str | Path],
    *,
    class_order: str,
    dpi: int = 300,
) -> tuple[Path, ...]:
    """One row of direction panels for a single metric ("map" or "r1") and one class order.

    ``curves`` is the nested mapping ``load_phase_curves`` returns. ``class_order`` is
    required rather than defaulted so a caller cannot render one class order while
    believing it drew the other.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_LABELS)}")
    if class_order not in curves:
        raise KeyError(f"no rows for class order {class_order!r}; file has {sorted(curves)}")
    curves = curves[class_order]
    with plt.rc_context(style.PAPER_RC):
        fig, axes = plt.subplots(1, len(TASKS), figsize=(6.5, 2.4), sharey=True,
                                 constrained_layout=True, squeeze=False)
        axes = axes[0]
        for ax, letter, (task, task_label) in zip(axes, PANEL_LETTERS, TASKS):
            _draw_panel(ax, curves, f"{task}_avg_{metric}")
            ax.set_title(f"{letter} {task_label}", pad=4)
            ax.set_xlabel("Phase")
            ax.set_xticks(range(1, LAST_PHASE + 1))
            ax.set_xlim(0.8, LAST_PHASE + 0.6)
        axes[0].set_ylabel(METRIC_LABELS[metric])
        axes[0].set_ylim(*YLIM[metric])

        handles, labels, ncol = _legend_rows(curves)
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, -0.30 if len(handles) > 2 * ncol else -0.20))
        return style.save_figure(fig, out_paths, dpi=dpi)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the retention-curve figure, one file per metric and output format.

    The curves CSV must carry ``method``, ``class_order`` and ``phase`` columns plus one
    ``<direction>_avg_<metric>`` column per entry of :data:`TASKS`, for example
    ``a2t_avg_map``. Methods outside :data:`CURVE_STYLES` are read and then not drawn,
    which is how the joint upper bound and the unregularised model stay out of the figure.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves", type=Path, required=True, help="phase-curves CSV")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--basename", default="retention_curves")
    parser.add_argument("--metric", choices=["map", "r1", "both"], default="both")
    # The published figure was drawn from a forward-only curves file, so forward is the
    # default; a file carrying both class orders now needs the choice made, not guessed.
    parser.add_argument("--class-order", default="forward", choices=CLASS_ORDERS,
                        help="class order to draw")
    parser.add_argument("--format", action="append", dest="formats",
                        help="output format; repeat for several (default pdf and png)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    curves = load_phase_curves(args.curves)
    metrics = ["map", "r1"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        out_paths = style.output_paths(args.out_dir, f"{args.basename}_{metric}",
                                       args.formats or ("pdf", "png"))
        for path in plot_retention_curves(curves, metric, out_paths,
                                          class_order=args.class_order, dpi=args.dpi):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
