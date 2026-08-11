"""LaTeX, Markdown and CSV renderers for the two result tables.

**Results table** (from the evaluation curves CSV, one row per evaluated
(train phase, eval phase) cell): a phase-k checkpoint's score is its mean over the
phases it has seen, eval phases 0..k, and the table collapses those per-phase
scores once more:

* *Average* is average-incremental accuracy, the mean of the per-phase scores over
  the declared phase window. The published protocol averages the trained phases
  1..7 and never the phase-0 bootstrap, so a window that names the bootstrap is
  refused rather than silently folded in;
* *Last Phase* is the final checkpoint of that window.

Row set, row order, class orders, column-group order and the phase window come
from the artifact config where it declares them and from :mod:`alphawise.naming`
otherwise. A method the config declares but the curves CSV does not carry raises
instead of rendering an empty row: a silent dash would drop rows and their bold
cells from Table 1 without warning.

**Transfer table** (from the continual-learning measures CSV): backward and forward
transfer for all three retrieval directions. Reporting every direction removes the
question of why one was chosen, since the principled pick, audio-to-text, is also
the least favourable one. Average accuracy is deliberately absent, it equals the
results table's Last Phase column; forgetting is absent because it is BWT's
negative whenever a method's per-task score decreases monotonically after learning
that task. Row identity comes from :data:`alphawise.naming.TRANSFER_DISPLAY`, the
same mapping the summarize stage writes with.

Every number is read from a CSV, so none is transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from ..naming import (
    AGGREGATION_LABELS,
    BOOTSTRAP_PHASE,
    CLASS_ORDERS,
    DIRECTIONS,
    DIRECTION_LABELS,
    DIRECTION_LABELS_TEX,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_LABELS,
    METRICS,
    NON_COMPETING,
    TRANSFER_METHODS,
    TRANSFER_VARIANTS,
    metric_cols,
    table_cols,
)

#: measure key -> (column header, +1 when higher is better).
MEASURES = {"bwt": (r"BWT $\uparrow$", +1), "fwt": (r"FWT $\uparrow$", +1)}

DEFAULT_BASENAME = "continual_results"
TRANSFER_BASENAME = "cl_transfer_table.tex"
#: Regenerating from the pipeline instead of the paper transcription must not move a
#: cell by more than this, unless the artifact config declares its own band.
DEFAULT_REFERENCE_TOLERANCE = 5.0e-4

TRANSFER_CAPTION = (
    "Backward and forward transfer on all three retrieval directions, with 840 "
    "exemplars. BWT measures how much performance on earlier phases changes by "
    "the end of training; FWT measures how well a model does on a phase it has "
    "not yet seen, relative to the shared phase-0 bootstrap. Both use R@1 under "
    "the same per-phase 79-class protocol as Table~1, over phases 1--7 (phase~0 "
    "is the shared bootstrap). Higher is better for both. Bold marks the better "
    "value in each base/AlphaWiSE pair; $\\Delta$ is AlphaWiSE minus its base."
)


def _fmt(value: object) -> str:
    return f"{float(value):.4f}"  # type: ignore[arg-type]


# --- curves CSV: load and aggregate ------------------------------------------------


def _check_curves_header(fieldnames: Sequence[str] | None, path: Path) -> None:
    """Refuse a file that is not the evaluator's schema, naming what is wrong."""
    have = set(fieldnames or ())
    required = ("method", "class_order", "train_phase", "eval_phase", *metric_cols())
    missing = [field for field in required if field not in have]
    if not missing:
        return
    hint = ""
    if "direction" in have or "at_r1" in have:
        hint = (
            " The file uses the pre-release column names (direction, at_r1, ...); "
            "re-run alphawise.eval.runner, which writes class_order and a2t_r1."
        )
    raise ValueError(f"{path}: curves CSV is missing column(s) {missing}.{hint}")


def aggregate_phase_curves(
    path: str | Path,
) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    """(method, class order) -> train phase -> {``a2t_avg_r1``: value, ...}.

    The evaluator writes one row per (train phase, eval phase) cell. A checkpoint's
    per-phase score is its mean over the eval phases it has seen, so cells above the
    diagonal, present when the same CSV doubles as the transfer matrix, are excluded
    here. This is the one collapse of the evaluator schema; the curves figure plots
    its output directly and the results table collapses it once more.
    """
    path = Path(path)
    per_key: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _check_curves_header(reader.fieldnames, path)
        pools = set()
        for row in reader:
            pools.add(row.get("pool"))
            train_phase, eval_phase = int(row["train_phase"]), int(row["eval_phase"])
            if eval_phase > train_phase:
                continue
            per_key[(row["method"], row["class_order"], train_phase)].append(row)
    if len(pools - {None}) > 1:
        raise ValueError(
            f"{path} mixes candidate pools {sorted(pools - {None})}; scores from "
            "different pools cannot share one table"
        )

    aggregated: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for (method, class_order, train_phase), cells in per_key.items():
        aggregated[(method, class_order)][train_phase] = {
            f"{metric.partition('_')[0]}_avg_{metric.partition('_')[2]}":
                mean(float(cell[metric]) for cell in cells)
            for metric in metric_cols()
        }
    return dict(aggregated)


def collapse_run(
    by_phase: Mapping[int, Mapping[str, float]],
    phases: Sequence[int],
    *,
    where: str,
) -> dict[str, float]:
    """One run's per-phase scores collapsed to the Average and Last Phase columns."""
    missing = [phase for phase in phases if phase not in by_phase]
    if missing:
        raise ValueError(
            f"{where}: curves CSV has no train phase(s) {missing}; the declared "
            f"window is {list(phases)} and averaging a shorter one would be silent"
        )
    last = phases[-1]
    row = {}
    for direction in DIRECTIONS:
        for metric in METRICS:
            per_phase = f"{direction}_avg_{metric}"
            row[f"{direction}_avg_{metric}"] = mean(by_phase[p][per_phase] for p in phases)
            row[f"{direction}_last_{metric}"] = by_phase[last][per_phase]
    return row


# --- the table specification, config first, module defaults second ------------------


@dataclass(frozen=True)
class TableSpec:
    """Everything the results renderers need beyond the numbers themselves."""

    methods: tuple[str, ...]
    methods_declared: bool  # declared rows are a contract; a missing one raises
    class_orders: tuple[str, ...]
    directions: tuple[str, ...]
    phases: tuple[int, ...]
    memory: str
    pool_classes: str  # classes per per-phase candidate pool, a caption label from config
    basename: str
    reference_tolerance: float


def _validated(values: Sequence[str], allowed: Sequence[str], what: str) -> tuple[str, ...]:
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected members of {list(allowed)}")
    return tuple(values)


def table_spec(cfg, curves: Mapping[tuple[str, str], Mapping[int, object]]) -> TableSpec:
    """Resolve the row/column contract from the artifact config, or from what exists.

    ``cfg`` is a resolved :class:`alphawise.config.Config` or None. Without a declared
    ``artifact.methods`` list the row set is every :data:`naming.METHOD_ORDER` key the
    curves carry, in that order; without declared phases the window is every trained
    phase present, bootstrap excluded.
    """
    get = cfg.get if cfg is not None else (lambda dotted, default=None: default)

    declared = get("artifact.methods")
    if declared:
        unknown = [key for key in declared if key not in METHOD_LABELS]
        if unknown:
            raise ValueError(
                f"artifact.methods names {unknown}, which have no display label; "
                "add them to alphawise.naming.METHOD_ORDER"
            )
        methods = tuple(declared)
    else:
        present = {method for method, _ in curves}
        methods = tuple(key for key, _ in METHOD_ORDER if key in present)

    orders = get("artifact.table.class_orders")
    if orders:
        class_orders = _validated(orders, CLASS_ORDERS, "class order")
    else:
        present_orders = {order for _, order in curves}
        _validated(sorted(present_orders), CLASS_ORDERS, "class order")
        class_orders = tuple(order for order in CLASS_ORDERS if order in present_orders)

    directions = _validated(get("artifact.table.directions") or DIRECTIONS,
                            DIRECTIONS, "direction")

    phases = get("artifact.table.phases")
    if phases:
        phases = tuple(sorted(int(phase) for phase in phases))
        if BOOTSTRAP_PHASE in phases:
            raise ValueError(
                "artifact.table.phases includes the phase-0 bootstrap; the published "
                "Average is over trained phases only, so name phases 1.. instead"
            )
    else:
        seen = {phase for by_phase in curves.values() for phase in by_phase}
        phases = tuple(sorted(seen - {BOOTSTRAP_PHASE}))
    if not phases:
        raise ValueError("no trained phases to average over")

    return TableSpec(
        methods=methods,
        methods_declared=bool(declared),
        class_orders=class_orders,
        directions=directions,
        phases=phases,
        memory=str(get("continual.memory.size", "840")),
        pool_classes=str(get("continual.classes_per_phase", "79")),
        basename=get("artifact.table.basename") or DEFAULT_BASENAME,
        reference_tolerance=float(get("artifact.table.reference_tolerance")
                                  or DEFAULT_REFERENCE_TOLERANCE),
    )


def collapse_table_rows(
    curves: Mapping[tuple[str, str], Mapping[int, Mapping[str, float]]],
    spec: TableSpec,
) -> dict[tuple[str, str], dict[str, float]]:
    """The table cells for every (method, class order) the spec selects.

    Declared methods must exist for every declared class order; default-derived
    methods are rendered only where they exist. Either way no row is ever a dash.
    """
    rows = {}
    for method in spec.methods:
        for class_order in spec.class_orders:
            by_phase = curves.get((method, class_order))
            if by_phase is None:
                if spec.methods_declared:
                    raise ValueError(
                        f"method {method!r} is declared by the artifact config but the "
                        f"curves CSV has no {class_order!r} rows for it; the table "
                        "would silently print an empty row"
                    )
                continue
            rows[(method, class_order)] = collapse_run(
                by_phase, spec.phases, where=f"{method} ({class_order})")
    return rows


# --- results table renderers --------------------------------------------------------


def _column_best(rows, class_order: str, cols: Sequence[str]) -> dict[str, float]:
    best = {}
    for col in cols:
        values = [
            row[col]
            for (method, row_order), row in rows.items()
            if row_order == class_order and method not in NON_COMPETING
        ]
        if values:
            best[col] = max(values)
    return best


def _cell(row, col: str, best: dict[str, float], markup: str) -> str:
    text = _fmt(row[col])
    is_best = abs(row[col] - best.get(col, float("inf"))) < 1e-9
    if not is_best or not markup:
        return text
    return f"**{text}**" if markup == "md" else r"\textbf{" + text + "}"


def _order_rows(rows, spec: TableSpec, class_order: str) -> list[tuple[str, str]]:
    return [(key, METHOD_LABELS[key]) for key in spec.methods
            if (key, class_order) in rows]


def render_results_md(rows, spec: TableSpec) -> str:
    """The results table as Markdown, one section per class order.

    ``rows`` is what :func:`collapse_table_rows` returns. The exemplar budget in the
    heading is a caption label from the config, never read back from the data.
    """
    cols = table_cols(spec.directions)
    lines = [
        f"# Cross-modal continual-learning retrieval ({spec.memory} exemplars)\n",
        f"Per-phase evaluation against a {spec.pool_classes}-class candidate pool. **Average** is "
        "average-incremental over trained phases "
        f"{spec.phases[0]}..{spec.phases[-1]} (each phase-k model scored as its mean "
        "over phases 0..k); **Last Phase** is the final model averaged over all "
        "phases. I→A is image→audio retrieval.\n",
    ]
    header = "| Method |" + "".join(
        f" {DIRECTION_LABELS[direction]} {AGGREGATION_LABELS[agg]} {METRIC_LABELS[metric]} |"
        for direction in spec.directions
        for agg in ("avg", "last")
        for metric in METRICS
    )
    for class_order in spec.class_orders:
        best = _column_best(rows, class_order, cols)
        lines.append(f"\n## {class_order.capitalize()}  "
                     "(best competing method per column in **bold**)\n")
        lines.append(header)
        lines.append("|" + "---|" * (len(cols) + 1))
        for key, label in _order_rows(rows, spec, class_order):
            markup = "" if key in NON_COMPETING else "md"
            cells = [_cell(rows[(key, class_order)], col, best, markup) for col in cols]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_results_tex(rows, spec: TableSpec) -> str:
    """The same table as a LaTeX ``table`` float, same numbers and same bolding.

    A self-contained float rather than a bare tabular, so it can be included as is.
    It uses booktabs rules and ``\\resizebox``, which the surrounding document has to
    provide. Column groups follow ``spec.directions``, the published Table 1 order,
    so a regenerated float drops into the manuscript without transposing anything.
    """
    cols = table_cols(spec.directions)
    n_dir = len(spec.directions)
    group_width = 2 * len(METRICS)
    spans, rules, start = [], [], 2
    for direction in spec.directions:
        spans.append(rf"\multicolumn{{{group_width}}}{{c}}{{{DIRECTION_LABELS_TEX[direction]}}}")
        rules.append(rf"\cmidrule(lr){{{start}-{start + group_width - 1}}}")
        start += group_width
    agg_header = " & ".join(
        rf"\multicolumn{{{len(METRICS)}}}{{c}}{{{AGGREGATION_LABELS[agg]}}}"
        for _ in spec.directions for agg in ("avg", "last")
    )
    metric_header = " & ".join(METRIC_LABELS[m] for _ in range(2 * n_dir) for m in METRICS)

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-modal continual learning (" + spec.memory + r" exemplars). "
        r"Per-phase " + spec.pool_classes + r"-class retrieval. Average = average-incremental (mean over phases "
        + f"{spec.phases[0]}--{spec.phases[-1]}" + r"); Last Phase = final model over "
        r"all phases. Bold marks the best competing method per column.}",
        r"\label{tab:continual_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        " & " + " & ".join(spans) + r" \\",
        "".join(rules),
        " & " + agg_header + r" \\",
        r"Method & " + metric_header + r" \\",
        r"\midrule",
    ]
    for class_order in spec.class_orders:
        best = _column_best(rows, class_order, cols)
        # The paper's table is one forward-only block, so the italic class-order
        # sub-header appears only when there genuinely is more than one block.
        if len(spec.class_orders) > 1:
            lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{"
                         + class_order.capitalize() + r"}} \\")
        for key, label in _order_rows(rows, spec, class_order):
            markup = "" if key in NON_COMPETING else "tex"
            cells = [_cell(rows[(key, class_order)], col, best, markup) for col in cols]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_results_csv(rows, spec: TableSpec, path: str | Path) -> None:
    """The same table as machine-readable rows, in spec order.

    This is also the schema of ``results/reference/paper_reported_*.csv``, so the
    rendered numbers and the transcription diff cell by cell with no reshaping.
    """
    cols = table_cols(spec.directions)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "class_order", *cols])
        writer.writeheader()
        for class_order in spec.class_orders:
            for key, _label in _order_rows(rows, spec, class_order):
                writer.writerow({"method": key, "class_order": class_order,
                                 **{col: _fmt(rows[(key, class_order)][col]) for col in cols}})


def compare_reference(rows, spec: TableSpec, reference_csv: str | Path) -> list[str]:
    """Cell-by-cell diff against the paper transcription; returns the failures.

    Table 1's provenance notes expect third-decimal disagreements between the pipeline
    and the hand transcription, so every difference is printed and only those beyond
    ``spec.reference_tolerance`` count as failures.
    """
    cols = table_cols(spec.directions)
    failures: list[str] = []
    compared, max_diff = 0, 0.0
    with open(reference_csv, newline="", encoding="utf-8") as handle:
        for ref in csv.DictReader(handle):
            key = (ref["method"], ref["class_order"])
            row = rows.get(key)
            if row is None:
                failures.append(f"{key}: reference row has no measured counterpart")
                continue
            for col in cols:
                if not ref.get(col):
                    continue
                diff = abs(float(ref[col]) - row[col])
                compared += 1
                max_diff = max(max_diff, diff)
                if diff > spec.reference_tolerance:
                    failures.append(
                        f"{key} {col}: measured {row[col]:.4f}, reference "
                        f"{float(ref[col]):.4f} (diff {diff:.4f})")
    print(f"reference: {compared} cells compared, max abs diff {max_diff:.4f}, "
          f"tolerance {spec.reference_tolerance:g}, {len(failures)} beyond it")
    for failure in failures:
        print(f"  MISMATCH {failure}")
    return failures


def render_results_table(curves_csv: str | Path, out_dir: str | Path,
                         cfg=None) -> tuple[Path, ...]:
    """Write ``<basename>.csv``, ``.md`` and ``.tex`` under ``out_dir`` from one curves CSV."""
    curves = aggregate_phase_curves(curves_csv)
    spec = table_spec(cfg, curves)
    rows = collapse_table_rows(curves, spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{spec.basename}.csv"
    write_results_csv(rows, spec, csv_path)
    md_path = out_dir / f"{spec.basename}.md"
    md_path.write_text(render_results_md(rows, spec), encoding="utf-8")
    tex_path = out_dir / f"{spec.basename}.tex"
    tex_path.write_text(render_results_tex(rows, spec), encoding="utf-8")
    return (csv_path, md_path, tex_path)


# --- transfer table -----------------------------------------------------------------


def _report_metrics(cfg) -> tuple[str, ...]:
    declared = cfg.get("artifact.metrics.report_directions") if cfg is not None else None
    metrics = tuple(declared or (f"{direction}_r1" for direction in DIRECTIONS))
    for metric in metrics:
        direction = metric.rsplit("_", 1)[0]
        if direction not in DIRECTION_LABELS_TEX:
            raise ValueError(f"report metric {metric!r} names no known retrieval "
                             f"direction; expected one of {list(DIRECTIONS)}")
    return metrics


def load_transfer_metrics(path: str | Path,
                          metrics: Sequence[str]) -> dict[tuple[str, str, str], dict[str, float]]:
    """(method, variant, metric) -> {measure: value}, failing loudly on a gap.

    ``method`` and ``variant`` are the display strings of
    :data:`alphawise.naming.TRANSFER_DISPLAY`, which the summarize stage stamps into
    the CSV, so both stages resolve row identity through the same mapping.
    """
    table = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["method"], row["variant"], row["metric"])] = {
                measure: float(row[measure]) for measure in MEASURES
            }
    for method in TRANSFER_METHODS:
        for metric in metrics:
            for variant in TRANSFER_VARIANTS:
                if (method, variant, metric) not in table:
                    raise KeyError(f"missing row: {(method, variant, metric)}")
    return table


def _transfer_cells(base: float, alphawise: float, higher_is_better: int) -> tuple[str, str, str]:
    base_wins = (base >= alphawise) if higher_is_better > 0 else (base <= alphawise)
    # \mathbf inside math mode rather than \textbf around it: \textbf cannot reach
    # digits that are already in math mode, so the bold would silently not render.
    def fmt(value, bold):
        return rf"$\mathbf{{{value:+.4f}}}$" if bold else f"${value:+.4f}$"
    return fmt(base, base_wins), fmt(alphawise, not base_wins), f"${alphawise - base:+.4f}$"


def render_transfer_tex(table, metrics: Sequence[str]) -> str:
    """The transfer table as a LaTeX ``table*`` float, three rows per base method.

    The rows are Base, + AlphaWiSE and their difference, one column block per
    retrieval direction. ``table`` is :func:`load_transfer_metrics` output, which has
    already refused to return unless every (method, variant, metric) cell exists, so
    nothing is guarded here. Needs booktabs and ``multirow``.
    """
    columns = [(metric, measure) for metric in metrics for measure in MEASURES]
    spans, rules, start = [], [], 3
    for metric in metrics:
        label = DIRECTION_LABELS_TEX[metric.rsplit("_", 1)[0]]
        spans.append(rf"\multicolumn{{{len(MEASURES)}}}{{c}}{{{label}}}")
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(MEASURES) - 1}}}")
        start += len(MEASURES)

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        f"  \\caption{{{TRANSFER_CAPTION}}}",
        r"  \label{tab:cl-transfer}",
        f"  \\begin{{tabular}}{{ll{'c' * len(columns)}}}",
        r"    \toprule",
        "    & & " + " & ".join(spans) + r" \\",
        "    " + " ".join(rules),
        "    Base method & Variant & "
        + " & ".join(MEASURES[measure][0] for _, measure in columns) + r" \\",
        r"    \midrule",
    ]
    for index, method in enumerate(TRANSFER_METHODS):
        base_row, alpha_row, delta_row = [], [], []
        for metric, measure in columns:
            base = table[(method, "Base", metric)][measure]
            alphawise = table[(method, "+ AlphaWiSE", metric)][measure]
            cells = _transfer_cells(base, alphawise, MEASURES[measure][1])
            for row, cell in zip((base_row, alpha_row, delta_row), cells):
                row.append(cell)
        lines.extend([
            f"    \\multirow{{3}}{{*}}{{{method}}}",
            "      & Base        & " + " & ".join(base_row) + r" \\",
            "      & + AlphaWiSE & " + " & ".join(alpha_row) + r" \\",
            r"      & $\Delta$    & " + " & ".join(delta_row) + r" \\",
        ])
        if index != len(TRANSFER_METHODS) - 1:
            lines.append(r"    \midrule")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_transfer_table(matrix_csv: str | Path, out_dir: str | Path, cfg=None) -> Path:
    """Write the transfer table's ``.tex`` under ``out_dir`` from one measures CSV."""
    metrics = _report_metrics(cfg)
    out_path = Path(out_dir) / TRANSFER_BASENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_transfer_tex(load_transfer_metrics(matrix_csv, metrics),
                                            metrics), encoding="utf-8")
    return out_path


# --- command line -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Render one of the two tables from a CSV, one subcommand each.

    ``results`` reads the evaluator's curves CSV (``method``, ``class_order``,
    ``train_phase``, ``eval_phase`` and the six ``a2t_r1`` .. ``i2t_map`` cells) and
    the artifact config that declares the rows; ``--reference`` additionally diffs the
    rendered cells against a transcription CSV in the same schema as the written one.
    ``transfer`` reads the summarize stage's measures CSV (``method``, ``variant``,
    ``metric`` plus one column per measure).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="table", required=True)

    results = sub.add_parser("results", help="the per-class-order retrieval table")
    results.add_argument("--curves", type=Path, required=True, help="evaluator curves CSV")
    results.add_argument("--config", action="append", default=[], required=True,
                         help="artifact config; declares rows, window and layout")
    results.add_argument("--reference", type=Path,
                         help="transcription CSV to diff the rendered cells against")
    results.add_argument("--out-dir", type=Path, required=True)

    transfer = sub.add_parser("transfer", help="the backward/forward transfer table")
    transfer.add_argument("--matrix", type=Path, required=True,
                          help="summarized continual-learning measures CSV")
    transfer.add_argument("--config", action="append", default=[], required=True)
    transfer.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    from ..config import load as load_config
    cfg = load_config(*args.config)

    if args.table == "results":
        written = render_results_table(args.curves, args.out_dir, cfg)
        for path in written:
            print(f"wrote {path}")
        if args.reference:
            curves = aggregate_phase_curves(args.curves)
            spec = table_spec(cfg, curves)
            if compare_reference(collapse_table_rows(curves, spec), spec, args.reference):
                return 1
        return 0

    print(f"wrote {render_transfer_table(args.matrix, args.out_dir, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
