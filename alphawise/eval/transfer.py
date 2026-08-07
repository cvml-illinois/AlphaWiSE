"""Continual-learning transfer measures (Table 3), derived from one curves CSV.

The evaluation runner scores every train-phase checkpoint on every phase of the
protocol, so its curves CSV already holds the full transfer matrix R[k][j]: the phase-k
checkpoint's score on phase j, over the same candidate pool as every other published
number. This module only rearranges those cells; it never loads a model.

Definitions are those of the ancestor tree's ``summarize_transfer_matrix.py``,
unchanged, with T the last task phase:

    Avg    = (1/T)     sum_{j=1..T}   R[T][j]
    MFT    = (1/T)     sum_{j=1..T}   R[j][j]
    BWT    = (1/(T-1)) sum_{j=1..T-1} (R[T][j] - R[j][j])
    Forget = (1/(T-1)) sum_{j=1..T-1} (max_{k in [j, T-1]} R[k][j] - R[T][j])
    FWT    = (1/(T-1)) sum_{j=2..T}   (R[j-1][j] - b_j)

b_j is the shared phase-0 bootstrap's score on phase j, not a random-init baseline:
every model here starts from a pretrained CLIP, so chance level would make FWT
meaningless. Because every trajectory starts from the same phase-0 weights, the
baseline is read once, from the bootstrap method's ``train_phase == 0`` rows, and
applied to every method. A per-method baseline would be the same numbers where a
method carries a phase 0 of its own and a hard failure where it does not, which is
most of Table 3's rows. Forgetting maxes over k <= T-1, excluding the final model,
which is the Chaudhry et al. convention; including it would force the metric
non-negative.

Two subcommands. ``matrix`` enforces the published protocol (every mapped method
complete over task phases 1..7, bootstrap baseline present) and writes the measures
CSV the table renderer reads; ``validate`` checks any curves file for internal
consistency (rectangular grid, no duplicate cells, one pool) without demanding the
full protocol, so a smoke-scale file can be validated too.

    python -m alphawise.eval.transfer matrix --curves <csv> --out <csv>
    python -m alphawise.eval.transfer validate --curves <csv>
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from ..naming import TRANSFER_DISPLAY, metric_cols

TASK_PHASES = (1, 2, 3, 4, 5, 6, 7)
BOOTSTRAP_PHASE = 0
# Columns of a complete matrix row, bootstrap included: continual.num_phases of them.
EVAL_PHASES = (0, 1, 2, 3, 4, 5, 6, 7)
MEASURES = ("avg", "mft", "bwt", "forget", "fwt")
#: The registry key whose train_phase == 0 rows are the shared bootstrap baseline.
BOOTSTRAP_METHOD = "normal"

Matrix = Dict[Tuple[int, int], Dict[str, float]]


def read_curves(path: Path) -> List[dict]:
    """Rows of one curves CSV, in file order, with every field left as a string."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix_from_rows(rows: Iterable[Mapping[str, object]]) -> Matrix:
    """Long-form evaluation rows to {(train_phase, eval_phase): {metric: value}}."""
    matrix: Matrix = {}
    for row in rows:
        cell = (int(row["train_phase"]), int(row["eval_phase"]))
        if cell in matrix:
            raise ValueError(f"duplicate matrix cell {cell}")
        matrix[cell] = {m: float(row[m]) for m in metric_cols()}
    return matrix


def matrices_by_method(rows: Iterable[Mapping[str, object]]) -> Dict[str, Matrix]:
    """One transfer matrix per method, in the curves file's first-seen order."""
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    return {method: matrix_from_rows(method_rows) for method, method_rows in grouped.items()}


def curves_pool(rows: Iterable[Mapping[str, object]]) -> str:
    """The single candidate pool every row was scored under, raising on a mixture.

    The pool changes the numbers, so measures derived across two pools would compare
    scores that are not comparable.
    """
    pools = {str(row["pool"]) for row in rows}
    if len(pools) != 1:
        raise ValueError(f"curves file mixes candidate pools {sorted(pools)}; "
                         f"evaluate every method under one pool")
    return pools.pop()


def baseline_row(matrix: Matrix, phase: int = BOOTSTRAP_PHASE) -> Dict[int, Dict[str, float]]:
    """The FWT baseline b: {eval_phase: {metric: value}} taken from one train phase."""
    return {j: scores for (k, j), scores in matrix.items() if k == phase}


def validate(matrix: Matrix, task_phases: Iterable[int] = TASK_PHASES,
             eval_phases: Iterable[int] = EVAL_PHASES) -> None:
    """Raise unless every (task phase, eval phase) cell was evaluated."""
    missing = sorted({(k, j) for k in task_phases for j in eval_phases} - set(matrix))
    if missing:
        raise ValueError(f"missing matrix cells {missing}")


def cl_metrics(matrix: Matrix, metric: str, baseline: Mapping[int, Mapping[str, float]],
               task_phases: Iterable[int] = TASK_PHASES) -> Dict[str, float]:
    """The five continual-learning measures for one retrieval metric."""
    task_phases = tuple(task_phases)
    final = task_phases[-1]
    r = lambda k, j: matrix[(k, j)][metric]

    avg = sum(r(final, j) for j in task_phases) / len(task_phases)
    mft = sum(r(j, j) for j in task_phases) / len(task_phases)
    earlier = task_phases[:-1]
    bwt = sum(r(final, j) - r(j, j) for j in earlier) / len(earlier)
    forget = sum(
        max(r(k, j) for k in task_phases if j <= k <= final - 1) - r(final, j)
        for j in earlier
    ) / len(earlier)
    later = task_phases[1:]
    fwt = sum(r(j - 1, j) - baseline[j][metric] for j in later) / len(later)
    return {"avg": avg, "mft": mft, "bwt": bwt, "forget": forget, "fwt": fwt}


def shared_baseline(matrices: Mapping[str, Matrix], bootstrap_method: str,
                    curves_path: Path) -> Dict[int, Dict[str, float]]:
    """The one FWT baseline every method is measured against, or a named error."""
    if bootstrap_method not in matrices:
        raise ValueError(
            f"missing FWT baseline: no {bootstrap_method!r} rows in {curves_path}. "
            f"Evaluate the bootstrap method (its registry entry declares phase 0) into "
            f"the same curves file."
        )
    baseline = baseline_row(matrices[bootstrap_method])
    absent = [j for j in TASK_PHASES[1:] if j not in baseline]
    if absent:
        raise ValueError(
            f"missing FWT baseline: {bootstrap_method!r} has no train_phase="
            f"{BOOTSTRAP_PHASE} row for eval phases {absent} in {curves_path}"
        )
    return baseline


def summarize_curves(curves_path: Path, out_path: Path, *,
                     bootstrap_method: str = BOOTSTRAP_METHOD) -> Path:
    """The Table 3 measures CSV: one row per (display method, variant, metric).

    Strict about the protocol: every Table 3 method present in the curves file must
    cover the full task-phase grid, and the shared bootstrap baseline must exist.
    Methods outside the Table 3 mapping are announced and skipped, so the same curves
    file can carry the whole results table.
    """
    rows = read_curves(curves_path)
    if not rows:
        raise ValueError(f"{curves_path} holds no evaluation rows")
    curves_pool(rows)
    matrices = matrices_by_method(rows)
    baseline = shared_baseline(matrices, bootstrap_method, curves_path)

    skipped = [m for m in matrices if m not in TRANSFER_DISPLAY and m != bootstrap_method]
    if skipped:
        print(f"skipping methods outside the transfer table: {skipped}", flush=True)

    out_rows = []
    for key, (display_method, variant) in TRANSFER_DISPLAY.items():
        if key not in matrices:
            continue
        matrix = matrices[key]
        try:
            validate(matrix)
        except ValueError as error:
            raise ValueError(f"{key}: {error}") from None
        for metric in metric_cols():
            out_rows.append({
                "method": display_method, "variant": variant, "metric": metric,
                **{k: f"{v:.6f}" for k, v in cl_metrics(matrix, metric, baseline).items()},
            })
    if not out_rows:
        raise ValueError(f"{curves_path} holds no transfer-table method "
                         f"(known: {sorted(TRANSFER_DISPLAY)})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "variant", "metric", *MEASURES])
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path


def validate_curves(curves_path: Path, *,
                    bootstrap_method: str = BOOTSTRAP_METHOD) -> List[str]:
    """Internal-consistency problems of one curves file, empty when it is sound.

    Checks what any curves file must satisfy whatever its scale: one candidate pool,
    no duplicate cells, and a rectangular grid (every method's every train phase scored
    on the same eval phases). Protocol coverage is deliberately not checked here; a
    smoke-scale file is consistent without spanning phases 1..7, and ``matrix`` is the
    strict gate for the published protocol.
    """
    rows = read_curves(curves_path)
    if not rows:
        return [f"{curves_path} holds no evaluation rows"]

    problems: List[str] = []
    try:
        curves_pool(rows)
    except ValueError as error:
        problems.append(str(error))

    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    matrices: Dict[str, Matrix] = {}
    for method, method_rows in grouped.items():
        try:
            matrices[method] = matrix_from_rows(method_rows)
        except ValueError as error:
            problems.append(f"{method}: {error}")

    eval_phase_sets = {
        method: {j for (_, j) in matrix} for method, matrix in matrices.items()
    }
    if len(set(map(frozenset, eval_phase_sets.values()))) > 1:
        problems.append(f"methods disagree on eval phases: "
                        f"{ {m: sorted(js) for m, js in eval_phase_sets.items()} }")
    for method, matrix in matrices.items():
        train_phases = sorted({k for (k, _) in matrix})
        expected = {(k, j) for k in train_phases for j in eval_phase_sets[method]}
        missing = sorted(expected - set(matrix))
        if missing:
            problems.append(f"{method}: incomplete grid, missing cells {missing}")

    # --bootstrap-method none declares a protocol with no bootstrap (the smoke corpus),
    # which forgoes FWT; only then is a missing baseline row not a finding.
    if bootstrap_method != "none" and (
            bootstrap_method not in matrices or not baseline_row(matrices[bootstrap_method])):
        problems.append(f"{bootstrap_method}: no train_phase={BOOTSTRAP_PHASE} row, "
                        f"so FWT has no baseline")
    return problems


def main(argv=None) -> int:
    """Run one of the two subcommands above. Returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser(
        "matrix", help="derive Avg, MFT, BWT, Forgetting and FWT from a curves CSV")
    matrix.add_argument("--curves", required=True, type=Path,
                        help="curves CSV written by alphawise.eval.runner")
    matrix.add_argument("--out", required=True, type=Path)
    matrix.add_argument("--bootstrap-method", default=BOOTSTRAP_METHOD,
                        help="method whose train_phase=0 rows are the shared FWT baseline")

    check = sub.add_parser("validate", help="check a curves CSV for internal consistency")
    check.add_argument("--curves", required=True, type=Path)
    check.add_argument("--bootstrap-method", default=BOOTSTRAP_METHOD)

    args = parser.parse_args(argv)
    if args.command == "matrix":
        out = summarize_curves(args.curves, args.out,
                               bootstrap_method=args.bootstrap_method)
        print(f"wrote {out}")
        return 0

    problems = validate_curves(args.curves, bootstrap_method=args.bootstrap_method)
    for problem in problems:
        print(f"INVALID: {problem}", flush=True)
    if problems:
        return 1
    print(f"{args.curves} is internally consistent", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
