"""Evaluate registered checkpoints over the full phase grid and write one curves CSV.

Descends from the ancestor tree's ``eval_table7.py`` (``_load_model``,
``_eval_checkpoint``), ``eval_840.py`` (checkpoint resolution) and
``eval_transfer_matrix.py`` (one row per evaluated cell).

The output is long form, one row per (method, train phase, eval phase) cell, and every
train phase is scored on every phase of the protocol. The cells with
``eval_phase <= train_phase`` are the retention curve the results tables aggregate; the
cells above the diagonal are the forward-transfer scores ``alphawise.eval.transfer``
needs. One sweep therefore serves both consumers, at the price of the above-diagonal
cells a retention-only reading would skip.

``--method`` is repeatable and appends every method's rows into the same CSV, so a whole
table is one file and one command. ``--class-order`` selects the registry section (and
with it the phase-0 bootstrap lineage); ``--pool`` names the evaluation candidate pool
and is recorded in every row, because it changed a published ranking once and must never
have to be reconstructed from context.

    python -m alphawise.eval.runner --config configs/artifacts/main_comparison.yaml \\
        --method normal --method ewc --class-order forward --pool per_phase \\
        --out results/main_comparison/phase_curves_840.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..config import Config, load as load_config
from ..data.audioset import CLASS_ORDERS, POOLS, ContinualAudioSet, collate_batch
from ..data.roots import DataRoots
from ..models.factory import build_audioclip
from ..naming import metric_cols
from ..utils.checkpoint import load_blob, model_state_of, strip_prefixes
from ..utils.paths import highest_epoch_checkpoint
from ..utils.seed import seed_everything, worker_init_fn
from . import retrieval

DEFAULT_SEED = 42

CSV_FIELDS = ("method", "class_order", "pool", "train_phase", "eval_phase") + tuple(metric_cols())

#: ``retrieval.evaluate_phase`` may still key its cells by the pre-rename short names.
#: Cells are resolved by name, never by position: the legacy column order (at, it, ia)
#: and the contract order (a2t, i2a, i2t) disagree, so a positional map would silently
#: swap the two image directions.
_LEGACY_DIRECTION = {"a2t": "at", "i2a": "ia", "i2t": "it"}


class EvalDataTooSmall(RuntimeError):
    """The eval split is smaller than ``eval.min_eval_samples``."""


def _cell_value(cell: Mapping[str, float], column: str) -> float:
    if column in cell:
        return cell[column]
    direction, _, metric = column.partition("_")
    return cell[f"{_LEGACY_DIRECTION[direction]}_{metric}"]


@dataclass(frozen=True)
class EvalSettings:
    """Everything the evaluation loop reads out of the config, resolved once per process.

    ``pool`` fixes the candidate pool the metrics are computed over: "per_phase" scores
    each phase against its own 79 classes (the published protocol), "cumulative" against
    every class seen up to it. ``num_phases`` is the protocol's full phase count rather
    than the subset a given run evaluates, because ``min_eval_samples`` is calibrated
    against the whole split.
    """

    roots: DataRoots
    batch_size: int
    num_workers: int
    pool: str
    min_eval_samples: int
    num_phases: int
    classes_per_phase: int
    wav_id_mode: Optional[str]
    seed: Optional[int]

    @classmethod
    def from_config(cls, cfg: Config, *, pool: Optional[str] = None,
                    batch_size: Optional[int] = None, num_workers: Optional[int] = None,
                    seed: Optional[int] = DEFAULT_SEED) -> "EvalSettings":
        """Read the ``eval.*`` keys plus ``continual.*`` from a resolved config.

        The keyword arguments are the CLI overrides; None means "keep the config value".
        """
        resolved_pool = cfg["eval.pool"] if pool is None else pool
        if resolved_pool not in POOLS:
            raise ValueError(f"eval.pool must be one of {POOLS}, got {resolved_pool!r}")
        return cls(
            roots=DataRoots.from_config(cfg, "eval"),
            batch_size=cfg["eval.batch_size"] if batch_size is None else batch_size,
            num_workers=cfg["eval.num_workers"] if num_workers is None else num_workers,
            pool=resolved_pool,
            min_eval_samples=cfg["eval.min_eval_samples"],
            num_phases=cfg["continual.num_phases"],
            classes_per_phase=cfg["continual.classes_per_phase"],
            wav_id_mode=cfg.get("data.wav_id_mode"),
            seed=seed,
        )


@dataclass(frozen=True)
class CheckpointEntry:
    """One registry row: where a method's run lives and which phases it has weights for.

    ``root`` holds a ``phase_<k>`` subdirectory per trained phase. ``filename`` is None
    when the epoch to read is not fixed, in which case :func:`resolve_checkpoint` takes the
    highest epoch present. ``phases`` is what the registry declares, not what is on disk,
    so a method with no bootstrap of its own simply omits phase 0.
    """

    root: Path
    filename: Optional[str]
    phases: Sequence[int]


def registry_entry(cfg: Config, method: str, class_order: str) -> CheckpointEntry:
    """The registry row backing one method name in one class order.

    This is the only place a method name becomes a filesystem location, and it never
    touches the disk: an entry may name a run that was not downloaded. Missing keys raise
    ``KeyError`` listing what the config does offer.
    """
    registry = cfg["checkpoints.registry"]
    if class_order not in registry:
        raise KeyError(f"no checkpoints.registry.{class_order}: {sorted(registry)}")
    entries = registry[class_order]
    if method not in entries:
        raise KeyError(f"{method!r} not in checkpoints.registry.{class_order}: {sorted(entries)}")
    entry = entries[method]
    return CheckpointEntry(
        root=Path(entry["root"]),
        filename=entry.get("filename"),
        phases=tuple(entry["phases"]),
    )


def resolve_checkpoint(entry: CheckpointEntry, phase: int) -> Optional[Path]:
    """Path to the phase's checkpoint, or None when the run does not have one.

    A null filename means "take the highest epoch present": epoch coverage varies
    between the released runs.
    """
    phase_dir = entry.root / f"phase_{phase}"
    if entry.filename is not None:
        candidate = phase_dir / entry.filename
        return candidate if candidate.is_file() else None
    return highest_epoch_checkpoint(phase_dir)


def load_checkpoint(path: Path, device: torch.device):
    """A registered checkpoint as an evaluation-ready backbone.

    Loaded non-strictly, because the released runs differ in which buffers they carry,
    and then checked for wholesale mismatch: a wrapper prefix that survived stripping
    used to leave the backbone untrained while every key silently went missing.
    """
    model = build_audioclip()
    state_dict = strip_prefixes(model_state_of(load_blob(path)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys (first: {unexpected[:3]})", flush=True)
    # Fail loud rather than silently evaluate an untrained model: a correct
    # checkpoint populates essentially the whole model.
    n_params = sum(1 for _ in model.state_dict())
    if len(missing) > 0.1 * n_params:
        raise RuntimeError(
            f"{path}: {len(missing)}/{n_params} params missing after load "
            f"(first missing: {missing[:3]}). Checkpoint key format likely mismatched."
        )
    model.to(device).eval()
    return model


def _phase_dataset(settings: EvalSettings, class_order: str, phase: int) -> ContinualAudioSet:
    """One phase's evaluation split, over the candidate pool the protocol prescribes.

    ``classes_per_phase`` has to be the value the checkpoints were trained under. It is
    threaded through rather than defaulted, because a mismatch does not raise: it would
    quietly score a model against a different partition than it learned.
    """
    return ContinualAudioSet(
        settings.roots, phase, split="eval", class_order=class_order,
        pool=settings.pool,
        classes_per_phase=settings.classes_per_phase,
        wav_id_mode=settings.wav_id_mode,
    )


def assert_sample_floor(datasets: Mapping[int, ContinualAudioSet], class_order: str,
                        minimum: int) -> int:
    """Tripwire: too few eval samples means the YouTube-ID parser dropped data.

    Checked over the full phase set rather than the subset a given run evaluates, so
    the configured threshold keeps the meaning it was calibrated with.
    """
    total = sum(len(ds) for ds in datasets.values())
    if total < minimum:
        raise EvalDataTooSmall(
            f"eval split has {total} samples across {len(datasets)} {class_order} phases, "
            f"expected at least {minimum} (eval.min_eval_samples)"
        )
    return total


def phase_datasets(settings: EvalSettings, class_order: str) -> Dict[int, ContinualAudioSet]:
    """Eval datasets for every phase of the protocol, keyed by phase index.

    All ``num_phases`` are built even when only a few are scored, so the sample-floor
    tripwire sees the whole split before any checkpoint is loaded. Built once per process
    and shared by every method, since the pool depends only on class order and phase.
    """
    datasets = {k: _phase_dataset(settings, class_order, k) for k in range(settings.num_phases)}
    total = assert_sample_floor(datasets, class_order, settings.min_eval_samples)
    print(f"tripwire OK: {total} eval samples ({class_order}, pool={settings.pool})", flush=True)
    return datasets


def _phase_loader(dataset: ContinualAudioSet, settings: EvalSettings) -> DataLoader:
    return DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=settings.num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=(settings.num_workers > 0),
        # The factory must be called: it returns the per-worker callable (or None).
        worker_init_fn=worker_init_fn(settings.seed),
    )


def evaluate_checkpoint(model, datasets: Mapping[int, ContinualAudioSet],
                        eval_phases: Iterable[int], settings: EvalSettings,
                        device: torch.device) -> Dict[int, Dict[str, float]]:
    """All six metrics for each requested eval phase, keyed by phase."""
    scores = {}
    for phase in eval_phases:
        dataset = datasets[phase]
        print(f"  phase {phase}: {len(dataset)} samples, {len(dataset.text_classes)} classes", flush=True)
        scores[phase] = retrieval.evaluate_phase(
            model, _phase_loader(dataset, settings), dataset, device,
        )
    return scores


def evaluate_entry(entry: CheckpointEntry, method: str,
                   datasets: Mapping[int, ContinualAudioSet],
                   train_phases: Sequence[int], settings: EvalSettings,
                   device: torch.device) -> List[dict]:
    """Long-form rows for the full (train phase, eval phase) grid of one registry entry.

    A registry-declared train phase whose checkpoint is absent raises rather than being
    skipped: a partial grid would flow silently into the tables and the transfer measures.
    """
    eval_phases = sorted(datasets)
    rows = []
    for train_phase in train_phases:
        checkpoint = resolve_checkpoint(entry, train_phase)
        if checkpoint is None:
            raise FileNotFoundError(
                f"{method}: no phase-{train_phase} checkpoint under {entry.root}; the "
                f"registry declares phases {list(entry.phases)}. Fetch the run or narrow "
                f"--phases."
            )
        print(f"{method}: train_phase={train_phase} <- {checkpoint.name}", flush=True)
        model = load_checkpoint(checkpoint, device)
        scores = evaluate_checkpoint(model, datasets, eval_phases, settings, device)
        for eval_phase, cell in scores.items():
            rows.append({
                "method": method,
                "pool": settings.pool,
                "train_phase": train_phase,
                "eval_phase": eval_phase,
                **{col: f"{_cell_value(cell, col):.6f}" for col in metric_cols()},
            })
    return rows


def write_csv(rows: Iterable[dict], out_path: Path) -> None:
    """Write long-form rows to ``out_path``, creating parent directories.

    The header is always the full :data:`CSV_FIELDS`, so a sweep that produced no rows
    still leaves a file the summarizers can read.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _phase_list(value: Optional[str]) -> Optional[tuple]:
    return None if value is None else tuple(int(p) for p in value.split(","))


def parse_args(argv=None) -> argparse.Namespace:
    """Command line for any number of methods in one class order.

    ``--phases`` names the train phases and defaults to each registry entry's declared
    phases; the eval phases are always the protocol's full range, because the transfer
    measures need the cells above the diagonal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[],
                        help="config overlay, repeatable; must resolve checkpoints.registry")
    parser.add_argument("--method", action="append", required=True,
                        help="key in checkpoints.registry.<class_order>, repeatable; "
                             "every method's rows land in the same CSV")
    parser.add_argument("--class-order", required=True, choices=CLASS_ORDERS)
    parser.add_argument("--phases", default=None,
                        help="comma list of train phases; default is each registry "
                             "entry's phases")
    parser.add_argument("--pool", default=None, choices=POOLS,
                        help="evaluation candidate pool; default is the config's eval.pool")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="overrides eval.batch_size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="overrides eval.num_workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seeds every RNG surface of the process. Evaluation is "
                             "deterministic by construction (no shuffling, centred "
                             "crops), so this pins the residual surface rather than "
                             "changing any published number")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Evaluate every requested method over the phase grid and write one curves CSV."""
    args = parse_args(argv)
    cfg = load_config(*args.config)
    settings = EvalSettings.from_config(
        cfg, pool=args.pool, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"methods={args.method} class_order={args.class_order} "
          f"pool={settings.pool} device={device}", flush=True)

    datasets = phase_datasets(settings, args.class_order)
    requested_phases = _phase_list(args.phases)
    rows: List[dict] = []
    for method in args.method:
        entry = registry_entry(cfg, method, args.class_order)
        train_phases = requested_phases if requested_phases is not None else entry.phases
        rows.extend(evaluate_entry(entry, method, datasets, train_phases, settings, device))
    # Class order is part of a row's identity: one CSV may hold both orders, and the
    # aggregation downstream keys on it.
    for row in rows:
        row["class_order"] = args.class_order
    write_csv(rows, args.out)
    print(f"wrote {args.out} ({len(rows)} cells, {len(args.method)} methods)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
