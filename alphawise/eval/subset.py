"""The small per-class evaluation subset the embedding-space analyses draw on.

This is a refactor of the subset drawing in ``_import/triclip/scripts/create_tsne_it.py``
(``build_eval_subset``), ported so nothing under ``_import/`` is executed. Given the raw
media roots it pairs every eval image with its wav by the 11-character YouTube id
(``basename[:11]``, the eval-split naming convention), buckets the pairs by ontology
class name, and draws a capped sample per requested class plus one background pair per
class that was not requested.

Two properties of the ancestor are deliberately changed, both recorded in the Figure 4
provenance notes:

* the ancestor's ``seed`` argument never reached the per-class draw, which shuffled with
  the unseeded global ``random`` module, so no two runs selected the same points. Here
  ``random.seed(seed)`` is called immediately before that draw (default 42);
* a requested class the pool cannot supply raises instead of being warned away: silently
  dropping one shifts every later class onto the wrong colour.

The module is pure path and csv work: it needs neither torch nor the audio stack, so a
subset can be drawn (or a recorded one replayed) on any machine that sees the media.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Sequence, Tuple

#: One usable sample: its image path and its wav path, matched by YouTube id.
Pair = Tuple[str, str]

YT_ID_LENGTH = 11
CSV_HEADER_ROWS = 3
DEFAULT_SEED = 42

SELECTIONS = ("explicit",)


@dataclass(frozen=True)
class EvalSubset:
    """The drawn subset. ``classes`` keeps the requested order, which fixes the palette."""

    classes: Tuple[str, ...]
    samples: Dict[str, Tuple[Pair, ...]]  # class name -> its drawn (image, wav) pairs
    background: Tuple[Pair, ...]  # one pair per class outside ``classes``


def _class_pools(img_root, wav_root, meta_csv, ont_path) -> Dict[str, list]:
    """Every usable (image, wav) pair of the eval split, bucketed by class name.

    A pair is usable when the image's YouTube id has both a wav and a metadata row; a
    multi-label clip lands in every one of its classes, exactly as the ancestor bucketed.
    """
    with open(ont_path) as handle:
        label_to_name = {entry["id"]: entry["name"] for entry in json.load(handle)}

    id_to_labels: Dict[str, list] = {}
    with open(meta_csv) as handle:
        reader = csv.reader(handle)
        for _ in range(CSV_HEADER_ROWS):
            next(reader)
        for row in reader:
            id_to_labels[row[0]] = [field.strip().replace('"', "") for field in row[3:]]

    id_to_wav = {Path(path).name[:YT_ID_LENGTH]: path
                 for path in glob(str(Path(wav_root) / "*.wav"))}

    pools: Dict[str, list] = defaultdict(list)
    for image_path in sorted(glob(str(Path(img_root) / "*.jpg"))):
        yt_id = Path(image_path).name[:YT_ID_LENGTH]
        if yt_id not in id_to_labels or yt_id not in id_to_wav:
            continue
        for label_id in id_to_labels[yt_id]:
            if label_id in label_to_name:
                pools[label_to_name[label_id]].append((image_path, id_to_wav[yt_id]))
    if not pools:
        raise FileNotFoundError(
            f"no usable (image, wav) pairs under {img_root} / {wav_root}; "
            "point the roots at the eval split"
        )
    return pools


def _read_manifest(path: Path) -> EvalSubset:
    recorded = json.loads(path.read_text())
    return EvalSubset(
        classes=tuple(recorded["classes"]),
        samples={cls: tuple(tuple(pair) for pair in pairs)
                 for cls, pairs in recorded["samples"].items()},
        background=tuple(tuple(pair) for pair in recorded["background"]),
    )


def _write_manifest(path: Path, subset: EvalSubset, seed: int,
                    samples_per_class: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "seed": seed,
        "samples_per_class": samples_per_class,
        "classes": list(subset.classes),
        "samples": {cls: [list(pair) for pair in pairs]
                    for cls, pairs in subset.samples.items()},
        "background": [list(pair) for pair in subset.background],
    }, indent=1))


def build_eval_subset(
    img_root: str | Path,
    wav_root: str | Path,
    meta_csv: str | Path,
    ont_path: str | Path,
    *,
    samples_per_class: int = 25,
    selection: str = "explicit",
    explicit_classes: Sequence[str] | None = None,
    with_background: bool = False,
    seed: int = DEFAULT_SEED,
    manifest: str | Path | None = None,
) -> EvalSubset:
    """Draw (or replay) the per-class evaluation subset.

    ``samples_per_class`` is a cap, not a count: a class supplies at most that many
    pairs and everything it has below the cap. ``selection`` names how classes are
    chosen; the release draws only ``"explicit"``, the caller's ``explicit_classes``
    in their given order. A requested class must clear the ancestor's eligibility
    floor, ``max(samples_per_class // 3, 3)`` usable pairs.

    With ``with_background``, one pair is drawn per unselected class that has any
    usable pair, from a dedicated ``random.Random(seed + 1)`` stream over the classes
    in sorted order, exactly as the ancestor drew its grey background points.

    ``manifest`` decouples the draw from the media: an existing manifest is replayed
    verbatim (nothing is globbed, nothing is seeded), a missing one is written after
    the draw so the run that produced a figure leaves its subset behind.
    """
    if manifest is not None and Path(manifest).is_file():
        return _read_manifest(Path(manifest))

    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}")
    if not explicit_classes:
        raise ValueError("selection 'explicit' needs a non-empty explicit_classes list")

    pools = _class_pools(img_root, wav_root, meta_csv, ont_path)

    floor = max(samples_per_class // 3, 3)
    short = [f"{cls} ({len(pools.get(cls, ()))} < {floor})"
             for cls in explicit_classes if len(pools.get(cls, ())) < floor]
    if short:
        raise ValueError(
            f"requested class(es) below the {floor}-pair eligibility floor: {short}. "
            "Dropping one silently would shift every later class onto the wrong colour."
        )

    # The fix for the ancestor bug: its seed argument never reached this draw, so the
    # global RNG must be seeded immediately before the shuffles it feeds.
    random.seed(seed)
    samples = {}
    for cls in explicit_classes:
        pairs = list(pools[cls])
        random.shuffle(pairs)
        samples[cls] = tuple(pairs[:samples_per_class])

    background: Tuple[Pair, ...] = ()
    if with_background:
        selected = set(explicit_classes)
        background_rng = random.Random(seed + 1)
        background = tuple(background_rng.choice(pools[cls])
                           for cls in sorted(pools) if cls not in selected)

    subset = EvalSubset(classes=tuple(explicit_classes), samples=samples,
                        background=background)
    if manifest is not None:
        _write_manifest(Path(manifest), subset, seed, samples_per_class)
    return subset
