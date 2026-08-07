"""Generate a tiny synthetic corpus with the same on-disk shape as AudioSet.

AudioSet cannot be redistributed and takes days to download, so nothing in this repository
would be runnable out of the box without a stand-in. This writes real .jpg and .wav files,
a segment csv with the same three comment rows and column layout, and an ontology json, so
the loaders, the phase partition, the exemplar arithmetic and the retrieval evaluation all
exercise their real code paths. Only the content is fake.

The signal is deliberately learnable: each class gets a distinct colour and a distinct
audio frequency, so a model that trains at all will separate the classes and a broken
pipeline shows up as a flat metric rather than as noise indistinguishable from a bug.

The train and eval splits are disjoint: their clip ids come from non-overlapping ranges,
and every clip carries its own small seeded jitter around the class signature, so no eval
file duplicates a training file. A non-chance retrieval number therefore means the model
generalised within a class, not that it re-scored bytes it had memorised. The whole
corpus is a pure function of the seed: the same seed writes the same files.

    python -m alphawise.data.synthetic --out data/synthetic --seed 42
"""

import argparse
import csv
import json
import math
import random
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 44100
CLIP_SECONDS = 1.0  # real clips are 10 s; 1 s keeps the corpus small and loads the same way
IMAGE_SIZE = 64
CSV_HEADER_ROWS = 3
YT_ID_LENGTH = 11
YT_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
# Non-overlapping id ranges keep the two splits disjoint: an eval clip must never be a
# training clip under another filename convention.
SPLIT_ID_OFFSET = {"train": 0, "eval": 10**9}


def synthetic_yt_id(index: int) -> str:
    """A deterministic 11-character id, the length the real parser slices for."""
    digits, base = [], len(YT_ID_ALPHABET)
    for _ in range(YT_ID_LENGTH):
        digits.append(YT_ID_ALPHABET[index % base])
        index //= base
    return "".join(reversed(digits))


def write_png(path: Path, rgb: tuple[int, int, int], size: int = IMAGE_SIZE) -> None:
    """A flat colour image, written through Pillow so it decodes like a real photo."""
    from PIL import Image

    Image.new("RGB", (size, size), rgb).save(path, format="JPEG", quality=90)


def write_wav(path: Path, frequency: float, seconds: float = CLIP_SECONDS,
              phase: float = 0.0) -> None:
    """A pure tone, so each class is separable in the audio branch."""
    frames = int(SAMPLE_RATE * seconds)
    samples = (int(24000 * math.sin(2 * math.pi * frequency * n / SAMPLE_RATE + phase))
               for n in range(frames))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def class_signature(index: int, total: int) -> tuple[tuple[int, int, int], float]:
    """A colour and a frequency far enough apart that the classes are distinguishable."""
    hue = index / max(1, total)
    rgb = tuple(int(255 * (0.5 + 0.5 * math.cos(2 * math.pi * (hue + phase))))
                for phase in (0.0, 1 / 3, 2 / 3))
    return rgb, 220.0 * (2 ** (index / max(1, total // 2 or 1)))


def jitter_signature(rgb: tuple[int, int, int], frequency: float,
                     rng: random.Random) -> tuple[tuple[int, int, int], float, float]:
    """One clip's own variation around its class signature: colour, frequency, phase.

    The jitter is small against the between-class spacing (adjacent hues are 16 RGB
    steps at 16 classes, adjacent frequencies about 9 percent apart), so classes stay
    separable while no two clips, and in particular no train and eval clip, are ever
    byte-identical.
    """
    jittered_rgb = tuple(min(255, max(0, channel + rng.randint(-10, 10))) for channel in rgb)
    return jittered_rgb, frequency * 2 ** rng.uniform(-0.02, 0.02), rng.uniform(0, 2 * math.pi)


def generate(out: Path, *, classes: int, per_class: int, seed: int, split: str) -> dict:
    """Write one split's corpus under ``out`` and return a summary of what was made.

    The layout is the one the loaders expect: ``images/<ytid>.jpg``, ``audio/Y<ytid>.wav``,
    a ``segments.csv`` whose first three lines are comments, and a flat ``ontology.json``.
    Every clip carries exactly one label, so the phase partition sees ``classes`` disjoint
    groups of ``per_class`` clips and the exemplar arithmetic has no multi-label case to
    resolve, which the real corpus does have.

    Deterministic under ``seed``: all per-clip variation is drawn from one private
    ``random.Random(seed)`` in a fixed order, so the same call writes the same bytes.
    """
    rng = random.Random(seed)
    images, wavs = out / "images", out / "audio"
    for directory in (images, wavs):
        directory.mkdir(parents=True, exist_ok=True)

    label_ids = [f"/synth/{index:03d}" for index in range(classes)]
    ontology = [{"id": label, "name": f"synthetic class {index:03d}"}
                for index, label in enumerate(label_ids)]
    (out / "ontology.json").write_text(json.dumps(ontology, indent=1))

    id_offset = SPLIT_ID_OFFSET[split]
    rows, made = [], 0
    for class_index, label in enumerate(label_ids):
        rgb, frequency = class_signature(class_index, classes)
        for _ in range(per_class):
            yt_id = synthetic_yt_id(id_offset + made)
            clip_rgb, clip_frequency, clip_phase = jitter_signature(rgb, frequency, rng)
            write_png(images / f"{yt_id}.jpg", clip_rgb)
            # Training wavs carry a Y prefix and evaluation wavs do not, and the two splits
            # are parsed by different id rules because of it. Reproducing the difference is
            # the point: a corpus that named both alike would not exercise the lookup.
            prefix = "Y" if split == "train" else ""
            write_wav(wavs / f"{prefix}{yt_id}.wav", clip_frequency, phase=clip_phase)
            rows.append([yt_id, "0.000", "10.000", label])
            made += 1

    with (out / "segments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for line in ("# Synthetic segments csv, generated by alphawise.data.synthetic",
                     f"# num_ytids={made}, num_unique_labels={classes}",
                     "# YTID, start_seconds, end_seconds, positive_labels"):
            handle.write(line + "\n")
        writer.writerows(rows)

    return {"clips": made, "classes": classes, "root": str(out)}


def main(argv=None) -> int:
    """Generate both splits, then print the environment that points the loaders at them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/synthetic", type=Path)
    parser.add_argument("--classes", type=int, default=16,
                        help="must divide into phases; 16 gives 8 phases of 2")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Train and evaluation are separate corpora upstream, drawn from disjoint clips and
    # named by different conventions. One corpus serving both would hide the id lookup,
    # and shared clips would let evaluation re-score what training memorised.
    for offset, split in enumerate(("train", "eval")):
        summary = generate(args.out / split, classes=args.classes,
                           per_class=args.per_class, seed=args.seed + offset, split=split)
        print(f"{split}: {summary['clips']} clips over {summary['classes']} classes "
              f"in {summary['root']}")

    print("\npoint the loaders at it with:")
    for split in ("TRAIN", "EVAL"):
        root = args.out / split.lower()
        print(f"  export AUDIOSET_{split}_IMG={root}/images")
        print(f"  export AUDIOSET_{split}_WAV={root}/audio")
        print(f"  export AUDIOSET_{split}_CSV={root}/segments.csv")
    print(f"  export AUDIOSET_ONT={args.out}/train/ontology.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
