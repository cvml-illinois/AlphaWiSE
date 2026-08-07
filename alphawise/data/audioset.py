"""AudioSet as a continual-learning corpus.

One dataset covers every configuration the development tree spread over ten
near-duplicate classes: forward or reverse class order, train or eval split, any phase
index, any exemplar budget, exemplars on or off. The exemplar budget is a plain number,
so 840 (the paper setting) and 8400 are the same code path with a different value.

Three orderings are load-bearing and must not be "tidied":

* images and wavs are enumerated with `sorted(glob(...))`, and the unbalanced exemplar
  path slices each previous phase's list in exactly that order;
* the balanced path sorts every label bucket with `sort(key=str)` before slicing;
* exemplars are appended after the current phase's samples, and `length` truncates the
  combined list.
"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Container, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image

from ..utils.transforms import (
    RandomCrop,
    RandomFlip,
    RandomNoise,
    RandomPadding,
    RandomScale,
    ToTensor1D,
)
from .roots import SPLITS, DataRoots

CLASSES_PER_PHASE = 79
CSV_HEADER_ROWS = 3
YT_ID_LENGTH = 11
SAMPLE_RATE = 44100
WAV_SCALE = 32768.0
WAV_OUT_LEN = 220500
CLASS_ORDERS = ("forward", "reverse")
# The evaluation candidate pool. "per_phase" scores each phase against its own ~79
# classes, the published protocol; "cumulative" scores against every class seen so far,
# a different protocol that gives different numbers.
POOLS = ("per_phase", "cumulative")

# A sample: image path, wav path, the names of its labels inside the allowed pool.
Triple = Tuple[str, str, List[str]]

WAV_ID_MODES = ("strip_y", "plain", "dual")
# The mode each split's published numbers were produced with (see build_wav_lookup).
_PAPER_WAV_ID_MODE = {"train": "strip_y", "eval": "plain"}


def extract_yt_id(path: str):
    """The 11-char YouTube id of an image or eval-wav basename, or None if too short.

    A leading "_" is part of the id, so it must not be stripped.
    """
    basename = Path(path).name
    if len(basename) < YT_ID_LENGTH:
        return None
    return basename[:YT_ID_LENGTH]


def build_wav_lookup(wav_paths: Sequence[str], mode: str) -> Dict[str, str]:
    """Map YouTube id -> wav path. The mode is a genuine fork in the published pipeline.

    AudioSet ships its training clips as "Y<id>.wav" (a literal "Y" prepended to the id)
    and its eval clips as "<id>_<start>_<end>.wav". The two source trees disagreed on how
    to undo that:

    * "strip_y" takes `basename[1:12]` for every wav, last writer wins. Correct for
      training wavs, wrong for eval wavs. This is what produced the published fusion
      numbers (and the published training runs), so it is the default on the train split.
    * "plain" takes `basename[:11]`, last writer wins. Correct for eval wavs, wrong for
      training wavs. Default on the eval split, and what the published eval used.
    * "dual" registers both candidates ("Y"-stripped only when the name really matches
      the training convention), first writer wins. It is the only mode that is correct on
      both corpora at once and it recovers the underscore-prefixed clips the other two
      drop, but it changes which triples exist and therefore does not reproduce the
      published numbers. Choose it for new runs, not for replication.
    """
    if mode not in WAV_ID_MODES:
        raise ValueError(f"wav_id_mode must be one of {WAV_ID_MODES}, got {mode!r}")
    lookup: Dict[str, str] = {}
    for path in wav_paths:
        basename = Path(path).name
        if mode == "strip_y":
            lookup[basename[1:YT_ID_LENGTH + 1]] = path
        elif mode == "plain":
            lookup[basename[:YT_ID_LENGTH]] = path
        else:
            for yt_id in _dual_candidates(basename):
                lookup.setdefault(yt_id, path)
    return lookup


def _dual_candidates(basename: str) -> List[str]:
    candidates = []
    if len(basename) >= YT_ID_LENGTH:
        candidates.append(basename[:YT_ID_LENGTH])
    # "Y" + 11-char id + ".wav"; a real id followed by "_start" or "-start" is not that.
    if basename.startswith("Y") and len(basename) >= 12 and basename[11] not in "_-":
        candidates.append(basename[1:12])
    return candidates


def resolve_wav_id_mode(mode, split: str) -> str:
    """The wav id mode to use, substituting the split's published default for None.

    "train" resolves to "strip_y" and "eval" to "plain", the pair every published number
    was produced with. An explicit mode is returned untouched, including "dual".
    """
    return _PAPER_WAV_ID_MODE[split] if mode is None else mode


def phase_label_ids(label_ids: Sequence[str], *, class_order: str = "forward",
                    classes_per_phase: int = CLASSES_PER_PHASE) -> List[List[str]]:
    """Partition the ontology into fixed-size phases, indexed by continual step.

    A trailing remainder of fewer than `classes_per_phase` labels is dropped, exactly as
    the floor division in the development tree did. Reverse class order means step 0
    trains on the last partition; with the 632-class ontology that is the same class set
    the LwF variant obtained by reversing the label list before partitioning.
    """
    if class_order not in CLASS_ORDERS:
        raise ValueError(f"class_order must be one of {CLASS_ORDERS}, got {class_order!r}")
    phases = [
        list(label_ids[i * classes_per_phase:(i + 1) * classes_per_phase])
        for i in range(len(label_ids) // classes_per_phase)
    ]
    return phases if class_order == "forward" else phases[::-1]


def build_exemplars(previous_phase_triples: Sequence[List[Triple]], memory_size: int,
                    *, balancing: bool = False) -> List[Triple]:
    """Fill the replay buffer from the phases already seen, nearest phase first.

    Unbalanced: each previous phase contributes its first `memory_size // n_phases`
    samples in sorted-glob order. Balanced: samples are pooled across phases into
    per-label buckets, each bucket sorted by its string form, and every label gets an even
    quota, topped up two at a time round-robin until the budget is exactly spent.
    """
    n_phases = len(previous_phase_triples)
    if n_phases == 0:
        return []
    if not balancing:
        quota = memory_size // n_phases
        return [t for phase in previous_phase_triples for t in phase[:quota]]

    buckets: Dict[str, List[Triple]] = {}
    total_labels = 0
    for phase_triples in previous_phase_triples:
        phase_buckets = defaultdict(list)
        for triple in phase_triples:
            phase_buckets[triple[2][0]].append(triple)
        for bucket in phase_buckets.values():
            bucket.sort(key=str)
        total_labels += len(phase_buckets)
        buckets.update(phase_buckets)

    # Even quotas keep the buffer splittable into pairs, which the top-up below relies on.
    per_label_quota = (memory_size // total_labels) & ~1
    exemplars: List[Triple] = []
    taken: Dict[str, int] = {}
    for label, bucket in buckets.items():
        take = min(per_label_quota, len(bucket)) & ~1
        exemplars.extend(bucket[:take])
        taken[label] = take

    remaining = memory_size - len(exemplars)
    while remaining >= 2:
        progressed = False
        for label, bucket in buckets.items():
            if remaining < 2:
                break
            start = taken[label]
            if start + 1 < len(bucket):
                exemplars.extend(bucket[start:start + 2])
                taken[label] += 2
                remaining -= 2
                progressed = True
        if not progressed:
            break
    return exemplars


@dataclass(frozen=True)
class Corpus:
    """One split read off disk: the image list plus the three lookups built from it."""

    images: List[str]
    id_to_wav: Dict[str, str]
    id_to_labels: Dict[str, List[str]]
    label_to_name: Dict[str, str]

    @classmethod
    def load(cls, roots: DataRoots, *, wav_id_mode: str) -> "Corpus":
        """Read one split off disk, with the image list in the sorted-glob order everything
        downstream slices in.

        Raises FileNotFoundError when either directory is empty rather than returning a
        corpus that yields no triples, which is otherwise only noticed at the first batch.
        """
        images = sorted(glob(str(Path(roots.images) / "*.jpg")))
        wavs = sorted(glob(str(Path(roots.wavs) / "*.wav")))
        for paths, root, kind in ((images, roots.images, "*.jpg"), (wavs, roots.wavs, "*.wav")):
            if not paths:
                raise FileNotFoundError(
                    f"no {kind} under {root}; point the data roots at the corpus")
        with open(roots.ontology) as f:
            label_to_name = {entry["id"]: entry["name"] for entry in json.load(f)}
        return cls(
            images=images,
            id_to_wav=build_wav_lookup(wavs, wav_id_mode),
            id_to_labels=read_label_csv(roots.meta_csv),
            label_to_name=label_to_name,
        )

    def triples(self, allowed_label_ids: Container[str]) -> List[Triple]:
        """Every image that has a wav, a metadata row, and at least one allowed label."""
        out: List[Triple] = []
        for image in self.images:
            yt_id = extract_yt_id(image)
            if yt_id not in self.id_to_labels or yt_id not in self.id_to_wav:
                continue
            names = [self.label_to_name[label] for label in self.id_to_labels[yt_id]
                     if label in allowed_label_ids]
            if names:
                out.append((image, self.id_to_wav[yt_id], names))
        return out


def read_label_csv(path) -> Dict[str, List[str]]:
    """YouTube id -> label ids, from the AudioSet segment csv."""
    id_to_labels = {}
    with open(path) as f:
        reader = csv.reader(f)
        for _ in range(CSV_HEADER_ROWS):
            next(reader)
        for row in reader:
            id_to_labels[row[0]] = [field.strip().replace('"', "") for field in row[3:]]
    return id_to_labels


def image_transforms():
    """CLIP's own preprocessing: bicubic resize to 224, centre crop, CLIP normalisation.

    The image branch is never augmented, on either split, so the same file yields the same
    pixels during training and evaluation.
    """
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711)),
    ])


def audio_transforms(*, augment: bool):
    """A pipeline yielding a fixed 220500-sample waveform, five seconds at 44.1 kHz.

    With `augment` the clip reaches that length by a random pad or crop and is also flipped,
    rescaled and noised. Without it, padding and cropping are centred and nothing else is
    applied, so evaluation is deterministic.
    """
    if augment:
        return torchvision.transforms.Compose([
            ToTensor1D(),
            RandomFlip(p=0.5),
            RandomScale(max_scale=1.50),
            RandomPadding(out_len=WAV_OUT_LEN),
            RandomCrop(out_len=WAV_OUT_LEN),
            RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=0.25),
        ])
    return torchvision.transforms.Compose([
        ToTensor1D(),
        RandomPadding(out_len=WAV_OUT_LEN, train=False),
        RandomCrop(out_len=WAV_OUT_LEN, train=False),
    ])


def load_triple(triple: Triple, image_transform, audio_transform):
    """Decode one triple into (audio tensor, image tensor, label names)."""
    # Imported here so the sampling logic stays importable without the audio stack.
    import librosa

    image_path, wav_path, text = triple
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    if wav.ndim == 1:
        wav = wav[:, np.newaxis]
    wav = (wav.T * WAV_SCALE).astype(np.float32)
    try:
        wav = audio_transform(wav)
    except Exception as error:
        # A handful of clips decode to zero samples; collate_batch drops them later.
        print(f"audio transform failed for {wav_path}: {error}", flush=True)
    return wav, image_transform(Image.open(image_path).convert("RGB")), text


def collate_batch(batch):
    """Transpose a batch into per-field lists, dropping clips whose audio failed to decode.

    Works for both the 3-tuple and the 4-tuple (exemplar-flag) item layout.
    """
    batch = [item for item in batch if getattr(item[0], "shape", (0,))[-1] > 0]
    if not batch:
        raise RuntimeError("collate_batch: every clip in the batch failed to decode")
    return [list(field) for field in zip(*batch)]


class ContinualAudioSet(torch.utils.data.Dataset):
    """One phase of the continual AudioSet protocol.

    Args:
        roots: locations of the split's images, wavs, csv and ontology.
        phase: continual step. 0 is the bootstrap phase (no exemplars); None disables
            phase splitting altogether and admits every class, which is the joint
            upper-bound view of the corpus.
        split: "train" adds augmentation and replay; "eval" scores against a candidate
            pool and never injects exemplars.
        class_order: "forward" or "reverse". It selects which ontology partition a step
            trains on, nothing else.
        memory_size: exemplar budget. 840 is the paper setting; any other value, 8400
            included, is the same code path.
        include_exemplars: append the replay buffer to the phase's samples. The buffer is
            built either way and stays available as `.exemplars`.
        balancing: pool the buffer per label instead of taking each phase's first slice.
        pool: eval only. "per_phase" scores against the current phase's 79 classes, the
            published protocol; "cumulative" against every class seen so far, a different
            protocol that gives different numbers.
        emit_exemplar_flag: yield 4-tuples ending in an is-exemplar bool, for the
            distillation methods that treat replayed samples differently.
        augment: audio augmentation, defaulting to the split. It is separate because the
            two are genuinely independent: fitting coefficients reads the training corpus
            with deterministic transforms, a combination the split alone cannot express.
        wav_id_mode: see build_wav_lookup. None picks the split's published behaviour.
        length: truncate the assembled sample list, after exemplars were appended.
        classes_per_phase: labels per continual step. At 79 the 632-class ontology splits
            into the bootstrap plus the seven task phases of the protocol.
    """

    def __init__(self, roots: DataRoots, phase, *, split: str = "train", augment=None,
                 class_order: str = "forward", memory_size: int = 840,
                 include_exemplars: bool = True, balancing: bool = False,
                 pool: str = "per_phase", emit_exemplar_flag: bool = False,
                 wav_id_mode=None, length=None,
                 classes_per_phase: int = CLASSES_PER_PHASE):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")

        corpus = Corpus.load(roots, wav_id_mode=resolve_wav_id_mode(wav_id_mode, split))
        phases = phase_label_ids(list(corpus.label_to_name), class_order=class_order,
                                 classes_per_phase=classes_per_phase)
        if phase is None:
            allowed = set(corpus.label_to_name)
        elif split == "eval" and pool == "cumulative":
            allowed = {label for i in range(phase, -1, -1) for label in phases[i]}
        else:
            allowed = set(phases[phase])

        # Training reads out of the full label vocabulary; evaluation retrieves against
        # the phase's candidate pool only, which is what makes the per-phase protocol.
        text_label_ids = sorted(corpus.label_to_name) if split == "train" else sorted(allowed)
        self.text_classes = [corpus.label_to_name[label] for label in text_label_ids]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.text_classes)}
        self.idx_to_class = dict(enumerate(self.text_classes))

        self.samples = corpus.triples(allowed)
        self.exemplars: List[Triple] = []
        if split == "train" and phase:
            previous = [corpus.triples(set(phases[i])) for i in range(phase - 1, -1, -1)]
            self.exemplars = build_exemplars(previous, memory_size, balancing=balancing)
            if include_exemplars:
                self.samples.extend(self.exemplars)
        if length:
            self.samples = self.samples[:length]

        self._exemplar_images = None
        if emit_exemplar_flag:
            self._exemplar_images = {image for image, _, _ in self.exemplars}
        self.image_transforms = image_transforms()
        self.audio_transforms = audio_transforms(
            augment=(split == "train") if augment is None else augment)

    def __getitem__(self, idx):
        triple = self.samples[idx]
        item = load_triple(triple, self.image_transforms, self.audio_transforms)
        if self._exemplar_images is None:
            return item
        return (*item, triple[0] in self._exemplar_images)

    def __len__(self):
        return len(self.samples)
