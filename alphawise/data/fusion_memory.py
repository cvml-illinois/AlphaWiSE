"""The dataset the AlphaWiSE interpolation coefficients are fitted on.

Fitting alpha is not training: the objective has to see the old classes the stability
endpoint protects and the new classes the plastic endpoint fits, in comparable amounts.
This dataset therefore pairs the full balanced replay buffer with a class balanced draw
from the current phase, so a phase is never dominated by whichever side happens to have
more data on disk.

At the paper's budget of 840 the sizes are fixed by arithmetic alone, 840 exemplars plus
840 // phase current-phase samples:

    phase   1     2     3     4     5     6     7
    total   1680  1260  1120  1050  1008  980   960

The paper's Eq. 3 draws its fitting batches from the exemplar memory M alone, |M| = 840,
so it understates the fitting set by the whole current-phase draw. The published
coefficients were fitted on the sizes above; this dataset reproduces that behaviour, not
the equation, and the divergence is recorded rather than smoothed over.
"""

import random
from collections import defaultdict
from typing import Dict, List

import torch

from .audioset import (
    Corpus,
    CLASSES_PER_PHASE,
    Triple,
    audio_transforms,
    build_exemplars,
    image_transforms,
    load_triple,
    phase_label_ids,
    resolve_wav_id_mode,
)
from .roots import DataRoots

CURRENT_DRAWS = ("phase_quota", "match_exemplars")


def draw_current_phase(triples: List[Triple], target: int, *, balanced: bool) -> List[Triple]:
    """Take `target` samples from the current phase, spread evenly over its labels.

    The balanced path buckets by primary label, shuffles each bucket, gives every label an
    equal quota and then hands out the remainder one at a time round-robin. Both paths
    shuffle through the global `random` module rather than a private generator: that is
    the stream the published coefficients were fitted on, so it is preserved rather than
    fixed. Seed it via `alphawise.utils.seed` when a run has to be repeatable.
    """
    if not balanced:
        pool = list(triples)
        random.shuffle(pool)
        return pool[:min(target, len(pool))]

    buckets: Dict[str, List[Triple]] = defaultdict(list)
    for triple in triples:
        buckets[triple[2][0]].append(triple)
    for bucket in buckets.values():
        random.shuffle(bucket)

    per_label_quota = target // len(buckets) if buckets else 0
    drawn: List[Triple] = []
    taken: Dict[str, int] = {}
    for label, bucket in buckets.items():
        take = min(per_label_quota, len(bucket))
        drawn.extend(bucket[:take])
        taken[label] = take

    remaining = target - len(drawn)
    while remaining > 0:
        progressed = False
        for label, bucket in buckets.items():
            if remaining <= 0:
                break
            start = taken[label]
            if start < len(bucket):
                drawn.append(bucket[start])
                taken[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return drawn


class FusionMemoryAudioSet(torch.utils.data.Dataset):
    """Replay buffer plus a matched current-phase draw, for fitting the alpha coefficients.

    Args:
        roots: locations of the split's images, wavs, csv and ontology.
        phase: continual step the coefficients are being fitted for.
        class_order: "forward" or "reverse", as in ContinualAudioSet.
        memory_size: exemplar budget, 840 in the paper.
        balancing: balance both the replay buffer and the current-phase draw per label.
        current_draw: "phase_quota" draws `memory_size // phase` current samples, one
            previous phase's worth, which is what the published runs used.
            "match_exemplars" draws as many as the whole buffer holds, a 50/50 split.
        split: "train" augments the audio, "eval" does not.
        augment: overrides the split's augmentation choice, so the training corpus can be
            read with deterministic transforms.
        wav_id_mode: see `alphawise.data.audioset.build_wav_lookup`. None keeps the
            published behaviour for the split.
        classes_per_phase: labels per continual step, as in ContinualAudioSet. It must
            match the value the phase's checkpoints were trained with.
    """

    def __init__(self, roots: DataRoots, phase: int, *, class_order: str = "forward",
                 memory_size: int = 840, balancing: bool = True,
                 current_draw: str = "phase_quota", split: str = "train", augment=None,
                 wav_id_mode=None, classes_per_phase: int = CLASSES_PER_PHASE):
        if current_draw not in CURRENT_DRAWS:
            raise ValueError(f"current_draw must be one of {CURRENT_DRAWS}, got {current_draw!r}")

        corpus = Corpus.load(roots, wav_id_mode=resolve_wav_id_mode(wav_id_mode, split))
        phases = phase_label_ids(list(corpus.label_to_name), class_order=class_order,
                                 classes_per_phase=classes_per_phase)

        text_label_ids = sorted(corpus.label_to_name)
        self.text_classes = [corpus.label_to_name[label] for label in text_label_ids]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.text_classes)}
        self.idx_to_class = dict(enumerate(self.text_classes))

        previous = [corpus.triples(set(phases[i])) for i in range(phase - 1, -1, -1)]
        self.exemplars = build_exemplars(previous, memory_size, balancing=balancing)

        if current_draw == "phase_quota" and phase:
            target = memory_size // phase
        else:
            target = len(self.exemplars)
        self.current_phase = draw_current_phase(
            corpus.triples(set(phases[phase])), target,
            balanced=balancing and bool(self.exemplars))

        self.samples = self.exemplars + self.current_phase
        self.image_transforms = image_transforms()
        self.audio_transforms = audio_transforms(
            augment=(split == "train") if augment is None else augment)

    def __getitem__(self, idx):
        return load_triple(self.samples[idx], self.image_transforms, self.audio_transforms)

    def __len__(self):
        return len(self.samples)
