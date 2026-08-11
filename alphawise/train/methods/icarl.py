"""iCaRL: herding-selected exemplars plus distillation from the previous phase.

TWO PARTS OF THIS METHOD WERE INERT IN THE RUNS THAT PRODUCED THE PUBLISHED NUMBERS.
Both are implemented correctly here, and both default to the inert behaviour so the
published numbers stay reproducible. Turn either on and the iCaRL rows change.

1. ``legacy_memory=True`` (default). The dev tree called the memory update with the model
   in the place of the memory object, so every call raised
   ``AttributeError: 'DistributedDataParallel' object has no attribute 'per_class_size'``
   into a bare ``except`` that printed and continued. The exemplar bank was never written.
   Confirmed in the run logs, one such line per phase.

2. ``legacy_distillation=True`` (default). The distillation term was guarded by
   ``isinstance(logits, torch.Tensor)``, but the backbone returns
   a three-tuple of similarity matrices, so the guard was never satisfied. Confirmed in
   the run logs: ``Distill: 0.000000`` for all 30 epochs of every phase, including the
   phases that had a teacher.

What the published iCaRL rows therefore measure is sequential fine-tuning on the
replay-augmented dataset the loader already provides: dataset-level replay, no herding,
no distillation.
"""

import copy
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from ...utils import dist
from .base import ContinualMethod, freeze


@torch.no_grad()
def herding_selection(features, count):
    """Greedily pick ``count`` rows whose running mean best tracks the class centroid.

    This is the selection rule from the iCaRL paper: at step k, add the sample that brings
    the mean of the k+1 selected features closest to the class mean.
    """
    total = features.size(0)
    if total == 0:
        return []
    count = min(count, total)

    centroid = features.mean(dim=0)
    centroid = centroid / (centroid.norm() + 1e-8)

    selected = []
    taken = torch.zeros(total, dtype=torch.bool)
    running = torch.zeros_like(centroid)
    for step in range(count):
        candidate_mean = (running.unsqueeze(0) + features) / (step + 1)
        distance = (centroid.unsqueeze(0) - candidate_mean).norm(dim=1)
        distance[taken] = float("inf")
        best = int(distance.argmin())
        selected.append(best)
        taken[best] = True
        running = running + features[best]
    return selected


class ExemplarMemory:
    """Per-class exemplar records and their embedding means, persisted between phases."""

    def __init__(self, total_size):
        self.total_size = total_size
        self.exemplars = {}
        self.audio_mean = {}
        self.image_mean = {}

    @property
    def per_class_size(self):
        """The fixed budget split evenly over the classes registered so far.

        It shrinks as classes accumulate, which is why every phase re-trims the older
        classes and not only the ones it just added.
        """
        return max(1, self.total_size // max(1, len(self.exemplars)))

    def save(self, path):
        torch.save({"total_size": self.total_size, "exemplars": self.exemplars,
                    "audio_mean": self.audio_mean, "image_mean": self.image_mean}, path)

    @classmethod
    def load(cls, path, default_size):
        """Restore a bank, using ``default_size`` when the file records no budget."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        memory = cls(blob.get("total_size", default_size))
        memory.exemplars = blob.get("exemplars", {})
        memory.audio_mean = blob.get("audio_mean", {})
        memory.image_mean = blob.get("image_mean", {})
        return memory


@torch.no_grad()
def _embed(model, dataset, indices, device, batch_size):
    """L2-normalised audio and image embeddings for the given dataset rows."""
    audio_features, image_features = [], []
    for start in range(0, len(indices), batch_size):
        rows = [dataset[i] for i in indices[start:start + batch_size]]
        audio = torch.stack([torch.as_tensor(row[0]) for row in rows]).to(device)
        image = torch.stack([torch.as_tensor(row[1]) for row in rows]).to(device)
        batch_indices = torch.arange(audio.size(0), device=device)
        (features, _), _ = model(audio=audio, image=image, batch_indices=batch_indices)
        audio_features.append(F.normalize(features[0], dim=-1).cpu())
        image_features.append(F.normalize(features[1], dim=-1).cpu())
    return torch.cat(audio_features), torch.cat(image_features)


class ICaRLMethod(ContinualMethod):
    """iCaRL: the contrastive loss plus binary distillation, over a herded exemplar bank.

    Two things would cross a phase boundary with the flags off: the exemplar bank, one file
    per run holding ``memory_size`` samples split evenly over the classes seen so far, and a
    frozen phase k-1 teacher from ``distill_start_phase`` onward, whose sigmoid scores the
    added ``distill_lambda``-weighted binary cross-entropy targets.

    Both ``legacy_memory`` and ``legacy_distillation`` default to True, which reproduces the
    published runs, in which neither the bank nor the distillation term was ever active. The
    module docstring records why and says what those rows therefore measure. Changing either
    default changes the iCaRL numbers.
    """

    name = "icarl"

    def __init__(self, memory_size=840, distill_lambda=1.0, distill_start_phase=2,
                 memory_filename="icarl_memory.pt", feature_batch_size=64,
                 legacy_memory=True, legacy_distillation=True):
        self.memory_size = memory_size
        self.distill_lambda = distill_lambda
        self.distill_start_phase = distill_start_phase
        self.memory_filename = memory_filename
        self.feature_batch_size = feature_batch_size
        self.legacy_memory = legacy_memory
        self.legacy_distillation = legacy_distillation
        self.teacher = None

    def _memory_path(self, ctx):
        # One bank per run, not per phase, so it lives beside the phase directories.
        return os.path.join(os.path.dirname(ctx.checkpoint_dir), self.memory_filename)

    def before_phase(self, ctx):
        """Build the teacher only for the phases that distill; the legacy path builds none."""
        needs_teacher = (not self.legacy_distillation and ctx.phase is not None
                         and ctx.phase >= self.distill_start_phase)
        if needs_teacher:
            self.teacher = freeze(copy.deepcopy(ctx.model)).to(ctx.device)

    def loss_terms(self, ctx, batch, out):
        """The distillation term, or no term at all when this phase has no teacher."""
        if self.teacher is None:
            return {}
        with torch.no_grad():
            (_, teacher_logits), _ = self.teacher(**batch.as_kwargs())
        # Binary cross-entropy against the teacher's sigmoid scores, per the paper's
        # multi-label formulation. AudioSet labels overlap, so a softmax target would be
        # wrong here even though the training objective itself is contrastive.
        distillation = 0.0
        for teacher_logit, student_logit in zip(teacher_logits, out.logits):
            if teacher_logit is None or student_logit is None:
                continue
            target = torch.sigmoid(teacher_logit.detach())
            distillation = distillation + F.binary_cross_entropy_with_logits(student_logit, target)
        return {"distill_loss": self.distill_lambda * distillation}

    def after_phase(self, ctx):
        """Fold this phase's classes into the exemplar bank and rewrite it.

        Master rank only, since the bank is a single shared file, and after the checkpoint,
        so herding embeds the weights that were saved.
        """
        if self.legacy_memory or not dist.is_master():
            return
        dataset = ctx.train_loader.dataset
        memory = self._load_memory(ctx)
        self._select_exemplars(ctx, dataset, memory)
        memory.save(self._memory_path(ctx))

    def _load_memory(self, ctx):
        path = self._memory_path(ctx)
        if os.path.isfile(path):
            return ExemplarMemory.load(path, self.memory_size)
        return ExemplarMemory(self.memory_size)

    def _select_exemplars(self, ctx, dataset, memory):
        """Herd exemplars for this phase's classes, then re-trim every class to quota."""
        rows_by_class = defaultdict(list)
        for row, (_, _, labels) in enumerate(dataset.image_wav_text):
            for label in labels:
                rows_by_class[label].append(row)

        new_classes = [name for name in rows_by_class if name not in memory.exemplars]
        for name in new_classes:
            memory.exemplars[name] = []
        quota = memory.per_class_size

        was_training = ctx.model.training
        ctx.model.eval()
        for name in new_classes:
            rows = rows_by_class[name]
            audio, image = _embed(ctx.model, dataset, rows, ctx.device, self.feature_batch_size)
            picked = herding_selection(audio, quota)
            memory.exemplars[name] = [dataset.image_wav_text[rows[i]] for i in picked]
            memory.audio_mean[name] = audio[picked].mean(dim=0)
            memory.image_mean[name] = image[picked].mean(dim=0)
        ctx.model.train(was_training)

        # Each new class shrinks every older class's share of the fixed budget.
        for name in memory.exemplars:
            memory.exemplars[name] = memory.exemplars[name][:quota]
