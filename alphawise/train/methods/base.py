"""The contract between the shared loop and a continual-learning method.

A method is a :class:`ContinualMethod`: a handful of callbacks and nothing else. It never sees
the epoch loop, the DDP wrap, the optimizer or the checkpoint format, which is what stops
the seven trainers of the dev tree from drifting apart again.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import torch


def _stack(values, device):
    """Collate one modality into a batch tensor on ``device``.

    The continual datasets yield a mix of tensors and numpy arrays within a single batch,
    so the coercion has to happen per element.
    """
    if torch.is_tensor(values):
        return values.to(device)
    tensors = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in values]
    return torch.stack(tensors).to(device)


@dataclass
class Batch:
    """One collated training batch, already on the device."""

    audio: torch.Tensor
    image: torch.Tensor
    text: Sequence
    indices: torch.Tensor
    #: Anything the collate function appended, e.g. DER's per-sample exemplar markers.
    extra: Tuple = ()

    @classmethod
    def collate(cls, batch, device):
        """Build a Batch from whatever the dataset returned, keeping any trailing elements.

        ``indices`` is a fresh ``arange`` over the batch, not dataset row ids: it is what the
        backbone uses to align text with the other two modalities, so it must number the rows
        of this batch and nothing else.
        """
        audio, image, text, *extra = batch
        audio = _stack(audio, device)
        image = _stack(image, device)
        indices = torch.arange(audio.shape[0], dtype=torch.int64, device=device)
        return cls(audio=audio, image=image, text=text, indices=indices, extra=tuple(extra))

    def as_kwargs(self):
        """The four arguments the backbone's ``forward`` takes. ``extra`` is for methods only."""
        return {
            "audio": self.audio,
            "image": self.image,
            "text": self.text,
            "batch_indices": self.indices,
        }


@dataclass
class StepOutput:
    """What the backbone returned for one batch."""

    #: L2-normalised (audio, image, text) embeddings.
    features: Tuple
    #: (audio-image, audio-text, image-text) similarity logits.
    logits: Tuple
    #: The backbone's own contrastive loss, already reduced to a scalar.
    loss: torch.Tensor


@dataclass
class PhaseContext:
    """Everything a method may read about the phase being trained."""

    config: Any
    #: The unwrapped backbone. It stays the same object after the loop's DDP wrap, so a
    #: method can hold on to it and still see the trained weights.
    model: torch.nn.Module
    train_loader: Any
    device: str
    checkpoint_dir: str
    #: Phase index in 1..7, or None for joint training over all classes at once.
    phase: Optional[int] = None
    class_order: str = "forward"
    #: Populated by the loop from the resume checkpoint, for methods that need its extras.
    resume_blob: Optional[dict] = field(default=None, repr=False)


class ContinualMethod:
    """A continual-learning method, expressed as callbacks on the shared loop."""

    name = "normal"

    #: DER recomputes the contrastive objective on fused features, so it replaces the
    #: backbone's own loss instead of adding to it. Every other method adds.
    uses_base_loss = True

    def before_phase(self, ctx):
        """Run once after the previous phase's weights are loaded and before the optimizer
        is built. Fisher estimation, teacher snapshots and LoRA injection belong here."""

    def modules(self):
        """Trainable modules the method owns besides the backbone, keyed by attribute name.

        The loop moves each to the device, wraps it for DDP, adds its parameters to the
        optimizer, saves its state under ``<name>_state_dict``, and assigns the wrapped
        module back to the method attribute of the same name, so the method always calls the
        object DDP can see.
        """
        return {}

    def loss_terms(self, ctx, batch, out):
        """Extra named loss terms for this batch. They are summed into the total, logged
        per epoch under their own names, and must be scalar tensors."""
        return {}

    def before_save(self, ctx):
        """Run after the last epoch, before the checkpoint is written. For methods whose
        trained state has to be folded back into the backbone first."""

    def after_phase(self, ctx):
        """Run after the checkpoint is written. For side artifacts and memory updates,
        which must not alter the weights that were just saved."""

    def checkpoint_extras(self, ctx):
        """Extra keys to merge into the saved blob."""
        return {}


def freeze(module):
    """Put a module in eval mode with no gradients, for use as a frozen teacher."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module
