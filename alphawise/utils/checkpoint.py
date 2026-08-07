"""Reading and writing the per-phase checkpoint blob.

One blob shape is used by every method: model, optimizer and scheduler state plus the
epoch and the epoch's mean loss, with an open ``extras`` slot for whatever a method needs
to carry across a phase boundary (EWC's Fisher, DER's projection head, and so on).

Checkpoints written by different launchers disagree on key prefixes: a DDP save prefixes
every key with ``module.`` and the dev tree's EWC wrapper added a further ``base.``.
Loading normalises both away, so a checkpoint from any of the seven trainers loads into a
plain backbone.
"""

import torch

from .dist import unwrap

_PREFIXES = ("module.", "base.")


def strip_prefixes(state_dict):
    """Drop one leading ``module.`` or ``base.`` from every key."""
    stripped = {}
    for key, value in state_dict.items():
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        stripped[key] = value
    return stripped


def load_blob(path, map_location="cpu"):
    """Load a checkpoint file. ``weights_only=False`` because the blob carries Fisher
    matrices and exemplar records, not just tensors."""
    return torch.load(path, map_location=map_location, weights_only=False)


def model_state_of(blob):
    """The model weights inside a blob, tolerating a bare state dict."""
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def load_model_state(model, path, strict=True, map_location="cpu"):
    """Load weights from ``path`` into ``model``. Returns the full blob for the caller,
    which may still want the optimizer state out of it."""
    blob = load_blob(path, map_location)
    unwrap(model).load_state_dict(strip_prefixes(model_state_of(blob)), strict=strict)
    return blob


def resume_trainer_state(optimizer, scheduler, blob):
    """Restore optimizer and scheduler from a blob, returning the epoch to resume at.

    Only meaningful when continuing an interrupted run of the same phase. A phase-to-phase
    bootstrap deliberately starts from a fresh optimizer.
    """
    optimizer.load_state_dict(blob["optimizer_state_dict"])
    scheduler.load_state_dict(blob["scheduler_state_dict"])
    return blob["epoch"]


def save_phase_checkpoint(model, optimizer, scheduler, path, epoch, avg_loss, extras=None):
    """Write the canonical blob. The caller guards for rank 0; this writes unconditionally."""
    blob = {
        "epoch": epoch,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "avg_epoch_loss": avg_loss,
    }
    if extras:
        blob.update(extras)
    torch.save(blob, path)
    return path
