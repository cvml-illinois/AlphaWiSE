"""DER: a frozen previous-phase representation fused with the plastic one.

The previous phase's model is kept frozen as a second feature extractor. Its embeddings are
concatenated per modality with the current model's and projected back to the embedding
dimension by a small trainable head; the contrastive objective is computed on that fused
representation. This is the one method that replaces the backbone's own loss rather than
adding to it, hence ``uses_base_loss = False``.

Two auxiliary terms accompany it: a contrastive loss on the plastic features of the
current-phase samples only (the replayed ones are excluded, which is what the collate
function's exemplar markers are for), and an L1 pull on the audio encoder that prepares the
post-phase channel pruning.

Phase 1 has no previous phase to fuse with; the orchestrator runs it with the normal method.
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from ...utils import dist
from .base import ContinualMethod, freeze


class ProjectionHead(nn.Module):
    """Maps a concatenated (frozen, plastic) pair back to the embedding dimension."""

    def __init__(self, embed_dim):
        super().__init__()
        self.audio = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.image = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.text = nn.Linear(2 * embed_dim, embed_dim, bias=False)

    def forward(self, frozen, plastic):
        fused = [
            layer(torch.cat([old, new], dim=-1))
            for layer, old, new in zip((self.audio, self.image, self.text), frozen, plastic)
        ]
        return [F.normalize(value, dim=-1) for value in fused]


def contrastive_loss(audio, image, text, scales):
    """Symmetric three-pair CLIP loss, averaged over the three pairs."""
    scale_ai, scale_at, scale_it = scales
    reference = torch.arange(audio.shape[0], dtype=torch.int64, device=audio.device)
    total = 0.0
    for logits in (scale_ai * audio @ image.T, scale_at * audio @ text.T,
                   scale_it * image @ text.T):
        total = total + F.cross_entropy(logits, reference) + F.cross_entropy(logits.T, reference)
    return total / 3.0


def prune_audio_channels(audio_encoder, ratio):
    """Permanently zero the lowest-L1-norm output channels of every audio conv layer."""
    pruned = 0
    for _, layer in audio_encoder.named_modules():
        if not isinstance(layer, nn.Conv2d):
            continue
        channels = layer.weight.shape[0]
        amount = max(1, int(ratio * channels))
        if amount >= channels:
            continue
        prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
        prune.remove(layer, "weight")
        pruned += 1
    return pruned


class DERMethod(ContinualMethod):
    """DER: the contrastive objective moved onto features fused from two representations.

    The loss is ``main + lambda_aux * aux + lambda_sparse * L1`` and replaces the backbone's
    own, hence ``uses_base_loss = False``. ``main`` is the contrastive loss on the projection
    of the frozen phase k-1 features concatenated with the current ones, ``aux`` the same
    loss on the current model's features for the non-replayed samples only, and the L1 term
    an audio-encoder sparsity pull that prepares the end-of-phase pruning.

    Carried between phases: the frozen previous model, rebuilt each phase from the resumed
    weights. The projection head is new every phase; its trained state is saved with the
    checkpoint but is never loaded back.
    """

    name = "der"
    uses_base_loss = False

    #: Parameters whose L1 norm is not penalised: normalisation layers carry scale, and the
    #: filter-bank coefficients are not weights in the usual sense.
    SPARSE_SKIP = ("bn", "norm", "fbsp")

    def __init__(self, lambda_aux=1.0, lambda_sparse=1e-5, prune_ratio=0.3,
                 pruned_filename="model_weights_pruned.pth"):
        self.lambda_aux = lambda_aux
        self.lambda_sparse = lambda_sparse
        self.prune_ratio = prune_ratio
        self.pruned_filename = pruned_filename
        self.teacher = None
        self.proj_head = None

    def before_phase(self, ctx):
        """Freeze the phase k-1 model and build the projection head this phase will train."""
        self.teacher = freeze(copy.deepcopy(ctx.model)).to(ctx.device)
        self.proj_head = ProjectionHead(ctx.model.embed_dim).to(ctx.device)

    def modules(self):
        return {"proj_head": self.proj_head}

    def _scales(self, model):
        return (
            torch.clamp(model.logit_scale_ai.exp(), 1.0, 100.0),
            torch.clamp(model.logit_scale_at.exp(), 1.0, 100.0),
            torch.clamp(model.logit_scale.exp(), 1.0, 100.0),
        )

    def loss_terms(self, ctx, batch, out):
        """The DER objective, whose ``main`` term stands in for the backbone's own loss.

        ``aux`` is dropped when the batch carries no exemplar markers or leaves fewer than
        two current-phase samples, since a contrastive loss needs at least two rows.
        """
        with torch.no_grad():
            (frozen_features, _), _ = self.teacher(**batch.as_kwargs())
        scales = self._scales(ctx.model)

        fused = self.proj_head(frozen_features, out.features)
        terms = {"main": contrastive_loss(*fused, scales)}

        aux = self._auxiliary(batch, out, scales)
        if aux is not None:
            terms["aux"] = self.lambda_aux * aux
        terms["sparse"] = self.lambda_sparse * self._sparsity(ctx.model)
        return terms

    def _auxiliary(self, batch, out, scales):
        """Contrastive loss on this phase's own samples, excluding replayed exemplars."""
        if not batch.extra:
            return None
        is_exemplar = torch.as_tensor(batch.extra[0], dtype=torch.bool, device=batch.audio.device)
        current = ~is_exemplar
        if int(current.sum()) < 2:
            return None
        features = [F.normalize(value[current], dim=-1) for value in out.features]
        return contrastive_loss(*features, scales)

    def _sparsity(self, model):
        total = torch.zeros((), device=next(model.parameters()).device)
        for name, parameter in model.audio.named_parameters():
            # audio.fbsp.bias is a legitimately zero-element tensor, and mean() over it is
            # NaN, which would poison the whole loss.
            if (parameter.requires_grad and parameter.numel() > 0
                    and not any(skip in name for skip in self.SPARSE_SKIP)):
                total = total + parameter.abs().mean()
        return total

    def after_phase(self, ctx):
        """Prune after the phase checkpoint is written, so that file keeps dense weights."""
        if self.prune_ratio <= 0 or not dist.is_master():
            return
        pruned = prune_audio_channels(ctx.model.audio, self.prune_ratio)
        path = os.path.join(ctx.checkpoint_dir, self.pruned_filename)
        torch.save({"model_state_dict": ctx.model.state_dict()}, path)
        print(f"pruned {pruned} audio conv layers at ratio {self.prune_ratio} -> {path}",
              flush=True)
