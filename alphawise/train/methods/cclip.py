"""C-CLIP: LoRA adapters plus contrastive knowledge consolidation.

Two ingredients. LoRA wraps the transformer MLP projections and the audio encoder's final
linear layer, so the backbone itself stays frozen for the duration of the phase and only
the low-rank updates move; at the phase boundary they are folded back into the base weights
with coefficient ``merge_alpha`` and the wrappers are stripped, leaving a plain backbone
checkpoint that loads like any other method's.

CKC replaces an L2 or KL pull with a contrastive one: the concatenated per-sample
embeddings of the new model, passed through a small projector, are aligned to the frozen
old model's concatenated embeddings by a bidirectional InfoNCE. That is an objective of the
same kind as the training loss rather than one that fights it.

This method manages its own freezing, so it runs the loop with ``freeze="method"``.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualMethod, freeze

#: Attributes that name a wrappable Linear: the two transformer MLP projections and the
#: audio encoder's output layer. `out_proj` inside nn.MultiheadAttention is deliberately
#: excluded: MHA reads `out_proj.weight` directly instead of calling the submodule, so a
#: wrapper there raises AttributeError at the first forward.
LORA_TARGETS = ("c_fc", "c_proj", "fc")


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable rank-``rank`` update."""

    def __init__(self, base, rank=16, lora_alpha=32, dropout=0.1):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero so the wrapped layer starts out exactly equal to the base layer.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        update = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merge(self, alpha):
        """Add ``alpha`` times the adapter into the base weight and reset the adapter.

        Zeroing A and B afterwards keeps merging idempotent: a second call adds nothing.
        """
        delta = (self.lora_B @ self.lora_A) * (self.scaling * alpha)
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.lora_A.zero_()
        self.lora_B.zero_()


def apply_lora(model, rank=16, lora_alpha=32, dropout=0.1):
    """Freeze the model and wrap every targetable Linear. Returns the number wrapped."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrapped = 0
    for parent in list(model.modules()):
        if isinstance(parent, nn.MultiheadAttention):
            continue
        for attribute in LORA_TARGETS:
            child = getattr(parent, attribute, None)
            if isinstance(child, nn.Linear):
                setattr(parent, attribute,
                        LoRALinear(child, rank=rank, lora_alpha=lora_alpha, dropout=dropout))
                wrapped += 1
    return wrapped


def merge_lora(model, alpha):
    """Fold every adapter into its base layer and remove the wrappers, in place."""
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge(alpha)
            merged += 1
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.base)
    return merged


class CKCProjector(nn.Module):
    """Keeps the new and old feature spaces connected without forcing them to coincide."""

    def __init__(self, dim=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def ckc_loss(new_features, old_features, projector, tau):
    """Bidirectional InfoNCE between projected new and raw old concatenated embeddings."""
    new = F.normalize(torch.cat([projector(value) for value in new_features], dim=-1), dim=-1)
    old = F.normalize(torch.cat(list(old_features), dim=-1), dim=-1)
    targets = torch.arange(new.size(0), device=new.device)
    return 0.5 * (F.cross_entropy(new @ old.t() / tau, targets)
                  + F.cross_entropy(old @ new.t() / tau, targets))


class CCLIPMethod(ContinualMethod):
    """C-CLIP: LoRA-only adaptation plus a ``ckc_lambda``-weighted contrastive consolidation.

    The added term is the bidirectional InfoNCE of :func:`ckc_loss` between the projected new
    embeddings and the frozen old model's, at temperature ``ckc_tau``. What crosses a phase
    boundary is only the merged backbone: the adapters are folded in at ``merge_alpha`` in
    :meth:`before_save` and the CKC projector, like the adapters, is reinitialised next
    phase. Because the method does its own freezing, the loop must run with ``freeze="method"``.
    """

    name = "cclip"

    #: Trainable on top of the adapters: the temperatures and the audio filter bank.
    UNFROZEN = ("logit_scale", "logit_scale_ai", "logit_scale_at")

    def __init__(self, lora_rank=16, lora_alpha=32, lora_dropout=0.1, merge_alpha=0.5,
                 ckc_lambda=1.0, ckc_tau=0.07, projector_hidden=None):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_alpha = merge_alpha
        self.ckc_lambda = ckc_lambda
        self.ckc_tau = ckc_tau
        self.projector_hidden = projector_hidden
        self.teacher = None
        self.ckc_projector = None

    def before_phase(self, ctx):
        """Snapshot the teacher, inject the adapters, then unfreeze the temperatures and FBSP.

        The loop's freeze policy is off for this method, so everything C-CLIP trains is
        decided here.
        """
        # The teacher must be copied before LoRA injection, so it stays a plain backbone.
        self.teacher = freeze(copy.deepcopy(ctx.model)).to(ctx.device)
        apply_lora(ctx.model, rank=self.lora_rank, lora_alpha=self.lora_alpha,
                   dropout=self.lora_dropout)
        for name in self.UNFROZEN:
            getattr(ctx.model, name).requires_grad = True
        for parameter in ctx.model.audio.fbsp.parameters():
            parameter.requires_grad = True
        dim = ctx.model.embed_dim
        self.ckc_projector = CKCProjector(dim, self.projector_hidden or dim)

    def modules(self):
        return {"ckc_projector": self.ckc_projector}

    def loss_terms(self, ctx, batch, out):
        """The CKC term, applied to every batch of every phase, the first one included.

        On phase 1 the teacher is the shared bootstrap, so the term is a pull toward the
        pretrained model rather than toward a previously adapted one.
        """
        with torch.no_grad():
            (old_features, _), _ = self.teacher(**batch.as_kwargs())
        loss = ckc_loss(out.features, old_features, self.ckc_projector, self.ckc_tau)
        return {"ckc_loss": self.ckc_lambda * loss}

    def before_save(self, ctx):
        """Fold the adapters in before the checkpoint is taken, so it holds a plain model."""
        merged = merge_lora(ctx.model, self.merge_alpha)
        print(f"merged {merged} LoRA adapters at alpha={self.merge_alpha}", flush=True)
