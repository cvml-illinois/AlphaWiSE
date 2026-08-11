"""Elastic Weight Consolidation: a diagonal-Fisher quadratic pull toward the last phase.

The anchor is the previous phase's weights, snapshotted from the model after the resume
checkpoint is loaded and before any training step. The Fisher diagonal is estimated from
squared gradients of the contrastive loss on this phase's loader, cached next to the phase
checkpoint so a resumed run does not pay for it twice.

The penalty is added to the loss as a term rather than computed inside a wrapper module
around the backbone, which is how the dev tree did it. The gradient is identical, and it
removes the wrapper's ``base.`` state-dict prefix along with the bespoke key-rewriting that
every subsequent loader needed in order to undo it.
"""

import os

import torch

from ...utils import dist
from .base import Batch, ContinualMethod


class EWCMethod(ContinualMethod):
    """EWC: the contrastive loss plus ``scaling * sum_i F_i (theta_i - theta*_i)^2``.

    Two pieces of state span a phase boundary: the anchor ``theta*``, which is the previous
    phase's trainable weights, and the Fisher diagonal ``F``, estimated on the current
    phase's loader and cached on disk. Both are written into the phase checkpoint. The sum
    runs over the parameters that were trainable when the phase began, so a frozen branch
    contributes nothing to the penalty.
    """

    name = "ewc"

    def __init__(self, scaling=0.8, fisher_batches=500, fisher_eps=1e-3,
                 fisher_clamp_max=1e4, fisher_filename="fisher.pt", reuse_cached_fisher=True):
        self.scaling = scaling
        self.fisher_batches = fisher_batches
        self.fisher_eps = fisher_eps
        self.fisher_clamp_max = fisher_clamp_max
        self.fisher_filename = fisher_filename
        self.reuse_cached_fisher = reuse_cached_fisher
        self.fisher = {}
        self.anchor = {}

    def before_phase(self, ctx):
        """Snapshot the anchor, then load or estimate the Fisher diagonal for this phase.

        The floor and the ceiling are applied after loading, not before saving, so a cached
        ``fisher.pt`` holds the raw estimate and a resumed run gets the same clamped values
        as the run that produced it.
        """
        self.anchor = {
            name: parameter.detach().clone()
            for name, parameter in ctx.model.named_parameters()
            if parameter.requires_grad
        }
        cache = os.path.join(ctx.checkpoint_dir, self.fisher_filename)
        if self.reuse_cached_fisher and os.path.isfile(cache):
            cached = torch.load(cache, map_location=ctx.device, weights_only=False)
            self.fisher = {name: cached[name] for name in self.anchor if name in cached}
        else:
            self.fisher = self._estimate(ctx)
            if dist.is_master():
                torch.save({k: v.detach().cpu() for k, v in self.fisher.items()}, cache)

        # A floor keeps parameters the estimate never touched from being free to move, and
        # the ceiling stops a handful of huge entries from dominating the whole penalty.
        for value in self.fisher.values():
            value.add_(self.fisher_eps).clamp_(max=self.fisher_clamp_max)

    def _estimate(self, ctx):
        """Summed squared gradients of the contrastive loss, scaled by ``fisher_batches``."""
        model = ctx.model
        fisher = {name: torch.zeros_like(value) for name, value in self.anchor.items()}
        was_training = model.training
        model.eval()

        batches = iter(ctx.train_loader)
        for _ in range(self.fisher_batches):
            raw = next(batches, None)
            if raw is None:
                break
            batch = Batch.collate(raw, ctx.device)
            _, loss = model(**batch.as_kwargs())
            model.zero_grad(set_to_none=True)
            loss.mean().backward()
            for name, parameter in model.named_parameters():
                if name in fisher and parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square()

        # Deliberately the requested batch count, not `seen`. A phase loader is usually
        # exhausted before fisher_batches is reached, so this is not the arithmetic mean
        # of squared gradients; it is what produced the published checkpoints, and using
        # `seen` instead would scale the whole EWC penalty by fisher_batches / seen.
        for value in fisher.values():
            value /= max(1, self.fisher_batches)
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        return fisher

    def loss_terms(self, ctx, batch, out):
        """The quadratic penalty for the current weights.

        It ignores the batch: the term is a function of the weights alone, recomputed every
        step because they move. Parameters the Fisher does not cover, anything added or
        unfrozen since :meth:`before_phase`, are left unpenalised rather than anchored.
        """
        penalty = 0.0
        for name, parameter in ctx.model.named_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (parameter - self.anchor[name]).pow(2)).sum()
        return {"ewc_penalty": self.scaling * penalty}

    def checkpoint_extras(self, ctx):
        """Fisher and anchor, under the dev tree's key names.

        Nothing in the release reads them back; a resumed phase re-estimates the Fisher or
        reuses the cache file, so these keys exist for inspection and for compatibility with
        checkpoints written by the dev tree.
        """
        return {"ewc_fisher_matrix": self.fisher, "ewc_params": self.anchor}
