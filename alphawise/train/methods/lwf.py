"""Learning without Forgetting: temperature-softened distillation from the last phase.

The teacher is a frozen copy of the model taken after the previous phase's weights are
loaded and before the first step, so it is exactly the phase k-1 model. Distillation runs
on all three similarity matrices the backbone produces (audio-image, audio-text,
image-text), which is what makes this the multimodal variant of the method.

VR-LwF is this method with a larger coefficient (0.8 instead of 0.1), not a separate method,
so the registry gives it a different default and no second implementation.
"""

import copy

# The dev tree's LwF trainer also imported torch.distributed and never used it.
# Distributed concerns belong to the loop, so no method imports it.
import torch
import torch.nn.functional as F

from .base import ContinualMethod, freeze


class LwFMethod(ContinualMethod):
    """LwF: the contrastive loss plus a soft-target cross-entropy to the phase k-1 model.

    The added term is ``distill_lambda * T^2`` times the cross-entropy between the teacher's
    softened similarity distributions and the student's, summed over the three modality
    pairs. The frozen teacher is the only state carried across a phase boundary; there is no
    exemplar memory and no per-parameter statistic, so the cost is one extra forward pass per
    step and one extra copy of the backbone in memory.
    """

    name = "lwf"

    def __init__(self, distill_lambda=0.1, temperature=2.0):
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.teacher = None

    def before_phase(self, ctx):
        """Freeze a copy of the weights just loaded, which are exactly the phase k-1 model."""
        self.teacher = freeze(copy.deepcopy(ctx.model)).to(ctx.device)

    def loss_terms(self, ctx, batch, out):
        """The distillation term, over whichever similarity matrices the batch produced.

        A pair is None when one of its two modalities is absent from the batch, and is
        skipped rather than counted as zero, so the term is a sum over the pairs that exist
        and not an average over three.
        """
        temperature = self.temperature
        with torch.no_grad():
            (_, teacher_logits), _ = self.teacher(**batch.as_kwargs())

        distillation = 0.0
        for teacher_logit, student_logit in zip(teacher_logits, out.logits):
            if teacher_logit is None or student_logit is None:
                continue
            target = F.softmax(teacher_logit / temperature, dim=-1)
            prediction = F.log_softmax(student_logit / temperature, dim=-1)
            distillation = distillation - (target * prediction).sum(dim=-1).mean()

        # The temperature-squared factor keeps the gradient magnitude comparable to the
        # unsoftened loss, as in the original formulation.
        return {"lwf_loss": self.distill_lambda * temperature ** 2 * distillation}
