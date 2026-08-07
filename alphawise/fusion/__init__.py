"""Post hoc interpolation of two frozen checkpoints, the method this repository ships.

:mod:`~alphawise.fusion.blend` is the portable seam: a pure tensor blend over state
dicts that imports only torch and applies to any architecture. Everything else is the
AudioCLIP-specific machinery around it: :mod:`~alphawise.fusion.core` wraps the blend
with the 499 trainable logits and the endpoint checks, :mod:`~alphawise.fusion.fit`
fits those logits on the exemplar memory, and :mod:`~alphawise.fusion.wiseft` is the
fixed-coefficient baseline.

Importing this package pulls in torch and nothing heavier, so ``from alphawise.fusion
import blend`` works on a machine with no AudioCLIP assets and no AudioSet.
"""

from .blend import blend, sigmoid_alphas

__all__ = ["blend", "sigmoid_alphas"]
