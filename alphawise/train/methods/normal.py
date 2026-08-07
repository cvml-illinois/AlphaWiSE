"""Sequential fine-tuning, the no-mitigation baseline.

Normal is the shared loop with nothing added: phase k starts from phase k-1's weights and
trains on phase k's data (plus whatever replay the dataset provides) with no penalty, no
teacher and no memory. It exists as a named method so the method registry has an entry for
it and so the loop's default and the "normal" baseline are provably the same code path.
"""

from .base import ContinualMethod


class NormalMethod(ContinualMethod):
    """Sequential fine-tuning: no extra loss term, no teacher, no memory.

    It carries nothing between phases beyond the weights themselves, which is what makes it
    the unconstrained endpoint the fusion configurations interpolate from. It adds no
    behaviour to :class:`~alphawise.train.methods.base.ContinualMethod`, deliberately.
    """

    name = "normal"
