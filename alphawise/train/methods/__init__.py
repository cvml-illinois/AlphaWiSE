"""The continual-learning methods, one small method each.

Every method plugs into :func:`alphawise.train.loop.run_phase`; none of them contains a
training loop.
"""

from .base import Batch, ContinualMethod, PhaseContext, StepOutput
from .cclip import CCLIPMethod
from .der import DERMethod
from .ewc import EWCMethod
from .icarl import ICaRLMethod
from .lwf import LwFMethod
from .normal import NormalMethod

_METHODS = {
    "normal": NormalMethod,
    "ewc": EWCMethod,
    "lwf": LwFMethod,
    "vlwf": LwFMethod,
    "icarl": ICaRLMethod,
    "der": DERMethod,
    "cclip": CCLIPMethod,
}

#: VR-LwF is LwF with a stronger distillation coefficient, not a separate method.
_OVERRIDES = {"vlwf": {"distill_lambda": 0.8}}


def build_method(name, **kwargs):
    """Construct the method registered under ``name``."""
    if name not in _METHODS:
        raise ValueError(f"unknown method {name!r}; known: {sorted(_METHODS)}")
    return _METHODS[name](**{**_OVERRIDES.get(name, {}), **kwargs})


__all__ = [
    "Batch", "ContinualMethod", "PhaseContext", "StepOutput",
    "CCLIPMethod", "DERMethod", "EWCMethod", "ICaRLMethod", "LwFMethod", "NormalMethod",
    "build_method",
]
