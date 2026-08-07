"""Training: one per-phase loop, plus one method per continual-learning method."""

from .methods import ContinualMethod, build_method
from .loop import LoopConfig, run_phase

__all__ = ["LoopConfig", "ContinualMethod", "build_method", "run_phase"]
