"""Opt-in RNG seeding.

Seeding is opt-in, and that is a deliberate reproducibility contract rather than an
oversight. None of the runs that produced the published checkpoints seeded anything, so
seeding unconditionally would silently change every number the paper reports. Passing
``seed=None`` (the default everywhere) therefore leaves ``random``, ``numpy`` and ``torch``
exactly as the historical code path left them: untouched. Passing an integer makes a run
repeatable from that point on.

Repeatable is not the same as deterministic. cuDNN autotuning and the nondeterministic
CUDA kernels stay enabled, because turning them off is a second, much larger behaviour
change. A seeded run reproduces its own sample order and initialisation, not necessarily
its floating-point results on a different GPU.
"""

import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and every CUDA device).

    ``seed=None`` is a no-op and returns None, so callers can pass an unset config value
    straight through without branching.
    """
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def worker_init_fn(seed):
    """Build a DataLoader ``worker_init_fn`` that gives each worker a distinct seed.

    This is a factory: pass ``worker_init_fn(seed)``, never the bare function. The
    DataLoader calls whatever it is given as ``fn(worker_id)``, so passing the factory
    itself builds and discards a closure once per worker and reseeds nothing, silently.

    Returns None for ``seed=None``, which is exactly what ``DataLoader`` expects in order
    to keep its default (unseeded) worker behaviour.
    """
    if seed is None:
        return None

    def init(worker_id):
        seed_everything(seed + worker_id)

    return init
