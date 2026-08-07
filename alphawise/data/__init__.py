"""Datasets for the continual protocol, and the generator that stands in for AudioSet.

Re-exports resolve on first use rather than at import. Generating the synthetic corpus needs
only the standard library and Pillow, and an eager re-export would pull torch and
torchvision in behind it, so a machine able to build the corpus could not run the command
that builds it.
"""

__all__ = ["ContinualAudioSet", "DataRoots", "FusionMemoryAudioSet", "collate_batch"]

_ORIGINS = {
    "ContinualAudioSet": ".audioset",
    "collate_batch": ".audioset",
    "FusionMemoryAudioSet": ".fusion_memory",
    "DataRoots": ".roots",
}


def __getattr__(name):
    if name not in _ORIGINS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_ORIGINS[name], __name__), name)


def __dir__():
    return sorted(__all__)
