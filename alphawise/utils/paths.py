"""Resolution of on-disk locations, so no module hardcodes an absolute path.

Every location is resolved in the same order: explicit argument, then environment
variable, then a repository-relative default. Nothing here may contain a machine- or
site-specific path, since this file ships with the release.
"""

import os
import re
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EPOCH_PATTERN = re.compile(r"model_weights_epoch_(\d+)\.pth$")


def _from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def assets_dir() -> Path:
    """Pretrained backbone weights and the BPE vocabulary (see scripts/fetch_assets.py)."""
    return _from_env("ALPHAWISE_ASSETS", _REPO_ROOT / "assets")


def asset(name: str) -> Path:
    """Resolve one asset, failing with the fetch instruction rather than a load error."""
    path = assets_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"missing asset {name!r} at {path}. Run `python scripts/fetch_assets.py`, "
            f"or point ALPHAWISE_ASSETS at a directory that already holds it."
        )
    return path


def highest_epoch_checkpoint(phase_dir: Path) -> Optional[Path]:
    """The highest-numbered ``model_weights_epoch_<N>.pth`` under ``phase_dir``, or None.

    Epoch coverage varies between the released runs, so "the phase's checkpoint" means
    the last epoch present, not a fixed filename.
    """
    best, best_epoch = None, -1
    if phase_dir.is_dir():
        for path in phase_dir.iterdir():
            match = _EPOCH_PATTERN.search(path.name)
            if match and int(match.group(1)) > best_epoch:
                best, best_epoch = path, int(match.group(1))
    return best


def resolve_bootstrap(cfg, class_order: str) -> Optional[Path]:
    """The shared phase-0 checkpoint one class order's trajectories start from, or None.

    Resolution order: the path the config declares at ``checkpoints.bootstrap.<class_order>``
    (the registry's spelling, ``model_weights_epoch_30.pth``), then the conventional
    ``<paths.bootstrap_root>/<class_order>/phase_0/`` directory holding either a plain
    ``model_weights.pth`` or the highest ``model_weights_epoch_<N>.pth``.

    A declared bootstrap that is missing on disk raises rather than degrading: silently
    training phase 1 from the pretrained backbone is not the published protocol, where
    every method shares one bootstrap, and it poisons every FWT number downstream. None
    is returned only when no bootstrap is declared and none is on disk, which is the
    legitimate fresh-run case the caller must announce out loud.
    """
    declared = cfg.get(f"checkpoints.bootstrap.{class_order}")
    if declared:
        path = Path(declared)
        if not path.is_file():
            raise FileNotFoundError(
                f"the config declares the shared phase-0 bootstrap at {path} and it is "
                f"missing. Every trajectory must start from that bootstrap; remove "
                f"checkpoints.bootstrap.{class_order} only to deliberately train from "
                f"the pretrained backbone, which is not the published protocol."
            )
        return path
    phase_dir = Path(cfg["paths.bootstrap_root"]) / class_order / "phase_0"
    plain = phase_dir / "model_weights.pth"
    if plain.is_file():
        return plain
    return highest_epoch_checkpoint(phase_dir)
