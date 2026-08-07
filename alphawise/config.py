"""Resolution of the shipped YAML configuration into one interpolated document.

Layers, lowest precedence first:

1. ``configs/base.yaml``, always;
2. the overlay files named by the caller, each expanded through its own ``includes:``;
3. the files named by ``ALPHAWISE_CONFIG`` (os.pathsep separated), the site-local escape
   hatch where an operator keeps absolute data roots that must never ship;
4. ``ALPHAWISE__<DOTTED__KEY>`` environment variables;
5. explicit overrides from the command line.

Layers 4 and 5 may only touch keys that already exist, so a typo fails loudly instead of
being silently ignored. Interpolation runs once at the end, after every layer has been
merged, so overriding a root path propagates into every value derived from it.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_PREFIX = "ALPHAWISE__"
CONFIG_DIR_ENV = "ALPHAWISE_CONFIG_DIR"
_REFERENCE = re.compile(r"\$\{([^{}]+)\}")
_MAX_INTERPOLATION_PASSES = 10


class ConfigError(Exception):
    """A config file is missing, a key is unknown, or a reference cannot be resolved."""


class Config:
    """A resolved document, addressed by dotted key: ``cfg["train.optimizer.lr"]``."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, dotted: str) -> Any:
        return _lookup(self._data, dotted)

    def __contains__(self, dotted: str) -> bool:
        try:
            _lookup(self._data, dotted)
        except ConfigError:
            return False
        return True

    def get(self, dotted: str, default: Any = None) -> Any:
        return self[dotted] if dotted in self else default

    def to_dict(self) -> dict:
        """A deep copy of the resolved document, so a caller may mutate it freely."""
        return copy.deepcopy(self._data)

    def dump(self, path: str | Path) -> None:
        """Write the resolved document as YAML, keys in their declared order.

        Every ``${...}`` reference has already been expanded, so the file records the values
        a run actually used rather than the environment it was resolved against.
        """
        Path(path).write_text(yaml.safe_dump(self._data, sort_keys=False))


def load(*overlays: str | Path, cli_overrides: Mapping[str, Any] | None = None,
         env: Mapping[str, str] | None = None) -> Config:
    """Merge every layer over ``base.yaml`` and interpolate once, giving a resolved document.

    Each overlay is a path, or a bare name resolved under :func:`config_dir`, and later
    overlays win over earlier ones. ``env`` defaults to ``os.environ`` and serves both the
    environment layers described above and the ``${env:...}`` references expanded at the
    end, so a caller can resolve a document against a synthetic environment.
    """
    env = os.environ if env is None else env
    document = _read("base.yaml", set())
    for overlay in overlays:
        _merge(document, _read(overlay, set()))
    for extra in env.get("ALPHAWISE_CONFIG", "").split(os.pathsep):
        if extra:
            _merge(document, _read(extra, set()))
    for name, value in env.items():
        if name.startswith(ENV_PREFIX):
            _assign(document, name[len(ENV_PREFIX):].lower().replace("__", "."), _scalar(value))
    for dotted, value in (cli_overrides or {}).items():
        _assign(document, dotted, value)
    resolved = _interpolate(document, env)
    _reject_unparsed_numbers(resolved)
    return Config(resolved)


#: YAML 1.1 requires a signed exponent, so ``1.0e4`` is a string while ``1.0e+4`` is a
#: float. A hyperparameter that silently becomes text survives every check until the line
#: that multiplies by it, so it is rejected here instead.
_UNSIGNED_EXPONENT = re.compile(r"^-?\d+(\.\d*)?[eE]\d+$")


def _reject_unparsed_numbers(node, path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            _reject_unparsed_numbers(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _reject_unparsed_numbers(value, f"{path}[{index}]")
    elif isinstance(node, str) and _UNSIGNED_EXPONENT.match(node):
        raise ConfigError(
            f"{path} is the string {node!r}, not a number: YAML needs a signed exponent. "
            f"Write {node.replace('e', 'e+').replace('E', 'E+')} instead."
        )


def overrides_from_cli(assignments: Iterable[str]) -> dict[str, Any]:
    """Parse repeated ``--set dotted.key=value`` flags into a mapping for :func:`load`."""
    overrides = {}
    for assignment in assignments:
        dotted, separator, value = assignment.partition("=")
        if not separator:
            raise ConfigError(f"expected dotted.key=value, got {assignment!r}")
        overrides[dotted.strip()] = _scalar(value)
    return overrides


def config_dir() -> Path:
    """Where the shipped configs live.

    Resolved at call time rather than at import, because the configs are repository data
    and are not installed into the package. After ``pip install`` the repository-relative
    default no longer exists, so an installed copy must be told where they are.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override).expanduser() if override else REPO_ROOT / "configs"


def _read(name: str | Path, seen: set[Path]) -> dict:
    path = Path(name)
    if not path.exists():
        path = config_dir() / name
    if not path.exists():
        raise ConfigError(
            f"no such config file: {name}. Looked in {config_dir()}. If this package is "
            f"installed rather than run from a checkout, set {CONFIG_DIR_ENV} to the "
            f"configs directory of the checkout."
        )
    path = path.resolve()
    if path in seen:
        return {}
    seen.add(path)
    document = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for include in document.pop("includes", []):
        _merge(merged, _read(include, seen))
    return _merge(merged, document)


def _merge(base: dict, overlay: Mapping) -> dict:
    """Deep merge by key. Scalars and lists replace wholesale; a list is a value."""
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _assign(document: dict, dotted: str, value: Any) -> None:
    """Override one leaf, refusing keys that no config file declares.

    A section marked ``__open__: true`` is a registry rather than a schema, so new keys
    may be created anywhere below it.
    """
    *sections, leaf = dotted.split(".")
    node = document
    is_open = bool(node.get("__open__"))
    for section in sections:
        child = node.get(section)
        if not isinstance(child, dict):
            if not is_open:
                raise ConfigError(f"unknown config key {dotted!r}: {section!r} is not a section")
            child = node[section] = {}
        node = child
        is_open = is_open or bool(node.get("__open__"))
    if leaf not in node and not is_open:
        raise ConfigError(f"unknown config key {dotted!r}; declare it in configs/base.yaml first")
    node[leaf] = value


def _lookup(document: Mapping, dotted: str) -> Any:
    node: Any = document
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise ConfigError(f"unknown config key {dotted!r}")
        node = node[key]
    return node


def _scalar(value: Any) -> Any:
    """Environment and command-line values arrive as text; read them with YAML rules."""
    return yaml.safe_load(value) if isinstance(value, str) else value


def _interpolate(document: dict, env: Mapping[str, str]) -> dict:
    # References may nest inside the default of another reference, so expand repeatedly
    # until a pass changes nothing.
    for _ in range(_MAX_INTERPOLATION_PASSES):
        expanded = _map_strings(document, lambda text: _expand(text, document, env))
        if expanded == document:
            return expanded
        document = expanded
    raise ConfigError("interpolation did not settle; check for a cycle in ${...} references")


def _map_strings(node: Any, expand) -> Any:
    if isinstance(node, dict):
        return {key: _map_strings(value, expand) for key, value in node.items()}
    if isinstance(node, list):
        return [_map_strings(value, expand) for value in node]
    return expand(node) if isinstance(node, str) else node


def _expand(text: str, document: Mapping, env: Mapping[str, str]) -> Any:
    whole = _REFERENCE.fullmatch(text)
    if whole:
        return _resolve(whole.group(1), document, env)
    return _REFERENCE.sub(lambda m: str(_resolve(m.group(1), document, env)), text)


def _resolve(reference: str, document: Mapping, env: Mapping[str, str]) -> Any:
    if reference == "repo_root":
        return str(REPO_ROOT)
    if reference.startswith("env:"):
        name, separator, default = reference[len("env:"):].partition(":-")
        if name in env:
            return _scalar(env[name])
        if not separator:
            raise ConfigError(f"environment variable {name} is required but not set")
        return _scalar(default)
    return _lookup(document, reference)
