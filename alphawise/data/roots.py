"""Where the AudioSet corpus lives.

Resolution order per location: explicit argument, then environment variable, then a
default under the data root. The defaults mirror the ones declared in `configs/base.yaml`
so the config layer and a bare `DataRoots.from_env()` agree on the same layout. No
site-specific path is ever written down here.
"""

import os
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "eval")

# (images, wavs, metadata csv) relative to the data root, per split.
_SPLIT_LAYOUT = {
    "train": ("train/img", "train/wav", "train/unbalanced_train_segments.csv"),
    "eval": ("eval/eval_img", "eval/eval_wav", "eval/eval_segments.csv"),
}
_ONTOLOGY = "ontology/ontology.json"


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def _resolve(explicit, env_var: str, default: Path) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class DataRoots:
    """The four locations one split of the corpus is read from."""

    images: Path
    wavs: Path
    meta_csv: Path
    ontology: Path

    @classmethod
    def from_config(cls, config, split: str) -> "DataRoots":
        """Take the roots from a resolved config document, which has already read the
        same environment variables `from_env` reads."""
        _check_split(split)
        return cls(
            images=Path(config[f"data.{split}.img_root"]),
            wavs=Path(config[f"data.{split}.wav_root"]),
            meta_csv=Path(config[f"data.{split}.meta_csv"]),
            ontology=Path(config["data.ontology"]),
        )

    @classmethod
    def from_env(cls, split: str, *, images=None, wavs=None, meta_csv=None,
                 ontology=None, data_root=None) -> "DataRoots":
        """The roots for one split, for callers that have no resolved config document.

        Each location falls back to `AUDIOSET_<SPLIT>_IMG`, `_WAV` or `_CSV`, then to this
        split's slot in the layout under `ALPHAWISE_DATA_ROOT`, itself defaulting to a
        relative `data`. The ontology is shared by both splits and comes from `AUDIOSET_ONT`.
        """
        _check_split(split)
        root = _resolve(data_root, "ALPHAWISE_DATA_ROOT", Path("data"))
        rel_images, rel_wavs, rel_csv = _SPLIT_LAYOUT[split]
        prefix = f"AUDIOSET_{split.upper()}"
        return cls(
            images=_resolve(images, f"{prefix}_IMG", root / rel_images),
            wavs=_resolve(wavs, f"{prefix}_WAV", root / rel_wavs),
            meta_csv=_resolve(meta_csv, f"{prefix}_CSV", root / rel_csv),
            ontology=_resolve(ontology, "AUDIOSET_ONT", root / _ONTOLOGY),
        )
