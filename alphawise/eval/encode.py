"""Checkpoint loading and subset encoding for the embedding-space analyses.

This is a refactor of ``_import/triclip/scripts/create_tsne_it.py`` (``load_model``,
``extract_pair_embeddings``), ported so nothing under ``_import/`` is executed. It
turns an :class:`~alphawise.eval.subset.EvalSubset` into per-modality embedding
arrays: one row per sample for image and audio, one row per class for text, plus an
optional background array with one row per unselected class.

Two orderings are load-bearing. Samples are encoded in subset order with one shared
skip decision per pair, so row ``i`` of the image array is the same clip as row ``i``
of the audio array and the image-audio panel stays genuinely paired. And labels index
``subset.classes``, whose order fixes the palette, so the arrays carry that mapping
unchanged.

Everything heavy (torch, the model stack and with it torchvision, PIL, librosa) is
imported inside the functions: importing this module costs numpy only, which keeps
the analysis modules importable in a torch-only environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .subset import EvalSubset

ENCODE_BATCH = 32

#: The per-sample modalities; text is encoded per class, not per sample.
SAMPLE_MODALITIES = ("image", "audio")
MODALITIES = (*SAMPLE_MODALITIES, "text")


@dataclass(frozen=True)
class SubsetEmbeddings:
    """One checkpoint's view of the subset; unrequested modalities stay None."""

    image: Optional[np.ndarray] = None
    image_labels: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    audio_labels: Optional[np.ndarray] = None
    text: Optional[np.ndarray] = None
    text_labels: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None


def load_model(checkpoint: str | Path, *, device: str = "cpu", assets_dir=None):
    """A registered checkpoint as an evaluation-ready backbone on ``device``.

    Delegates to the evaluation runner's loader so the analyses score exactly the
    model the tables scored, prefix stripping and the missing-key tripwire included.
    ``assets_dir`` overrides where the pretrained CLIP weights are looked up, through
    the same environment variable :func:`alphawise.utils.paths.assets_dir` reads.
    """
    import os

    import torch

    if assets_dir is not None:
        os.environ["ALPHAWISE_ASSETS"] = str(assets_dir)
    from .runner import load_checkpoint  # heavy: pulls the model stack and torchvision

    return load_checkpoint(Path(checkpoint), torch.device(device))


def _image_loader():
    from PIL import Image

    from ..data.audioset import image_transforms  # heavy: pulls torchvision

    transform = image_transforms()
    return lambda path: transform(Image.open(path).convert("RGB"))


def _audio_loader():
    import librosa

    from ..data.audioset import SAMPLE_RATE, WAV_SCALE, audio_transforms

    transform = audio_transforms(augment=False)

    def load(path):
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if wav.ndim == 1:
            wav = wav[:, np.newaxis]
        return transform((wav.T * WAV_SCALE).astype(np.float32))

    return load


def _encode_batch(model, modality: str, tensors, device) -> np.ndarray:
    import torch

    batch = torch.stack(tensors).to(device)
    if modality == "image":
        ((_, features, _), _), _ = model(image=batch)
    else:
        indices = torch.arange(batch.shape[0], dtype=torch.int64, device=device)
        ((features, _, _), _), _ = model(audio=batch, batch_indices=indices)
    return features.cpu().numpy()


def encode_subset(
    model,
    subset: EvalSubset,
    *,
    modalities: frozenset | set,
    batch_size: int = ENCODE_BATCH,
    encode_background: bool = False,
    background_modality: str = "image",
) -> SubsetEmbeddings:
    """Encode the subset's requested modalities with one checkpoint.

    A pair whose decode fails in any requested modality is skipped whole, announced on
    stdout, so the image and audio rows stay index-aligned. Text is one embedding per
    class, restricted (as the ancestor restricted it) to the classes that actually
    yielded sample embeddings, with labels naming their palette slots. The background
    pairs are encoded in ``background_modality`` only, one row each.

    Embeddings are returned unnormalised, exactly as the ancestor plotted them; the
    projection downstream compares them by cosine distance either way.
    """
    import torch

    unknown = set(modalities) - set(MODALITIES)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; expected among {MODALITIES}")
    if encode_background and background_modality not in SAMPLE_MODALITIES:
        raise ValueError(f"background_modality must be one of {SAMPLE_MODALITIES}, "
                         f"got {background_modality!r}")

    sample_mods = [m for m in SAMPLE_MODALITIES if m in modalities]
    load_by_mod = {}
    for modality in set(sample_mods) | ({background_modality} if encode_background else set()):
        load_by_mod[modality] = _image_loader() if modality == "image" else _audio_loader()

    device = next(model.parameters()).device
    fields: dict = {}
    with torch.no_grad():
        if "text" in modalities:
            ((_, _, text_features), _), _ = model(text=[[cls] for cls in subset.classes])
            text_all = text_features.cpu().numpy()

        embeddings = {m: [] for m in sample_mods}
        labels = {m: [] for m in sample_mods}
        for class_index, cls in enumerate(subset.classes):
            loaded = {m: [] for m in sample_mods}
            for image_path, wav_path in subset.samples[cls]:
                try:
                    decoded = {m: load_by_mod[m](image_path if m == "image" else wav_path)
                               for m in sample_mods}
                except Exception as error:
                    print(f"  skip {Path(image_path).name}: {error}", flush=True)
                    continue
                for m in sample_mods:
                    loaded[m].append(decoded[m])
            for start in range(0, len(loaded[sample_mods[0]]) if sample_mods else 0, batch_size):
                for m in sample_mods:
                    features = _encode_batch(model, m, loaded[m][start:start + batch_size], device)
                    embeddings[m].append(features)
                    labels[m].extend([class_index] * len(features))

        for m in sample_mods:
            fields[m] = np.concatenate(embeddings[m], axis=0)
            fields[f"{m}_labels"] = np.array(labels[m])

        if "text" in modalities:
            # One text row per class that yielded sample embeddings; with no sample
            # modality requested there is nothing to restrict by, so every class stays.
            present = (np.unique(np.concatenate([fields[f"{m}_labels"] for m in sample_mods]))
                       if sample_mods else np.arange(len(subset.classes)))
            fields["text"] = text_all[present]
            fields["text_labels"] = present

        if encode_background and subset.background:
            load = load_by_mod[background_modality]
            tensors = []
            for image_path, wav_path in subset.background:
                try:
                    tensors.append(load(image_path if background_modality == "image"
                                        else wav_path))
                except Exception as error:
                    print(f"  skip background {Path(image_path).name}: {error}", flush=True)
            parts = [_encode_batch(model, background_modality,
                                   tensors[start:start + batch_size], device)
                     for start in range(0, len(tensors), batch_size)]
            if parts:
                fields["background"] = np.concatenate(parts, axis=0)

    return SubsetEmbeddings(**fields)
