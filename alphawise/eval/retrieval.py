"""Directed retrieval metrics over the per-phase candidate pool.

Three directions, R@1 and mAP each, ranked inside one phase's ~79-class candidate
pool (the non-cascading eval pool):

    A->T   audio query, class-text candidates
    I->T   image query, class-text candidates
    I->A   image query, per-sample audio bank, relevance = shared label

This is a refactor of the pre-release evaluation module, itself a byte-equivalent
port of the reference retrieval loops that produced the published numbers. The
numerics are load-bearing and are reproduced operation for operation, including
two things that would otherwise look like defects:

* the softmax before the argmax in the two text directions. It is monotonic, so it
  cannot change R@1, but it is kept so the published numbers stay reproducible;
* mAP averages per-query AP only over queries with at least one relevant candidate,
  and I->A skips a query whose audio bank contains no relevant item at all.

The six-key per-phase aggregation and the average over seen phases descend from the
pre-release aggregation and metric modules.
"""

from typing import Iterable, Mapping, NamedTuple

import torch
import tqdm

from ..naming import metric_cols

#: The six metric columns, ``a2t_r1`` .. ``i2t_map``, in the canonical direction
#: order. Sourced from :mod:`alphawise.naming` so this module cannot respell what
#: the runner writes and the tables read.
METRIC_KEYS = metric_cols()


class RetrievalScore(NamedTuple):
    """One direction's score on one phase. Both fields are fractions in [0, 1]."""

    mean_ap: float
    r_at_1: float


def _class_text_embeddings(model, dataset):
    """(1, num_classes, dim) text bank for the phase's candidate pool."""
    texts_for_all_classes = [[c] for c in dataset.text_classes]
    ((_, _, text_emb_all), _), _ = model(text=texts_for_all_classes)
    return text_emb_all.unsqueeze(1).transpose(0, 1)


def _encode_queries(model, queries, modality):
    """Query embeddings and the clamped temperature that ranks them against text."""
    batch_indices = torch.arange(queries.shape[0], dtype=torch.int64, device=queries.device)
    if modality == "audio":
        ((features, _, _), _), _ = model(audio=queries, batch_indices=batch_indices)
        scale = model.logit_scale_at
    else:
        ((_, features, _), _), _ = model(image=queries, batch_indices=batch_indices)
        # I->T ranks with CLIP's own temperature, not the audio-text one.
        scale = model.logit_scale
    return features, torch.clamp(scale.exp(), min=1.0, max=100.0)


def _average_precision(scores, relevant):
    """AP of one query, ranking the whole candidate pool by ``scores``.

    ``relevant`` is the set of candidate indices carrying one of the query's labels. It
    must be non-empty, since the denominator is its size; callers filter empty label sets
    out rather than scoring them as zero.
    """
    ranked_classes = scores.argsort(descending=True)
    hits = 0.0
    sum_precision = 0.0
    for rank, cls_idx in enumerate(ranked_classes, start=1):
        if cls_idx.item() in relevant:
            hits += 1
            precision_at_rank = hits / rank
            sum_precision += precision_at_rank
    return sum_precision / len(relevant)


@torch.no_grad()
def _retrieve_class_text(model, loader, dataset, device, modality, desc):
    """Shared A->T and I->T loop: rank the class-text bank for every query in the loader.

    R@1 counts a query correct when the single top-scoring class is any one of its labels,
    so on multi-label samples it is a hit-at-one, not an exact-set match.
    """
    text_emb_all = _class_text_embeddings(model, dataset)

    correct_count = 0
    total_queries = 0
    ap_scores = []

    for wav, img, text_s in tqdm.tqdm(loader, desc=desc):
        queries = torch.stack(wav if modality == "audio" else img).to(device)
        query_features, logit_scale = _encode_queries(model, queries, modality)
        y_pred = (logit_scale * query_features.unsqueeze(1) @ text_emb_all.transpose(-1, -2)).squeeze(1)
        y_pred_classes = y_pred.softmax(dim=-1).argmax(dim=-1)

        for i, labels in enumerate(text_s):
            total_queries += 1
            relevant = set(dataset.class_to_idx[label] for label in labels)
            if y_pred_classes[i].item() in relevant:
                correct_count += 1
            if relevant:
                ap_scores.append(_average_precision(y_pred[i], relevant))

    r_at_1 = correct_count / total_queries if total_queries > 0 else 0.0
    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    return RetrievalScore(mean_ap, r_at_1)


def audio_to_text(model, loader, dataset, device):
    """Audio query against the phase's class-text candidates.

    Every eval clip of the phase is a query; the candidates are the phase's ~79 class
    names, one text embedding each, and a candidate is relevant when the query carries
    that label. Scored with the audio-text temperature ``logit_scale_at``.
    """
    return _retrieve_class_text(model, loader, dataset, device, "audio", "A->T")


def image_to_text(model, loader, dataset, device):
    """Image query against the same class-text candidates as :func:`audio_to_text`.

    The candidate pool is identical, so the two directions are directly comparable. Only
    the temperature differs: CLIP's own ``logit_scale``, which is monotonic and therefore
    changes neither R@1 nor mAP.
    """
    return _retrieve_class_text(model, loader, dataset, device, "image", "I->T")


@torch.no_grad()
def image_to_audio(model, loader, dataset, device, image_loader=None):
    """Image -> audio over a per-sample audio bank; relevance is a shared label.

    ``image_loader`` is an optional second DataLoader. When it is None the same
    loader is iterated twice, which matches the reference (it rebuilds a loader
    after exhausting the first one).
    """
    all_audio_embs = []
    all_label_sets = []
    for wav, _, text_lbls in tqdm.tqdm(loader, desc="I->A pass1"):
        wav = torch.stack(wav).to(device)
        ((audio_feats, _, _), _), _ = model(
            audio=wav,
            batch_indices=torch.arange(wav.shape[0], dtype=torch.int64, device=device),
        )
        all_audio_embs.append(audio_feats.cpu())
        all_label_sets.extend(
            [set(dataset.class_to_idx[l] for l in lbl) for lbl in text_lbls]
        )

    audio_bank = torch.cat(all_audio_embs, dim=0).to(device)
    num_bank = audio_bank.shape[0]

    num_classes = len(dataset.text_classes)
    audio_label_mat = torch.zeros(num_bank, num_classes, dtype=torch.bool, device=device)
    for j, lbls in enumerate(all_label_sets):
        for c in lbls:
            audio_label_mat[j, c] = True

    logit_scale_ai = getattr(model, "logit_scale_ai", model.logit_scale).exp().clamp(1.0, 100.0)

    correct_at1 = 0
    total_queries = 0
    ap_scores = []

    img_loader = image_loader if image_loader is not None else loader
    ranks_t = torch.arange(1, num_bank + 1, device=device, dtype=torch.float32)
    for _, imgs, text_lbls in tqdm.tqdm(img_loader, desc="I->A pass2"):
        imgs = torch.stack(imgs).to(device)
        ((_, img_feats, _), _), _ = model(
            image=imgs,
            batch_indices=torch.arange(imgs.shape[0], dtype=torch.int64, device=device),
        )

        sims = logit_scale_ai * img_feats @ audio_bank.t()
        ranked = sims.argsort(descending=True)
        batch_size = img_feats.shape[0]

        query_label_mat = torch.zeros(batch_size, num_classes, dtype=torch.bool, device=device)
        for i, lbls in enumerate(text_lbls):
            for l in lbls:
                query_label_mat[i, dataset.class_to_idx[l]] = True
        relevant_mat = (query_label_mat.float() @ audio_label_mat.float().t()) > 0

        total_queries += batch_size
        correct_at1 += relevant_mat[torch.arange(batch_size, device=device), ranked[:, 0]].sum().item()

        for i in range(batch_size):
            rel_mask = relevant_mat[i][ranked[i]].float()
            n_relevant = rel_mask.sum().item()
            if n_relevant == 0:
                continue
            cumhits = rel_mask.cumsum(0)
            sum_prec = ((cumhits / ranks_t) * rel_mask).sum().item()
            ap_scores.append(sum_prec / n_relevant)

    r_at_1 = correct_at1 / total_queries if total_queries > 0 else 0.0
    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    return RetrievalScore(mean_ap, r_at_1)


def evaluate_phase(model, loader, dataset, device):
    """All six metrics for one phase's candidate pool, keyed by :data:`METRIC_KEYS`.

    Each cell is assigned by direction name, never by position: the legacy key order
    (at, it, ia) and the canonical order (a2t, i2a, i2t) disagree, so a positional
    zip would silently swap the two image directions.
    """
    at = audio_to_text(model, loader, dataset, device)
    it = image_to_text(model, loader, dataset, device)
    ia = image_to_audio(model, loader, dataset, device)
    return {
        "a2t_r1": at.r_at_1, "a2t_map": at.mean_ap,
        "i2a_r1": ia.r_at_1, "i2a_map": ia.mean_ap,
        "i2t_r1": it.r_at_1, "i2t_map": it.mean_ap,
    }


def average_metrics(per_phase: Iterable[Mapping[str, float]]):
    """Unweighted mean of each metric over the phases a checkpoint has seen.

    The paper's retention score: a phase-N checkpoint is scored on phase loaders
    0..N and the phases are averaged with equal weight, whatever their size. The
    last-phase value quoted next to it is simply the highest-phase entry.
    """
    scored = list(per_phase)
    return {k: sum(p[k] for p in scored) / len(scored) for k in METRIC_KEYS}
