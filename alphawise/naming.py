"""Canonical vocabulary shared by every stage of the pipeline: one spelling per concept.

The renderers, the evaluators and the configs used to carry private copies of the
method list, the retrieval-direction spellings and the transfer-table row mapping,
and the copies could drift: a renderer that keys a method differently from the
checkpoint registry silently renders its rows as dashes.
Both sides of every such seam now import from here, so a disagreement is an
ImportError or a KeyError instead of a silently empty row.

Vocabulary this module pins down:

* **method key**: the registry spelling of a run, for example ``alphawise_n_ewc``.
  A fused method is named after its two endpoints, plastic first. In the paper's
  notation those endpoints are ``theta_un`` and ``theta_reg``: ``un`` is the
  plastic endpoint, the one multiplied by alpha, and ``reg`` is the stability
  endpoint, weighted by ``1 - alpha``. The slot is a role, not a method, which is
  why EWC sits in the ``un`` slot of the ``ewc_lwf`` pair.
* **direction**: a retrieval direction, ``a2t``, ``i2a`` or ``i2t``. The word
  never means anything else; the forward/reverse axis is a class order.
* **class order**: ``forward`` or ``reverse``, the axis that selects the class
  partition and the phase-0 bootstrap a run trained under.
* **phase**: one continual step. Phase 0 is the shared bootstrap; phases 1..7 are
  the trained ones.
"""

from __future__ import annotations

from typing import Sequence

#: Method key -> display label, in default table row order: the published Table 1
#: order, with the non-paper rows (Normal, VR-LwF, the extra fusions, the joint
#: ceiling) in the positions the development tree used. Artifact configs narrow
#: and reorder this via their ``methods:`` list; they never respell it.
METHOD_ORDER: tuple[tuple[str, str], ...] = (
    ("normal", "Normal"),
    ("lwf", "LwF"),
    ("vlwf", "VR-LwF"),
    ("ewc", "EWC"),
    ("der", "DER"),
    ("icarl", "iCaRL"),
    ("cclip", "C-CLIP"),
    ("wiseft_n_ewc", "WiSE-FT (N+EWC)"),
    ("wiseft_n_icarl", "WiSE-FT (N+iCaRL)"),
    ("wiseft_n_lwf", "WiSE-FT (N+LwF)"),
    ("alphawise_n_ewc", "AlphaWiSE (N+EWC)"),
    ("alphawise_n_lwf", "AlphaWiSE (N+LwF)"),
    ("alphawise_n_icarl", "AlphaWiSE (N+iCaRL)"),
    ("alphawise_ewc_lwf", "AlphaWiSE (EWC+LwF)"),
    ("alphawise_vlwf_ewc", "AlphaWiSE (VR-LwF+EWC)"),
    ("upper_bound", "Joint (all-classes)"),
)

METHOD_LABELS: dict[str, str] = dict(METHOD_ORDER)

#: The joint model is the ceiling, not a competing method, so it never takes a bold.
NON_COMPETING: frozenset[str] = frozenset({"upper_bound"})

#: Retrieval directions, in the published Table 1 column-group order.
DIRECTIONS: tuple[str, ...] = ("a2t", "i2a", "i2t")
DIRECTION_LABELS: dict[str, str] = {"a2t": "A→T", "i2a": "I→A", "i2t": "I→T"}
DIRECTION_LABELS_TEX: dict[str, str] = {
    "a2t": r"A$\rightarrow$T",
    "i2a": r"I$\rightarrow$A",
    "i2t": r"I$\rightarrow$T",
}

#: The forward/reverse axis. Values live in ``checkpoints.registry.<class order>``.
CLASS_ORDERS: tuple[str, ...] = ("forward", "reverse")

#: Phase 0 is the shared bootstrap every trajectory starts from. It backs the FWT
#: baseline and is never a row of any Average.
BOOTSTRAP_PHASE: int = 0

METRICS: tuple[str, ...] = ("r1", "map")
METRIC_LABELS: dict[str, str] = {"r1": "R@1", "map": "mAP"}

#: How a method's per-phase scores collapse into the results table.
AGGREGATIONS: tuple[str, ...] = ("avg", "last")
AGGREGATION_LABELS: dict[str, str] = {"avg": "Average", "last": "Last Phase"}


def metric_cols(directions: Sequence[str] = DIRECTIONS) -> tuple[str, ...]:
    """The per-cell metric columns the evaluator writes, ``a2t_r1`` .. ``i2t_map``."""
    return tuple(f"{direction}_{metric}" for direction in directions for metric in METRICS)


def table_cols(directions: Sequence[str] = DIRECTIONS) -> tuple[str, ...]:
    """The collapsed results-table columns, ``a2t_avg_r1`` .. ``i2t_last_map``."""
    return tuple(
        f"{direction}_{aggregation}_{metric}"
        for direction in directions
        for aggregation in AGGREGATIONS
        for metric in METRICS
    )


#: Registry key -> (display method, variant) for the transfer table. The summarize
#: stage (``alphawise.eval.transfer``) writes these two columns into its measures CSV
#: and the renderer (``alphawise.analysis.tables``) requires them back, so the two
#: stages cannot drift: a key outside this mapping has no transfer-table row.
TRANSFER_DISPLAY: dict[str, tuple[str, str]] = {
    "ewc": ("EWC", "Base"),
    "icarl": ("iCaRL", "Base"),
    "lwf": ("LwF", "Base"),
    "alphawise_n_ewc": ("EWC", "+ AlphaWiSE"),
    "alphawise_n_icarl": ("iCaRL", "+ AlphaWiSE"),
    "alphawise_n_lwf": ("LwF", "+ AlphaWiSE"),
}
#: Row-block and row order of the rendered transfer table.
TRANSFER_METHODS: tuple[str, ...] = ("EWC", "iCaRL", "LwF")
TRANSFER_VARIANTS: tuple[str, ...] = ("Base", "+ AlphaWiSE")

#: Fusion pair -> method key of the fused run it produces. The pair string is the
#: ``<plastic>_<stable>`` stem shared by the fusion overlay filename and the coefficient
#: files ``alphawise.analysis.alpha_profile`` reads.
FUSION_PAIRS: dict[str, str] = {
    "normal_ewc": "alphawise_n_ewc",
    "normal_lwf": "alphawise_n_lwf",
    "normal_icarl": "alphawise_n_icarl",
    "ewc_lwf": "alphawise_ewc_lwf",
    "vlwf_ewc": "alphawise_vlwf_ewc",
}
