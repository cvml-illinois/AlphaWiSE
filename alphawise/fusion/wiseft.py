"""WiSE-FT baseline: interpolation with one constant coefficient, no fitting.

This is AlphaWiSE with every ``alpha_p`` held at the same value, so it reuses
:class:`~alphawise.fusion.core.TensorWiseInterpolation` rather than reimplementing the
blend. At ``alpha=0.5`` the result is the uniform average of the two endpoints.

The checkpoint-producing route is ``fusion.alpha`` in
``configs/fusion/wiseft_half.yaml``, driven by ``python -m alphawise.fusion.fit``,
which writes the same filenames as a fitted run so the evaluation registry needs no
special case. This module is the one-call form of the same operation for library use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from torch import Tensor

from .core import TensorWiseInterpolation

Endpoint = Union[str, Path]


def fixed_blend(endpoint_un: Endpoint, endpoint_reg: Endpoint, *, alpha: float = 0.5,
                device=None) -> dict[str, Tensor]:
    """The materialized state dict of ``alpha * theta_un + (1 - alpha) * theta_reg``.

    ``endpoint_un`` is the plastic endpoint, the one alpha multiplies; ``endpoint_reg``
    is the stability endpoint. The slots are roles, not methods.
    """
    model = TensorWiseInterpolation(
        endpoint_un, endpoint_reg, init_alpha=alpha, device=device
    ).eval()
    return model.materialize()
