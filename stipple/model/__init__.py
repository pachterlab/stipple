"""Spatial-field model and MAP-fitting machinery.

Public API: :class:`StippleModel` -- the fitting entry point;
:func:`run_map_optimization`; :func:`resolve_n_knots_target`.
"""
from __future__ import annotations

from .model import run_map_optimization
from .stipple_model import StippleModel
from .utils import MIN_KNOTS_RATIO, resolve_n_knots_target

__all__ = [
    "StippleModel",
    "run_map_optimization",
    "resolve_n_knots_target",
    "MIN_KNOTS_RATIO",
]
