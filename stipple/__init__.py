"""Spatial radial-basis-function model of per-location RNA capture
efficiency.

- :mod:`stipple.model` -- :class:`stipple.model.StippleModel`, the fitting
  entry point.
- :mod:`stipple.plotting` -- diagnostic/spatial plotting functions, read
  from a fitted ``adata``.
- :mod:`stipple.utils` -- standalone post-fit utilities, e.g.
  ``compute_dunn_smyth_residuals``/``set_dunn_smyth_residuals``,
  ``compute_pearson_residuals``/``set_pearson_residuals``.
- :mod:`stipple.tools` -- ``group_de_wald_test``.
"""
from __future__ import annotations

from . import model, plotting, tools, utils

__version__ = "0.12.0"
__all__ = ["__version__", "model", "plotting", "utils", "tools"]
