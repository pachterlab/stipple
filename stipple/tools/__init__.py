"""Statistical tests operating on a fitted :class:`stipple.model.StippleModel`.

Public API: :func:`group_de_wald_test` -- Fisher-information (Wald) test
for pairwise between-group differential expression. :func:`classify_spatial_variable_genes`
-- two-stage spatial/differential gene classification, called after
:func:`group_de_wald_test`. :func:`set_residual_layers` -- write Pearson
residuals (and their capture-rate-regressed-out residuals) into ``adata``,
standalone from gene classification.
"""
from __future__ import annotations

from .classify_spatial_variable_genes import classify_spatial_variable_genes
from .group_de_wald_test import group_de_wald_test
from .set_residual_layers import set_residual_layers

__all__ = ["group_de_wald_test", "classify_spatial_variable_genes", "set_residual_layers"]
