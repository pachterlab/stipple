r"""Write Pearson residuals -- and their capture-rate-regressed-out
residuals -- into ``adata.obsm``, from a fitted :class:`stipple.model.StippleModel`.

Standalone from :func:`stipple.tools.classify_spatial_variable_genes` (which
computes the same two quantities internally, for both residual bases at
once, purely to feed its own Moran's I calls) -- callable on its own so a
caller doesn't have to run gene classification just to get these residuals
written into ``adata``, and vice versa.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anndata import AnnData

from ._residuals import _group_zscore_residuals, _regress_out_capture_rate

if TYPE_CHECKING:
    from ..model import StippleModel

_RESIDUAL_BASES = ("naive", "model")


def set_residual_layers(model: "StippleModel", adata: AnnData, basis: Literal["naive", "model"] = "naive") -> None:
    """Write a Pearson residual and its capture-rate-regressed-out residual
    into ``adata.obsm``, in place.

    ``basis="naive"``: the Pearson residual is each gene's raw counts
    z-scored *within its own group*, using the group's own empirical
    mean/sd of raw counts (``ddof=1``) -- it ignores the model's fitted
    mean/variance entirely. ``basis="model"``: the Pearson residual is
    :meth:`StippleModel.get_residuals`'s (``type="pearson"``), from the
    MAP fit's mean/variance.

    Either way, a second quantity is also written: the OLS residual of
    regressing that Pearson residual on the estimated capture rate
    (:meth:`StippleModel.get_capture_rates`) -- spatial/technical signal
    remaining after removing the capture-rate field's linear effect. This
    is **not** itself a Pearson residual (it's a residual of a residual,
    from an unrelated second regression); see the module docstring for why
    it's named separately.

    Parameters
    ----------
    model
        A fitted :class:`stipple.model.StippleModel`.
    adata
        Must have the same number of locations/genes, in the same order,
        as the data this model was fit on (mirrors
        :meth:`StippleModel.write_obsm`'s validation).
    basis
        ``"naive"`` (default) or ``"model"`` -- which Pearson residual to
        compute (see above). Selects the ``adata.obsm`` slot names, both
        suffixed by this value: ``"stipple_pearson_residuals_{basis}"``
        and ``"stipple_capture_rate_ols_residuals_{basis}"``.
    """
    if basis not in _RESIDUAL_BASES:
        raise ValueError(f"basis must be one of {_RESIDUAL_BASES}, got {basis!r}")
    model._check_fitted()
    if adata.n_obs != model.data.n:
        raise ValueError(
            f"adata.n_obs ({adata.n_obs}) does not match the number of "
            f"locations this model was fit on ({model.data.n})"
        )
    if adata.n_vars != model.data.G:
        raise ValueError(
            f"adata.n_vars ({adata.n_vars}) does not match the number of "
            f"genes this model was fit on ({model.data.G})"
        )

    if basis == "naive":
        Y_ng = model.model.Y_ng.numpy()
        group_idx = model.data.group_idx.numpy()
        pearson_resid = _group_zscore_residuals(Y_ng, group_idx, model.data.K)
    else:
        pearson_resid = model.get_residuals(type="pearson").T  # (G, n) -> (n, G)

    p_hat = model.get_capture_rates()
    ols_resid = _regress_out_capture_rate(pearson_resid, p_hat)

    adata.obsm[f"stipple_pearson_residuals_{basis}"] = pearson_resid
    adata.obsm[f"stipple_capture_rate_ols_residuals_{basis}"] = ols_resid
