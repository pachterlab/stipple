"""Standalone post-fit residual utilities, operating on a fitted-and-written
``adata`` rather than a live :class:`stipple.model.StippleModel`.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.stats import nbinom, norm, poisson

_DUNN_SMYTH_EPS: float = 1e-10  # clip u away from {0, 1} before norm.ppf, which is +-inf there
_PEARSON_EPS: float = 1e-10  # floor var before sqrt, guards a zero-variance edge case

__all__ = [
    "compute_dunn_smyth_residuals",
    "set_dunn_smyth_residuals",
    "compute_pearson_residuals",
    "set_pearson_residuals",
]


def _randomized_quantile_residuals(
    observed: np.ndarray, mean: np.ndarray, var: np.ndarray, likelihood: str, rng: np.random.Generator,
) -> np.ndarray:
    """Randomized quantile (Dunn-Smyth) residuals: ``r = Phi^-1(u)``,
    ``u ~ Uniform(F(y-1), F(y))`` under the fitted Poisson/NegBin CDF at
    each entry's own ``(mean, variance)``.

    Parameters
    ----------
    observed, mean, var : np.ndarray
        Must be broadcastable to a common shape.
    likelihood
        ``"poisson"`` or ``"negbinomial"``.
    rng
        Source of the randomization draw.
    """
    if likelihood == "poisson":
        lower = poisson.cdf(observed - 1, mean)
        upper = poisson.cdf(observed, mean)
    else:
        phi = np.clip(mean**2 / np.maximum(var - mean, 1e-6), 1e-3, 1e6)
        p = phi / (phi + mean)
        lower = nbinom.cdf(observed - 1, phi, p)
        upper = nbinom.cdf(observed, phi, p)
    u = rng.uniform(lower, upper)
    return norm.ppf(np.clip(u, _DUNN_SMYTH_EPS, 1.0 - _DUNN_SMYTH_EPS))


def compute_dunn_smyth_residuals(adata: AnnData, random_state=None) -> np.ndarray:
    """Randomized quantile (Dunn-Smyth) residuals at every (location,
    gene) pair, from a fitted model's results already written into
    ``adata`` (:meth:`stipple.model.StippleModel.set_anndata_results` or
    at least ``write_obsm``/``write_uns``).

    Parameters
    ----------
    adata
        Must already have fitted results written (see above).
    random_state
        Seed (or ``numpy.random.Generator``) for the randomized-quantile
        draw. ``None`` (default) draws fresh randomness each call.

    Returns
    -------
    np.ndarray, shape (n, G)
    """
    from ..plotting._utils import _get_counts

    if "stipple_fitted_mean" not in adata.obsm:
        raise RuntimeError(
            "adata.obsm['stipple_fitted_mean'] not found -- run "
            "model.set_anndata_results(adata) (or model.write_obsm(adata)) first."
        )
    observed = _get_counts(adata)
    mean = np.asarray(adata.obsm["stipple_fitted_mean"])
    var = np.asarray(adata.obsm["stipple_fitted_variance"])
    likelihood = adata.uns["stipple"]["likelihood"]
    rng = np.random.default_rng(random_state)
    return _randomized_quantile_residuals(observed, mean, var, likelihood, rng)


def set_dunn_smyth_residuals(adata: AnnData, random_state=None) -> None:
    """Compute Dunn-Smyth residuals (:func:`compute_dunn_smyth_residuals`)
    and write them into ``adata.obsm["stipple_dunn_smyth_residuals"]``, in
    place.

    Parameters
    ----------
    adata
        Must already have fitted results written (see
        :func:`compute_dunn_smyth_residuals`).
    random_state
        Forwarded to :func:`compute_dunn_smyth_residuals`.
    """
    adata.obsm["stipple_dunn_smyth_residuals"] = compute_dunn_smyth_residuals(adata, random_state=random_state)


def compute_pearson_residuals(adata: AnnData) -> np.ndarray:
    """Pearson residuals at every (location, gene) pair:
    ``(observed - mean) / sqrt(maximum(var, eps))``, from a fitted
    model's results already written into ``adata``.

    Parameters
    ----------
    adata
        Must already have fitted results written (see
        :func:`compute_dunn_smyth_residuals`).

    Returns
    -------
    np.ndarray, shape (n, G)
    """
    from ..plotting._utils import _get_counts

    if "stipple_fitted_mean" not in adata.obsm:
        raise RuntimeError(
            "adata.obsm['stipple_fitted_mean'] not found -- run "
            "model.set_anndata_results(adata) (or model.write_obsm(adata)) first."
        )
    observed = _get_counts(adata)
    mean = np.asarray(adata.obsm["stipple_fitted_mean"])
    var = np.asarray(adata.obsm["stipple_fitted_variance"])
    return (observed - mean) / np.sqrt(np.maximum(var, _PEARSON_EPS))


def set_pearson_residuals(adata: AnnData) -> None:
    """Compute Pearson residuals (:func:`compute_pearson_residuals`) and
    write them into ``adata.obsm["stipple_pearson_residuals"]``, in place.

    Parameters
    ----------
    adata
        Must already have fitted results written (see
        :func:`compute_pearson_residuals`).
    """
    adata.obsm["stipple_pearson_residuals"] = compute_pearson_residuals(adata)
