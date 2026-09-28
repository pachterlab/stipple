"""Private residual helpers shared by :func:`stipple.tools.set_residual_layers`
and :func:`stipple.tools.classify_spatial_variable_genes`."""
from __future__ import annotations

import numpy as np


def _group_zscore_residuals(Y_ng: np.ndarray, group_idx: np.ndarray, K: int) -> np.ndarray:
    """"Naive" Pearson residual: z-score each gene's raw counts *within
    its own group*, using the group's own empirical mean/sd of raw counts
    (``ddof=1``) -- not the model's fitted mean/variance.

    Parameters
    ----------
    Y_ng : np.ndarray, shape (n, G)
        Raw observed counts.
    group_idx : np.ndarray, shape (n,)
        Integer group assignment, in ``[0, K)``.
    K : int

    Returns
    -------
    np.ndarray, shape (n, G)
    """
    resid = np.empty_like(Y_ng, dtype=float)
    for k in range(K):
        mask = group_idx == k
        group_vals = Y_ng[mask]  # (n_k, G)
        group_mean = group_vals.mean(axis=0)
        group_sd = group_vals.std(axis=0, ddof=1)
        resid[mask] = (group_vals - group_mean) / group_sd
    return resid


def _regress_out_capture_rate(Y_ng: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
    """OLS residuals of regressing every gene's column in ``Y_ng`` on the
    shared covariate ``p_hat`` (``lm(y ~ p_hat)``'s ``resid()``), vectorized
    across genes via one shared closed-form simple regression. Not itself a
    Pearson residual -- this is a residual *of* a Pearson residual (or any
    other ``Y_ng``), from a second, unrelated OLS fit against the estimated
    capture rate.

    Parameters
    ----------
    Y_ng : np.ndarray, shape (n, G)
    p_hat : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (n, G)
    """
    x_centered = p_hat - p_hat.mean()
    y_centered = Y_ng - Y_ng.mean(axis=0, keepdims=True)
    denom = np.sum(x_centered**2)
    b = (x_centered[:, None] * y_centered).sum(axis=0) / denom  # (G,)
    return y_centered - b[None, :] * x_centered[:, None]
