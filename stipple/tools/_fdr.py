"""Private FDR-correction helpers for :func:`stipple.tools.group_de_wald_test`.
Not part of the public API.
"""
from __future__ import annotations

import numpy as np

_FDR_METHODS = ("bh", "qvalue")
_FDR_SCOPES = ("global", "per_contrast")


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (assumes ``pi0 = 1``).
    ``nan`` entries are ignored (not counted toward the number of tests,
    returned as ``nan``).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    shape = pvalues.shape
    flat = pvalues.ravel()
    valid = ~np.isnan(flat)
    qvalues = np.full_like(flat, np.nan)

    m = int(valid.sum())
    if m == 0:
        return qvalues.reshape(shape)

    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(flat[valid_idx])]
    ranks = np.arange(1, m + 1)
    q = flat[order] * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity from largest p-value down
    qvalues[order] = np.clip(q, 0.0, 1.0)

    return qvalues.reshape(shape)


def _pi0_storey(pvalues: np.ndarray, lam: float = 0.5) -> float:
    """Storey's (2002) fixed-lambda estimate of pi0, the proportion of
    true null hypotheses: ``mean(p > lam) / (1 - lam)``, clipped to
    ``[0, 1]``.

    Parameters
    ----------
    pvalues
    lam
        Fixed lambda threshold.
    """
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]
    if p.size == 0:
        return 1.0
    pi0 = np.mean(p > lam) / (1.0 - lam)
    return float(np.clip(pi0, 0.0, 1.0))


def _qvalue(pvalues: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Storey's q-value: the BH q-value scaled by the estimated ``pi0``
    (see :func:`_pi0_storey`).
    """
    pi0 = _pi0_storey(pvalues, lam=lam)
    return pi0 * _benjamini_hochberg(pvalues)


def _apply_fdr(pvalue_gkk: np.ndarray, method: str, scope: str = "global") -> np.ndarray:
    """Apply an FDR correction to every pairwise group comparison. Each
    unordered pair (``k < kp``) is corrected once and mirrored onto both
    ``[:, k, kp]`` and ``[:, kp, k]``.

    Parameters
    ----------
    pvalue_gkk
        ``(G, K, K)``, ``nan`` on the ``k == kp`` diagonal.
    method
        ``"bh"`` or ``"qvalue"``.
    scope
        ``"global"``: pool every pairwise group comparison into one
        correction family. ``"per_contrast"``: correct each ``(k, kp)``
        pair as its own family of ``G`` tests.

    Returns
    -------
    np.ndarray, shape (G, K, K), nan on the diagonal.
    """
    if method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {(None,) + _FDR_METHODS}, got {method!r}")
    if scope not in _FDR_SCOPES:
        raise ValueError(f"fdr_scope must be one of {_FDR_SCOPES}, got {scope!r}")
    correct = _benjamini_hochberg if method == "bh" else _qvalue

    G, K, _ = pvalue_gkk.shape
    qvalue_gkk = np.full((G, K, K), np.nan)
    k_idx, kp_idx = np.triu_indices(K, k=1)  # each unordered pair once
    if k_idx.size == 0:
        return qvalue_gkk

    pooled = pvalue_gkk[:, k_idx, kp_idx]  # (G, n_pairs)
    if scope == "global":
        corrected = correct(pooled)
    else:
        corrected = np.column_stack([correct(pooled[:, j]) for j in range(pooled.shape[1])])
    qvalue_gkk[:, k_idx, kp_idx] = corrected
    qvalue_gkk[:, kp_idx, k_idx] = corrected
    return qvalue_gkk
