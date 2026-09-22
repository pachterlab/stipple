r"""Two-stage spatial/differential gene classification: naive vs. fitted
Pearson residuals, Moran's I before/after regressing out the estimated
capture-rate field, combined with a :func:`group_de_wald_test` result.

Ports ``stipple/experimental/classification/plot-classification.R`` (an
unshipped analysis script) into the shipped package, preserving its
``case_when`` classification logic -- including its literal quirks (a
rounding asymmetry in the borderline check, `phi`-based DE never feeding
the final label, and some branches that are unreachable given branch
order) -- verbatim rather than "fixing" them, since the goal is
reproducing/explaining the R script's output, not correcting it.

Two deliberate deviations from the R script: (1) no hex-grid aspect
correction is applied to the coordinates before building the k-NN graph
(``model.data.coords_raw`` is used as-is -- the neighbor *set* is
effectively unchanged without it, and skipping it keeps this general for
non-Visium/square-grid data); (2) Moran's I is computed via
:mod:`squidpy` (``spatial_neighbors``/``spatial_autocorr``) rather than R's
``spdep``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..model import StippleModel

_MORAN_SVG_THRESHOLD: float = 0.2
_MORAN_HIGH_THRESHOLD: float = 0.4
_ATTENUATION_THRESHOLD: float = 0.1
_BORDERLINE_LOWER: float = 0.18  # inclusive, applied to the *rounded* (2dp) Moran's I
_MU_QVALUE_THRESHOLD: float = 0.05
_PHI_QVALUE_THRESHOLD: float = 0.1


def _group_zscore_residuals(Y_ng: np.ndarray, group_idx: np.ndarray, K: int) -> np.ndarray:
    """"Naive"/observed Pearson residual: z-score each gene's raw counts
    *within its own group*, using the group's own empirical mean/sd of raw
    counts (``ddof=1``) -- not the model's fitted mean/variance.

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
    """Residuals of regressing every gene's column in ``Y_ng`` on the
    shared covariate ``p_hat`` (``lm(y ~ p_hat)``'s ``resid()``), vectorized
    across genes via one shared closed-form simple regression.

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


def _compute_moran_i_via_squidpy(
    coords: np.ndarray, gene_names: list[str], n_neighbors: int, arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Moran's I of every ``(n, G)`` array in ``arrays``, on one shared
    k-NN spatial graph, via :func:`squidpy.gr.spatial_neighbors`/
    :func:`squidpy.gr.spatial_autocorr`.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
    gene_names : list[str], length G
    n_neighbors : int
        ``n_neighs`` passed to ``spatial_neighbors``.
    arrays : dict[str, np.ndarray]
        Each value shape ``(n, G)``; keys become column-name prefixes.

    Returns
    -------
    dict[str, np.ndarray]
        Same keys as ``arrays``, each a length-``G`` array of Moran's I.
    """
    import anndata as ad
    import squidpy as sq

    n = coords.shape[0]
    G = len(gene_names)
    keys = list(arrays.keys())
    var_names = [f"{key}::{gene}" for key in keys for gene in gene_names]
    X = np.concatenate([arrays[key] for key in keys], axis=1)  # (n, len(keys)*G)

    scratch = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"loc_{i}" for i in range(n)]),
        var=pd.DataFrame(index=var_names),
    )
    scratch.obsm["spatial"] = coords

    sq.gr.spatial_neighbors_knn(scratch, n_neighs=n_neighbors)
    result = sq.gr.spatial_autocorr(
        scratch, mode="moran", genes=var_names, corr_method=None, n_perms=None,
        show_progress_bar=False, copy=True,
    )

    moran_i: dict[str, np.ndarray] = {}
    for key in keys:
        gene_keys = [f"{key}::{gene}" for gene in gene_names]
        moran_i[key] = result.loc[gene_keys, "I"].to_numpy()
    return moran_i


def _compute_is_flagged(classification_obs: np.ndarray, classification_fitted: np.ndarray) -> np.ndarray:
    """``classification_obs``/``classification_fitted`` disagree, and at
    least one of them is ``"Field-independent SVG"``."""
    obs_indep = classification_obs == "Field-independent SVG"
    fitted_indep = classification_fitted == "Field-independent SVG"
    return (classification_obs != classification_fitted) & (obs_indep | fitted_indep)


def _classify_all(
    classification_obs: np.ndarray,
    classification_de_wald_mu: np.ndarray,
    is_flagged: np.ndarray,
    is_borderline: np.ndarray,
    is_attenuated: np.ndarray,
) -> pd.array:
    """The final ``classification_all`` cascade, ported verbatim (including
    its unreachable-given-branch-3 branches, preserved rather than pruned --
    see the module docstring) from ``plot-classification.R`` lines 157-190.
    First-match-wins, same as R's ``case_when``.
    """
    obs_indep = classification_obs == "Field-independent SVG"
    obs_not = np.char.find(classification_obs.astype(str), "Not") >= 0
    obs_align = np.char.find(classification_obs.astype(str), "align") >= 0
    mu_not = classification_de_wald_mu == "Not DE mu"

    condlist = [
        is_flagged & is_borderline & mu_not,
        is_flagged & is_borderline & ~mu_not,
        obs_indep,
        obs_not & mu_not,
        ~mu_not & obs_not,
        ~mu_not & obs_align,
        mu_not & obs_align,
        ~mu_not & obs_indep & is_attenuated,
        ~is_attenuated & obs_indep & ~mu_not,
    ]
    choicelist = [
        "Field-independent SVG",
        "Field-independent SVG",
        "Field-independent SVG",
        "Null Gene",
        "DVG",
        "DVG + Field-aligned SVG",
        "Field-aligned SVG",
        "DVG",
        "Field-independent SVG",
    ]
    classification_all = np.select(condlist, choicelist, default=None)
    return pd.array(classification_all, dtype="string")


def classify_spatial_variable_genes(
    model: "StippleModel",
    wald_result: pd.DataFrame,
    n_neighbors: int = 6,
    moran_svg_threshold: float = _MORAN_SVG_THRESHOLD,
    moran_high_threshold: float = _MORAN_HIGH_THRESHOLD,
    attenuation_threshold: float = _ATTENUATION_THRESHOLD,
    mu_qvalue_threshold: float = _MU_QVALUE_THRESHOLD,
    phi_qvalue_threshold: float = _PHI_QVALUE_THRESHOLD,
) -> pd.DataFrame:
    """Classify every gene as ``"Null Gene"``, ``"DVG"``, ``"DVG + Field
    -aligned SVG"``, ``"Field-aligned SVG"``, or ``"Field-independent
    SVG"`` (or ``NaN`` if none of the classification rules match), from a
    fitted model and a precomputed :func:`group_de_wald_test` result.

    For each gene: the "naive"/observed Pearson residual (raw counts
    z-scored within each group; see :func:`_group_zscore_residuals`) and
    the fitted Pearson residual (:meth:`StippleModel.get_residuals`,
    ``type="pearson"``) are each regressed on the estimated capture rate
    (:meth:`StippleModel.get_capture_rates`) -- "stage 1" is Moran's I of
    the raw residual, "stage 2" is Moran's I of that regression's residual
    (spatial signal remaining after removing the capture-rate field's
    linear effect). A gene is flagged spatially variable at ``moran_*
    >= moran_svg_threshold`` and "attenuated" if stage 1 and stage 2
    Moran's I differ by at least ``attenuation_threshold``. These, plus
    whether every pairwise ``mu_qvalue``/``phi_qvalue`` in ``wald_result``
    exceeds ``mu_qvalue_threshold``/``phi_qvalue_threshold`` for that gene,
    feed a classification cascade ported verbatim from
    ``stipple/experimental/classification/plot-classification.R``
    (preserving its quirks -- see the module docstring).

    Parameters
    ----------
    model
        A fitted :class:`stipple.model.StippleModel`.
    wald_result
        The tidy DataFrame returned by :func:`stipple.tools.group_de_wald_test`
        (called separately, beforehand -- this function does not recompute
        it), with ``fdr_method`` set so ``mu_qvalue`` (and, for NegBin,
        ``phi_qvalue``) columns are present.
    n_neighbors
        Number of nearest neighbors in the shared spatial graph used for
        every Moran's I computation (``spatial_neighbors(..., n_neighs=
        n_neighbors)``).
    moran_svg_threshold, moran_high_threshold, attenuation_threshold,
    mu_qvalue_threshold, phi_qvalue_threshold
        Thresholds as in the R source (defaults: ``0.2``, ``0.4``, ``0.1``,
        ``0.05``, ``0.1`` respectively).

    Returns
    -------
    pandas.DataFrame
        One row per gene: raw Moran's I (``moran_resid_stage1/2``,
        ``moran_fit_resid_stage1/2``), every boolean flag (``is_svg_stage1/2``,
        ``is_fit_svg_stage1/2``, ``is_attenuated``, ``is_high_svg_stage1``,
        ``is_borderline``, ``is_flagged``), ``classification_obs``,
        ``classification_fitted``, ``classification_de_wald_mu/phi``, and
        ``classification_all``.
    """
    model._check_fitted()
    if "mu_qvalue" not in wald_result.columns:
        raise ValueError(
            "wald_result has no 'mu_qvalue' column -- call group_de_wald_test(model, "
            "fdr_method=...) with fdr_method set before classify_spatial_variable_genes"
        )
    has_phi = "phi_qvalue" in wald_result.columns

    gene_names = model.data.gene_names
    group_idx = model.data.group_idx.numpy()
    K = model.data.K
    coords = model.data.coords_raw
    Y_ng = model.model.Y_ng.numpy()
    p_hat = model.get_capture_rates()

    resid_obs = _group_zscore_residuals(Y_ng, group_idx, K)
    resid_fitted = model.get_residuals(type="pearson").T  # (G, n) -> (n, G)
    resid_obs_stage2 = _regress_out_capture_rate(resid_obs, p_hat)
    resid_fitted_stage2 = _regress_out_capture_rate(resid_fitted, p_hat)

    moran = _compute_moran_i_via_squidpy(
        coords, gene_names, n_neighbors,
        {
            "resid_stage1": resid_obs,
            "resid_stage2": resid_obs_stage2,
            "fit_resid_stage1": resid_fitted,
            "fit_resid_stage2": resid_fitted_stage2,
        },
    )
    moran_resid_stage1 = moran["resid_stage1"]
    moran_resid_stage2 = moran["resid_stage2"]
    moran_fit_resid_stage1 = moran["fit_resid_stage1"]
    moran_fit_resid_stage2 = moran["fit_resid_stage2"]

    is_svg_stage1 = moran_resid_stage1 >= moran_svg_threshold
    is_svg_stage2 = moran_resid_stage2 >= moran_svg_threshold
    is_fit_svg_stage1 = moran_fit_resid_stage1 >= moran_svg_threshold
    is_fit_svg_stage2 = moran_fit_resid_stage2 >= moran_svg_threshold
    is_attenuated = np.abs(moran_resid_stage2 - moran_resid_stage1) >= attenuation_threshold
    is_high_svg_stage1 = moran_resid_stage1 >= moran_high_threshold

    classification_obs = np.select(
        [is_svg_stage1 & is_svg_stage2, is_svg_stage1 & ~is_svg_stage2],
        ["Field-independent SVG", "Field-aligned SVG"],
        default="Not Classified",
    )
    classification_fitted = np.select(
        [is_fit_svg_stage1 & is_fit_svg_stage2, is_fit_svg_stage1 & ~is_fit_svg_stage2],
        ["Field-independent SVG", "Field-aligned SVG"],
        default="Not Classified",
    )

    not_de_mu = wald_result.groupby("gene")["mu_qvalue"].apply(lambda s: (s > mu_qvalue_threshold).all())
    not_de_mu_genes = set(not_de_mu.index[not_de_mu])
    classification_de_wald_mu = np.array(
        ["Not DE mu" if g in not_de_mu_genes else "DE mu" for g in gene_names],
    )
    if has_phi:
        not_de_phi = wald_result.groupby("gene")["phi_qvalue"].apply(lambda s: (s > phi_qvalue_threshold).all())
        not_de_phi_genes = set(not_de_phi.index[not_de_phi])
        classification_de_wald_phi = np.array(
            ["Not DE phi" if g in not_de_phi_genes else "DE phi" for g in gene_names],
        )
    else:
        classification_de_wald_phi = np.full(len(gene_names), None, dtype=object)

    stage_morans = np.stack(
        [moran_resid_stage1, moran_resid_stage2, moran_fit_resid_stage1, moran_fit_resid_stage2], axis=1,
    )
    # Literal R quirk: the lower bound is checked against the *rounded* (2dp)
    # value, but the upper bound against the unrounded value.
    borderline_per_col = (np.round(stage_morans, 2) >= _BORDERLINE_LOWER) & (stage_morans < moran_svg_threshold)
    is_borderline = borderline_per_col.any(axis=1)

    is_flagged = _compute_is_flagged(classification_obs, classification_fitted)
    classification_all = _classify_all(
        classification_obs, classification_de_wald_mu, is_flagged, is_borderline, is_attenuated,
    )

    return pd.DataFrame(
        {
            "gene": gene_names,
            "moran_resid_stage1": moran_resid_stage1,
            "moran_resid_stage2": moran_resid_stage2,
            "moran_fit_resid_stage1": moran_fit_resid_stage1,
            "moran_fit_resid_stage2": moran_fit_resid_stage2,
            "is_svg_stage1": is_svg_stage1,
            "is_svg_stage2": is_svg_stage2,
            "is_fit_svg_stage1": is_fit_svg_stage1,
            "is_fit_svg_stage2": is_fit_svg_stage2,
            "is_attenuated": is_attenuated,
            "is_high_svg_stage1": is_high_svg_stage1,
            "is_borderline": is_borderline,
            "is_flagged": is_flagged,
            "classification_obs": classification_obs,
            "classification_fitted": classification_fitted,
            "classification_de_wald_mu": classification_de_wald_mu,
            "classification_de_wald_phi": classification_de_wald_phi,
            "classification_all": classification_all,
        }
    )
