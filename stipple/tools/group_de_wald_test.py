r"""Fisher-information (Wald) test for pairwise between-group differential
expression of ``mu_gk`` (and, for the negative-binomial likelihood,
``phi_gk``).

Approximates each gene's ``(log_mu_gk, log_phi_gk)`` posterior as locally
log-normal at the MAP via the observed Fisher information, and returns
pairwise TREAT-style standard errors, confidence intervals, and
minimum-effect-size tests.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from ..model import likelihood as _likelihood

if TYPE_CHECKING:
    from ..model import StippleModel

LOG2E: float = 1.0 / np.log(2.0)  # natural-log -> log2 scale factor (SEs/log-params scale linearly)
_FDR_METHODS = ("bh", "qvalue")
_FDR_SCOPES = ("global", "per_contrast")


def _pairwise_wald_stats(
    log_param: np.ndarray, se_log: np.ndarray, extra_diff_var: np.ndarray | None = None, delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise log-fold-change/Wald-statistic/p-value arrays from a
    per-gene-per-group log-scale point estimate and standard error.
    TREAT-style minimum-effect-size test of H0: ``|log_param[k] -
    log_param[kp]| <= delta`` vs H1: ``... > delta``.

    Parameters
    ----------
    log_param : np.ndarray, shape (G, K)
        Per-gene-per-group log-scale point estimate.
    se_log : np.ndarray, shape (G, K)
        Standard error of ``log_param``.
    extra_diff_var : np.ndarray, shape (K, K), optional
        Added directly to ``Var(log_param[:,k] - log_param[:,kp])`` for
        every gene.
    delta
        Log-fold-change threshold (same log base as ``log_param``).

    Returns
    -------
    log_fc, wald, pval : np.ndarray, each shape (G, K, K)
        ``nan`` on the ``k == kp`` diagonal.
    """
    G, K = log_param.shape
    log_fc = np.full((G, K, K), np.nan)
    wald = np.full((G, K, K), np.nan)
    pval = np.full((G, K, K), np.nan)
    for k in range(K):
        for kp in range(K):
            if k == kp:
                continue
            lfc = log_param[:, k] - log_param[:, kp]
            diff_var = se_log[:, k] ** 2 + se_log[:, kp] ** 2
            if extra_diff_var is not None:
                diff_var = diff_var + extra_diff_var[k, kp]
            se_diff = np.sqrt(diff_var)

            with np.errstate(divide="ignore", invalid="ignore"):
                z_a = (lfc - delta) / se_diff  # H1: lfc > delta
                z_b = (lfc + delta) / se_diff  # H1: lfc < -delta
            p_a = 1.0 - norm.cdf(z_a)
            p_b = norm.cdf(z_b)
            p = np.minimum(1.0, 2.0 * np.minimum(p_a, p_b))
            w = np.where(lfc >= 0, z_a, z_b)

            boundary = np.isnan(w) | np.isnan(p)  # se_diff==0 and |lfc|==delta exactly (0/0)
            w = np.where(boundary, 0.0, w)
            p = np.where(boundary, 1.0, p)

            log_fc[:, k, kp] = lfc
            wald[:, k, kp] = w
            pval[:, k, kp] = p
    return log_fc, wald, pval


def _tidy_group_expression_frame(
    gene_names: list[str],
    group_labels: list[str],
    delta: float,
    mu_se_log2: np.ndarray,
    mu_se: np.ndarray,
    mu_ci_lower: np.ndarray | None,
    mu_ci_upper: np.ndarray | None,
    mu_log2_fc: np.ndarray,
    mu_wald: np.ndarray,
    mu_pvalue: np.ndarray,
    mu_qvalue: np.ndarray | None,
    fdr_method: str | None,
    phi_se_log2: np.ndarray | None = None,
    phi_se: np.ndarray | None = None,
    phi_log2_fc: np.ndarray | None = None,
    phi_wald: np.ndarray | None = None,
    phi_pvalue: np.ndarray | None = None,
    phi_qvalue: np.ndarray | None = None,
    cov_log_mu_log_phi: np.ndarray | None = None,
) -> pd.DataFrame:
    """Tidy, wide-format result: one row per (gene, group, reference_group)
    pairwise comparison (``k != kp``, every ordered pair).

    Parameters
    ----------
    gene_names, group_labels
        Length ``G``/``K`` labels for the output columns.
    delta
        Log-fold-change threshold used, recorded on every row.
    mu_se_log2, mu_se, mu_ci_lower, mu_ci_upper : np.ndarray, shape (G, K)
        Per-gene-per-group standard error/CI, repeated across every row
        sharing that ``(gene, group)``.
    mu_log2_fc, mu_wald, mu_pvalue : np.ndarray, shape (G, K, K)
        Pairwise fold-change/statistic/p-value.
    mu_qvalue : np.ndarray, shape (G, K, K), optional
        FDR-corrected p-values.
    fdr_method
        Name recorded on every row, or ``None`` to omit the column.
    phi_* : optional
        Same shapes as their ``mu_*`` counterparts; included only for the
        negative-binomial likelihood.
    cov_log_mu_log_phi : np.ndarray, shape (G, K), optional
    """
    G, K, _ = mu_log2_fc.shape
    k_idx, kp_idx = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    off_diag = k_idx != kp_idx
    k_flat = k_idx[off_diag]  # (K*(K-1),)
    kp_flat = kp_idx[off_diag]
    n_pairs = k_flat.size
    n_rows = G * n_pairs

    group_labels_arr = np.asarray(group_labels, dtype=object)

    def _pair(arr: np.ndarray) -> np.ndarray:
        return arr[:, k_flat, kp_flat].ravel()  # (G, n_pairs) -> (n_rows,), gene-major order

    def _single(arr: np.ndarray) -> np.ndarray:
        return arr[:, k_flat].ravel()  # (G, K) -> (n_rows,), repeated per reference_group

    data = {
        "gene": np.repeat(np.asarray(gene_names, dtype=object), n_pairs),
        "group": np.tile(group_labels_arr[k_flat], G),
        "reference_group": np.tile(group_labels_arr[kp_flat], G),
        "delta": np.full(n_rows, delta),
        "mu_se_log2": _single(mu_se_log2),
        "mu_se": _single(mu_se),
        "mu_log2_fold_change": _pair(mu_log2_fc),
        "mu_wald_stat": _pair(mu_wald),
        "mu_pvalue": _pair(mu_pvalue),
    }
    if mu_ci_lower is not None:
        data["mu_ci_lower"] = _single(mu_ci_lower)
        data["mu_ci_upper"] = _single(mu_ci_upper)
    if mu_qvalue is not None:
        data["mu_qvalue"] = _pair(mu_qvalue)

    if phi_log2_fc is not None:
        data["phi_se_log2"] = _single(phi_se_log2)
        data["phi_se"] = _single(phi_se)
        data["phi_log2_fold_change"] = _pair(phi_log2_fc)
        data["phi_wald_stat"] = _pair(phi_wald)
        data["phi_pvalue"] = _pair(phi_pvalue)
        if phi_qvalue is not None:
            data["phi_qvalue"] = _pair(phi_qvalue)
        if cov_log_mu_log_phi is not None:
            data["cov_log_mu_log_phi"] = _single(cov_log_mu_log_phi)

    if fdr_method is not None:
        data["fdr_method"] = np.full(n_rows, fdr_method, dtype=object)

    return pd.DataFrame(data)


def _capture_rate_variance_by_group(
    model: "StippleModel", svd_rank: int | None = None, variance_threshold: float = 0.99,
) -> dict[str, Any]:
    """Delta-method posterior covariance of each group's mean ``log p_i``
    (capture rate), via a truncated-SVD Laplace approximation of ``w``'s
    own posterior restricted to the subspace spanned by ``B``'s top
    singular directions. Feeds :func:`group_de_wald_test`'s
    ``propagate_capture_rate_uncertainty`` option.

    Parameters
    ----------
    svd_rank
        Number of singular directions to compute (``None`` defaults to
        ``min(M, 200)``). If the retained ``variance_threshold`` isn't
        reached within this many directions, a warning suggests raising
        it.
    variance_threshold
        Fraction of ``B``'s singular-value-squared mass the retained
        subspace must cover (default ``0.99``).

    Returns
    -------
    dict
        ``group_pair_covariance`` (``(K, K)``): ``Cov(mean_i in k log
        p_i, mean_i in kp log p_i)`` -- the diagonal is each group's own
        variance. ``retained_variance_fraction`` (float): the actual
        singular-value-squared mass covered by the retained subspace.
        ``m_star`` (int): number of retained directions.
    """
    model._check_fitted()
    inner = model.model
    B = inner.B  # (n, M)
    M = B.shape[1]
    if svd_rank is None:
        svd_rank = min(M, 200)
    svd_rank = min(svd_rank, M)

    if M <= 1000:
        _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        V = Vh.t()[:, :svd_rank]
        S = S[:svd_rank]
    else:
        torch.manual_seed(0)
        _, S, V = torch.svd_lowrank(B, q=svd_rank, niter=4)
    energy = S**2
    cum_frac = torch.cumsum(energy, dim=0) / energy.sum()
    m_star = int(torch.searchsorted(cum_frac, variance_threshold).item()) + 1
    m_star = min(m_star, svd_rank)
    retained_variance_fraction = float(cum_frac[m_star - 1].item())
    if m_star == svd_rank and retained_variance_fraction < variance_threshold:
        model.message(
            f"truncated SVD of B retained only {retained_variance_fraction:.1%} of its "
            f"singular-value mass at svd_rank={svd_rank} (target {variance_threshold:.1%}); "
            "raise svd_rank for a tighter capture-rate-uncertainty approximation"
        )

    w_hat = inner.w.detach()
    K = inner.K
    mu_gk = inner.log_mu_gk.detach().exp()
    phi_gk = inner.log_phi_gk.detach().exp() if inner.HAS_DISPERSION else None
    Y_ng = inner.Y_ng
    group_idx = inner.group_idx
    sigma2 = model.sigma2
    group_locations = [torch.nonzero(group_idx == k, as_tuple=True)[0] for k in range(K)]

    def _p_of(w_star: torch.Tensor, V_sub: torch.Tensor) -> torch.Tensor:
        w = w_hat + V_sub @ w_star
        u = B @ w
        group_mean_u = inner._group_sum(u) / inner._group_count
        u_tilde = u - inner._group_mean_at_locations(group_mean_u)
        q = torch.sigmoid(u_tilde)
        return inner.p0 * q / q.mean()

    def log_posterior(w_star: torch.Tensor, V_sub: torch.Tensor) -> torch.Tensor:
        p = _p_of(w_star, V_sub)
        if inner.HAS_DISPERSION:
            ll = _likelihood.negbin_loglik(p, mu_gk, phi_gk, group_idx, Y_ng)
        else:
            ll = _likelihood.poisson_loglik(p, mu_gk, group_idx, Y_ng)
        w_log_prior = torch.distributions.Normal(0.0, sigma2**0.5).log_prob(w_hat + V_sub @ w_star).sum()
        return ll + w_log_prior

    V_target = V[:, :m_star]
    w_star0 = torch.zeros(m_star, dtype=w_hat.dtype)
    hess = torch.autograd.functional.hessian(lambda ws: log_posterior(ws, V_target), w_star0)
    fisher_full = -(hess + hess.t()) / 2.0  # negate, then symmetrize away autograd fp asymmetry

    m_try = m_star
    L = None
    while m_try >= 1:
        sub = fisher_full[:m_try, :m_try]
        try:
            L = torch.linalg.cholesky(sub)
            break
        except torch._C._LinAlgError:
            m_try = m_try // 2
    if L is None:
        raise RuntimeError(
            "capture-rate uncertainty propagation: even a single-direction Fisher "
            "block was not positive-definite -- the MAP fit may not be well converged"
        )
    if m_try < m_star:
        model.message(
            f"capture-rate Fisher block was not positive-definite at m_star={m_star}; "
            f"shrunk to the top {m_try} singular directions (still highest-identified) instead"
        )
    m_star = m_try
    V_star = V[:, :m_star]

    v_bar = torch.empty((K, m_star), dtype=w_hat.dtype)
    for k in range(K):
        w_star_grad = torch.zeros(m_star, dtype=w_hat.dtype, requires_grad=True)
        mean_log_p_k = _p_of(w_star_grad, V_star)[group_locations[k]].log().mean()
        (grad_k,) = torch.autograd.grad(mean_log_p_k, w_star_grad)
        v_bar[k] = grad_k.detach()

    sol = torch.cholesky_solve(v_bar.t(), L)  # (m_star, K) = Sigma_w* @ v_bar^T
    group_pair_covariance = (v_bar @ sol).numpy()  # (K, K)

    return {
        "group_pair_covariance": group_pair_covariance,
        "retained_variance_fraction": retained_variance_fraction,
        "m_star": m_star,
    }


def group_de_wald_test(
    model: "StippleModel",
    delta: float = 1.0,
    propagate_capture_rate_uncertainty: bool = False,
    fdr_method: str | None = None,
    fdr_scope: str = "global",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Fisher-information (Wald) pairwise group DE test.

    Parameters
    ----------
    model
        A fitted :class:`stipple.model.StippleModel`.
    delta
        Log2 fold-change threshold for the pairwise test, applied to both
        ``mu`` and ``phi``: H0: ``|log2fc| <= delta`` vs H1: ``... >
        delta``. ``delta=1.0`` (default) is a literal 2-fold change.
        ``delta=0.0`` gives the null-at-zero two-sided Wald test.
    propagate_capture_rate_uncertainty
        ``False`` (default): every SE conditions on the capture rate ``p``
        as fixed. ``True`` adds ``w``'s own estimation uncertainty into
        ``mu_gk``'s SE and the pairwise between-group covariance, via
        :func:`_capture_rate_variance_by_group`.
    fdr_method
        ``None`` (default): ``mu_pvalue``/``phi_pvalue`` are uncorrected.
        ``"bh"``/``"qvalue"``: add ``mu_qvalue``/``phi_qvalue`` columns.
    fdr_scope
        ``"global"`` (default): pool every ``(gene, group pair)``
        combination into one correction family. ``"per_contrast"``:
        correct each group pair's ``G`` p-values as its own family.
        Ignored when ``fdr_method`` is ``None``.

    Returns
    -------
    tuple[pandas.DataFrame, dict | None]
        ``(result, capture_rate_diagnostics)``. ``result``: one row per
        ``(gene, group, reference_group)`` ordered pair -- ``mu_se_log2``,
        ``mu_se``, ``mu_ci_lower``, ``mu_ci_upper`` (per-``(gene, group)``,
        repeated across ``reference_group`` rows); ``mu_log2_fold_change``,
        ``mu_wald_stat``, ``mu_pvalue`` (per-pair); ``mu_qvalue`` if
        ``fdr_method`` set. NegBin only: ``phi_se_log2``, ``phi_se``,
        ``phi_log2_fold_change``, ``phi_wald_stat``, ``phi_pvalue``
        (``phi_qvalue`` if set), and ``cov_log_mu_log_phi`` (natural-log
        scale). ``capture_rate_diagnostics``: ``None`` unless
        ``propagate_capture_rate_uncertainty=True``, else a dict
        (``group_pair_covariance`` ``(K, K)``, ``retained_variance_fraction``,
        ``m_star``).
    """
    if fdr_method is not None and fdr_method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be None or one of {_FDR_METHODS}, got {fdr_method!r}")
    if fdr_scope not in _FDR_SCOPES:
        raise ValueError(f"fdr_scope must be one of {_FDR_SCOPES}, got {fdr_scope!r}")
    model._check_fitted()
    model.message("computing block Fisher information standard errors for mu_gk/phi_gk")

    G, K = model.data.G, model.data.K
    log_mu_gk = model.model.log_mu_gk.detach()  # (G, K)
    has_dispersion = model.model.HAS_DISPERSION
    log_phi_gk = model.model.log_phi_gk.detach() if has_dispersion else None
    group_idx = model.model.group_idx  # (n,)
    Y_ng = model.model.Y_ng  # (n, G)
    mu_hyperprior = model.model.mu_hyperprior
    phi_hyperprior = model.model.phi_hyperprior if has_dispersion else None
    block_size = 2 * K if has_dispersion else K

    with torch.no_grad():
        p_hat = model.model.p  # (n,) -- deterministic MAP capture rate

    group_locations = [torch.nonzero(group_idx == k, as_tuple=True)[0] for k in range(K)]

    def _scalar_fisher_mu(g: int, k: int) -> float:
        idx_k = group_locations[k]
        y_gk = Y_ng[idx_k, g]
        p_k = p_hat[idx_k]
        mu_hp_gk = torch.distributions.LogNormal(mu_hyperprior.loc[g, 0], mu_hyperprior.scale[g, 0])
        phi_fixed = log_phi_gk[g, k].exp() if has_dispersion else None

        def log_posterior(log_mu: torch.Tensor) -> torch.Tensor:
            mu_i = (log_mu.exp() * p_k).clamp_min(_likelihood.MU_EPS)
            if has_dispersion:
                nb = torch.distributions.NegativeBinomial(
                    total_count=phi_fixed, logits=mu_i.log() - phi_fixed.log()
                )
                ll = nb.log_prob(y_gk).sum()
            else:
                ll = torch.distributions.Poisson(rate=mu_i).log_prob(y_gk).sum()
            return ll + mu_hp_gk.log_prob(log_mu.exp())

        hess = torch.autograd.functional.hessian(log_posterior, log_mu_gk[g, k])
        return -hess.item()

    def _scalar_fisher_phi(g: int, k: int) -> float:
        idx_k = group_locations[k]
        y_gk = Y_ng[idx_k, g]
        p_k = p_hat[idx_k]
        phi_hp_gk = torch.distributions.LogNormal(phi_hyperprior.loc[g, 0], phi_hyperprior.scale[g, 0])
        mu_i = (log_mu_gk[g, k].exp() * p_k).clamp_min(_likelihood.MU_EPS)

        def log_posterior(log_phi: torch.Tensor) -> torch.Tensor:
            phi_i = log_phi.exp()
            nb = torch.distributions.NegativeBinomial(total_count=phi_i, logits=mu_i.log() - phi_i.log())
            ll = nb.log_prob(y_gk).sum()
            return ll + phi_hp_gk.log_prob(log_phi.exp())

        hess = torch.autograd.functional.hessian(log_posterior, log_phi_gk[g, k])
        return -hess.item()

    def _gene_block(g: int) -> dict[str, np.ndarray]:
        y_g = Y_ng[:, g]
        mu_hp = torch.distributions.LogNormal(mu_hyperprior.loc[g, 0], mu_hyperprior.scale[g, 0])
        phi_hp = (
            torch.distributions.LogNormal(phi_hyperprior.loc[g, 0], phi_hyperprior.scale[g, 0])
            if has_dispersion
            else None
        )

        def log_posterior(theta: torch.Tensor) -> torch.Tensor:
            log_mu_k = theta[:K]
            mu_i = (log_mu_k.exp()[group_idx] * p_hat).clamp_min(_likelihood.MU_EPS)
            if has_dispersion:
                log_phi_k = theta[K:]
                phi_i = log_phi_k.exp()[group_idx]
                nb = torch.distributions.NegativeBinomial(
                    total_count=phi_i, logits=mu_i.log() - phi_i.log()
                )
                ll = nb.log_prob(y_g).sum()
                return ll + mu_hp.log_prob(log_mu_k.exp()).sum() + phi_hp.log_prob(log_phi_k.exp()).sum()
            ll = torch.distributions.Poisson(rate=mu_i).log_prob(y_g).sum()
            return ll + mu_hp.log_prob(log_mu_k.exp()).sum()

        theta0 = torch.cat([log_mu_gk[g], log_phi_gk[g]]) if has_dispersion else log_mu_gk[g]
        hess = torch.autograd.functional.hessian(log_posterior, theta0)
        fisher = (-hess).numpy()

        eigvals = np.linalg.eigvalsh(fisher)
        if np.any(eigvals <= 0):
            model.message(f"non-positive-definite Fisher information block for gene {g}; falling back to scalar SEs")
            se_log_mu = np.empty(K)
            se_log_phi = np.empty(K) if has_dispersion else None
            cov_mu_phi = np.full(K, np.nan) if has_dispersion else None
            for k in range(K):
                fi_mu = _scalar_fisher_mu(g, k)
                se_log_mu[k] = fi_mu**-0.5 if fi_mu > 0 else np.inf
                if has_dispersion:
                    fi_phi = _scalar_fisher_phi(g, k)
                    se_log_phi[k] = fi_phi**-0.5 if fi_phi > 0 else np.inf
        else:
            cov = np.linalg.inv(fisher)
            se_log_mu = np.sqrt(np.diag(cov)[:K])
            se_log_phi = np.sqrt(np.diag(cov)[K:block_size]) if has_dispersion else None
            cov_mu_phi = np.array([cov[k, K + k] for k in range(K)]) if has_dispersion else None

        return {"se_log_mu": se_log_mu, "se_log_phi": se_log_phi, "cov_mu_phi": cov_mu_phi}

    torch.set_num_threads(1)
    try:
        blocks = joblib.Parallel(n_jobs=model.num_threads, prefer="threads")(
            joblib.delayed(_gene_block)(g) for g in range(G)
        )
    finally:
        torch.set_num_threads(model.num_threads)

    se_log_mu = np.stack([b["se_log_mu"] for b in blocks], axis=0)  # (G, K)
    log_mu_np = log_mu_gk.numpy()
    mu_gk_hat = np.exp(log_mu_np)

    capture_rate_diagnostics = None
    pair_cov_diff = None
    if propagate_capture_rate_uncertainty:
        capture_rate_diagnostics = _capture_rate_variance_by_group(model)
        group_pair_cov = capture_rate_diagnostics["group_pair_covariance"]  # (K, K)
        group_var = np.diag(group_pair_cov)  # (K,)
        # se_log_mu (updated below) already carries group_var[k] once
        # per group, so _pairwise_wald_stats' se_log[k]**2+se_log[kp]**2
        # already has group_var[k]+group_var[kp] -- only the missing
        # cross term goes into extra_diff_var, not the full covariance.
        se_log_mu = np.sqrt(se_log_mu**2 + group_var[None, :])
        pair_cov_diff = -2.0 * group_pair_cov  # (K, K): cross-covariance correction only

    se_mu = mu_gk_hat * se_log_mu
    ci_lower = np.exp(log_mu_np - 1.96 * se_log_mu)
    ci_upper = np.exp(log_mu_np + 1.96 * se_log_mu)

    se_log2_mu = se_log_mu * LOG2E
    pair_cov_diff_log2 = pair_cov_diff * LOG2E**2 if pair_cov_diff is not None else None
    log_fc_mu, wald_mu, pval_mu = _pairwise_wald_stats(
        log_mu_np * LOG2E, se_log2_mu, extra_diff_var=pair_cov_diff_log2, delta=delta,
    )

    def _fdr(pval: np.ndarray) -> np.ndarray | None:
        if fdr_method is None:
            return None
        from ._fdr import _apply_fdr

        return _apply_fdr(pval, fdr_method, scope=fdr_scope)

    mu_qvalue = _fdr(pval_mu)

    se_log2_phi = se_phi = log_fc_phi = wald_phi = pval_phi = phi_qvalue = cov_mu_phi = None
    if has_dispersion:
        se_log_phi = np.stack([b["se_log_phi"] for b in blocks], axis=0)  # (G, K)
        log_phi_np = log_phi_gk.numpy()
        phi_gk_hat = np.exp(log_phi_np)
        se_phi = phi_gk_hat * se_log_phi
        se_log2_phi = se_log_phi * LOG2E
        log_fc_phi, wald_phi, pval_phi = _pairwise_wald_stats(log_phi_np * LOG2E, se_log2_phi, delta=delta)
        phi_qvalue = _fdr(pval_phi)
        cov_mu_phi = np.stack([b["cov_mu_phi"] for b in blocks], axis=0)  # (G, K)

    model.message("block Fisher information standard errors computed")

    result = _tidy_group_expression_frame(
        model.data.gene_names, model.data.group_labels, delta,
        mu_se_log2=se_log2_mu, mu_se=se_mu, mu_ci_lower=ci_lower, mu_ci_upper=ci_upper,
        mu_log2_fc=log_fc_mu, mu_wald=wald_mu, mu_pvalue=pval_mu, mu_qvalue=mu_qvalue, fdr_method=fdr_method,
        phi_se_log2=se_log2_phi, phi_se=se_phi, phi_log2_fc=log_fc_phi, phi_wald=wald_phi,
        phi_pvalue=pval_phi, phi_qvalue=phi_qvalue, cov_log_mu_log_phi=cov_mu_phi,
    )
    return result, capture_rate_diagnostics
