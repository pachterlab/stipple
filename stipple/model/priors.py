"""Fixed gene-level hyperpriors and plug-in hyperparameter estimates for
the StippleModel model.

Gene-level hyperpriors on ``mu_gk``/``phi_gk`` are centered on plug-in
method-of-moments estimates from the observed count matrix (see
:func:`estimate_plugins`, :func:`gene_hyperpriors`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

# Fixed hyperparameters (not user-adjustable; see module docstring).
SIGMA_LOG_MU: float = 2.0
SIGMA_LOG_PHI: float = 1.0
PHI_MIN: float = 0.1
PHI_NEAR_POISSON: float = 100.0
MU_FLOOR: float = 1e-3  # guards log(0) for genes with all-zero counts

_DTYPE = torch.float64


@dataclass
class PlugInEstimates:
    """Result of :func:`estimate_plugins`."""

    mu_hat: np.ndarray  # (G,) per-gene observed mean count
    phi_hat: np.ndarray  # (G,) per-gene method-of-moments NegBin dispersion


def _mean_var(X: np.ndarray | sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (gene) mean and unbiased sample variance of ``X`` (n x G).

    Handles dense ``np.ndarray`` and scipy sparse matrices; for sparse
    input the variance is computed as ``E[X^2] - E[X]^2`` (with Bessel's
    correction) without densifying ``X``.
    """
    n = X.shape[0]
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = (X**2).mean(axis=0)
    bessel = n / max(n - 1, 1)
    var = np.maximum((mean_sq - mean**2) * bessel, 0.0)
    return mean, var


def estimate_plugins(X: np.ndarray | sp.spmatrix) -> PlugInEstimates:
    """Method-of-moments NegBin plug-in estimates per gene.

    Parameters
    ----------
    X
        Count matrix of shape ``(n_locations, n_genes)`` — same
        orientation as ``adata.X``. Dense or scipy sparse.

    Returns
    -------
    PlugInEstimates
        ``mu_hat_g`` = observed mean count for gene ``g`` (floored at
        :data:`MU_FLOOR`). ``phi_hat_g`` = ``mu_hat_g**2 / (var_hat_g -
        mu_hat_g)``, clipped to a minimum of :data:`PHI_MIN`; if
        ``var_hat_g <= mu_hat_g`` (underdispersed), ``phi_hat_g`` is set to
        :data:`PHI_NEAR_POISSON` (near-Poisson behavior).
    """
    mu_raw, var_raw = _mean_var(X)

    zero_genes = mu_raw <= 0
    if np.any(zero_genes):
        warnings.warn(
            f"{int(zero_genes.sum())} gene(s) have all-zero counts across "
            f"all locations; flooring mu_hat to {MU_FLOOR}",
            stacklevel=2,
        )
    mu_hat = np.maximum(mu_raw, MU_FLOOR)

    underdispersed = var_raw <= mu_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_mom = mu_hat**2 / (var_raw - mu_hat)
    phi_hat = np.where(underdispersed, PHI_NEAR_POISSON, np.maximum(phi_mom, PHI_MIN))

    return PlugInEstimates(mu_hat=mu_hat, phi_hat=phi_hat)


@dataclass
class GeneHyperpriors:
    """Result of :func:`gene_hyperpriors`."""

    mu_prior: torch.distributions.LogNormal  # loc shape (G,1) = log(mu_hat), scale=SIGMA_LOG_MU
    phi_prior: torch.distributions.LogNormal  # loc shape (G,1) = log(phi_hat), scale=SIGMA_LOG_PHI
    mu_hat: torch.Tensor  # (G,)
    phi_hat: torch.Tensor  # (G,)


def gene_hyperpriors(plugins: PlugInEstimates) -> GeneHyperpriors:
    """Build the LogNormal hyperpriors on ``mu_gk`` and ``phi_gk``.

    The hyperprior on ``mu_gk`` is LogNormal centered on ``mu_hat_g`` with
    ``sigma=2`` (uninformative, non-negative support); the hyperprior on
    ``phi_gk`` is LogNormal centered on ``phi_hat_g`` with ``sigma=1``.
    ``loc`` tensors are shaped ``(G, 1)`` so ``.log_prob(value)`` broadcasts
    against a ``(G, K)`` parameter tensor.

    Parameters
    ----------
    plugins
        Plug-in estimates from :func:`estimate_plugins`.
    """
    mu_hat = torch.as_tensor(plugins.mu_hat, dtype=_DTYPE)
    phi_hat = torch.as_tensor(plugins.phi_hat, dtype=_DTYPE)
    mu_prior = torch.distributions.LogNormal(loc=mu_hat.log().unsqueeze(1), scale=SIGMA_LOG_MU)
    phi_prior = torch.distributions.LogNormal(loc=phi_hat.log().unsqueeze(1), scale=SIGMA_LOG_PHI)
    return GeneHyperpriors(mu_prior=mu_prior, phi_prior=phi_prior, mu_hat=mu_hat, phi_hat=phi_hat)
