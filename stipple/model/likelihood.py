"""Negative-binomial and Poisson log-likelihood functions for the
StippleModel basis model. ``Y_ng`` is always shape ``(n, G)`` (locations x
genes).
"""
from __future__ import annotations

import torch

MU_EPS: float = 1e-8  # numerical floor before log() in the NB/Poisson rate computation


def _mu_expected(p: torch.Tensor, mu_gk: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    """``(mu_gk[:, group_idx].T * p[:, None]).clamp_min(MU_EPS)``, shape (n, G)."""
    mu_ng = mu_gk.t()[group_idx]  # (n, G) -- gather by group
    return (mu_ng * p.unsqueeze(-1)).clamp_min(MU_EPS)


def negbin_loglik(
    p: torch.Tensor,
    mu_gk: torch.Tensor,
    phi_gk: torch.Tensor,
    group_idx: torch.Tensor,
    Y_ng: torch.Tensor,
) -> torch.Tensor:
    """Total negative-binomial log-likelihood, summed over every
    (location, gene) pair.

    Parameters
    ----------
    p : torch.Tensor, shape (n,)
        Capture-rate field.
    mu_gk, phi_gk : torch.Tensor, shape (G, K)
        Per-gene-per-group mean and dispersion.
    group_idx : torch.Tensor, shape (n,)
        Integer group index in ``[0, K)`` for each location.
    Y_ng : torch.Tensor, shape (n, G)
        Observed counts.
    """
    mu_expected = _mu_expected(p, mu_gk, group_idx)  # (n, G)
    phi_ng = phi_gk.t()[group_idx]  # (n, G)
    logits = mu_expected.log() - phi_ng.log()  # never needs log(1-p): NB is total_count/logits-parameterized
    nb = torch.distributions.NegativeBinomial(total_count=phi_ng, logits=logits)
    return nb.log_prob(Y_ng).sum()


def poisson_loglik(
    p: torch.Tensor,
    mu_gk: torch.Tensor,
    group_idx: torch.Tensor,
    Y_ng: torch.Tensor,
) -> torch.Tensor:
    """Total Poisson log-likelihood, summed over every (location, gene) pair.

    Parameters
    ----------
    p : torch.Tensor, shape (n,)
    mu_gk : torch.Tensor, shape (G, K)
    group_idx : torch.Tensor, shape (n,)
    Y_ng : torch.Tensor, shape (n, G)
    """
    mu_expected = _mu_expected(p, mu_gk, group_idx)  # (n, G)
    poisson = torch.distributions.Poisson(rate=mu_expected)
    return poisson.log_prob(Y_ng).sum()
