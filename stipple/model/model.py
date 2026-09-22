"""Basis-function spatial-field model and MAP-fitting machinery for
StippleModel.

See :class:`_StippleBasisModel` (the basis-weighted capture-rate field,
jointly MAP-fit via Adam with gene-level NegBin/Poisson parameters) and
:func:`run_map_optimization` (the shared full-batch Adam loop).
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import torch

from . import likelihood as _likelihood
from . import priors as _priors

_DTYPE = torch.float64


class _StippleBasisModel(torch.nn.Module):
    """Basis-weighted capture-efficiency field, jointly MAP-fit
    (Adam/autograd) with gene-level NegBin/Poisson parameters.

    Parameters
    ----------
    B : torch.Tensor, shape (n, M)
        Normalized (partition-of-unity) basis matrix (see :mod:`stipple.model.basis`).
    Y_gn : torch.Tensor, shape (G, n)
        Observed count matrix.
    group_idx : torch.Tensor, shape (n,)
    G, K, n : int
    mu_hat, phi_hat : torch.Tensor, shape (G,)
        Plug-in estimates (see :func:`stipple.model.priors.estimate_plugins`).
        ``phi_hat`` is always required even when ``likelihood="poisson"``,
        which simply ignores it.
    sigma2 : float
        Fixed prior variance of each ``w_k ~ Normal(0, sigma2)``.
    known_capture_rate : float
    likelihood : str
        ``"negbinomial"`` (default) or ``"poisson"``.
    gene_batch_size : int | None
        If given (and ``< G``), each :meth:`log_joint` call during
        training uses a random subset of this many genes instead of all
        ``G``. ``None`` (default) uses every gene, every call. Only
        affects training -- :meth:`data_loglik` always uses the full
        likelihood.
    gene_batch_seed : int
        Seeds the gene-batch sampling sequence.
    """

    def __init__(
        self,
        B: torch.Tensor,
        Y_gn: torch.Tensor,
        group_idx: torch.Tensor,
        G: int,
        K: int,
        n: int,
        mu_hat: torch.Tensor,
        phi_hat: torch.Tensor,
        sigma2: float,
        known_capture_rate: float,
        likelihood: str = "negbinomial",
        gene_batch_size: int | None = None,
        gene_batch_seed: int = 0,
    ) -> None:
        super().__init__()
        if likelihood not in ("negbinomial", "poisson"):
            raise ValueError(f"likelihood must be 'negbinomial' or 'poisson', got {likelihood!r}")
        if gene_batch_size is not None and not (0 < gene_batch_size <= G):
            raise ValueError(f"gene_batch_size must be in (0, G={G}], got {gene_batch_size}")
        M = B.shape[1]
        self.register_buffer("B", B)  # (n, M)
        self.register_buffer("Y_ng", Y_gn.t().contiguous())  # (n, G)
        self.register_buffer("group_idx", group_idx.long())  # (n,)
        self.G, self.K, self.n, self.M = G, K, n, M

        self.register_buffer("_group_count", self._group_sum(torch.ones(n, dtype=_DTYPE)))  # (K,) locations per group
        self.likelihood_name = likelihood
        self.HAS_DISPERSION = likelihood == "negbinomial"

        self.w = torch.nn.Parameter(torch.zeros(M, dtype=_DTYPE))  # unconstrained -- w=0 gives the exact flat field

        mu0 = mu_hat.clamp_min(_priors.MU_FLOOR).log().unsqueeze(1).expand(G, K).clone()
        self.log_mu_gk = torch.nn.Parameter(mu0)  # (G, K)
        if self.HAS_DISPERSION:
            phi0 = phi_hat.log().unsqueeze(1).expand(G, K).clone()
            self.log_phi_gk = torch.nn.Parameter(phi0)  # (G, K)

        gh = _priors.gene_hyperpriors(_priors.PlugInEstimates(mu_hat.numpy(), phi_hat.numpy()))
        self.mu_hyperprior = gh.mu_prior  # plain attr, not registered
        self.phi_hyperprior = gh.phi_prior if self.HAS_DISPERSION else None

        self.sigma2 = float(sigma2)
        self.p0 = float(known_capture_rate)

        self.gene_batch_size = gene_batch_size if (gene_batch_size is not None and gene_batch_size < G) else None
        self.gene_batch_seed = gene_batch_seed
        self._gene_batch_generator = (
            torch.Generator().manual_seed(gene_batch_seed) if self.gene_batch_size is not None else None
        )

    def _group_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum ``x`` (shape ``(n,)``) per group, returning shape ``(K,)``."""
        return torch.zeros(self.K, dtype=_DTYPE).scatter_add_(0, self.group_idx, x)

    def _group_mean_at_locations(self, group_mean: torch.Tensor) -> torch.Tensor:
        """Broadcast a ``(K,)`` per-group mean back out to ``(n,)``, one
        value per location (its own group's mean)."""
        return group_mean[self.group_idx]

    @property
    def p(self) -> torch.Tensor:
        """(n,) capture efficiency. ``u = B @ w`` is centered per group on
        the logit scale, then mapped through a single global (pooled)
        normalization: ``p_i = known_capture_rate * sigmoid(u_tilde_i) /
        mean_i(sigmoid(u_tilde))``.
        """
        u = self.B @ self.w
        group_mean_u = self._group_sum(u) / self._group_count
        u_tilde = u - self._group_mean_at_locations(group_mean_u)
        q = torch.sigmoid(u_tilde)
        return self.p0 * q / q.mean()

    def data_loglik(self) -> torch.Tensor:
        """Raw, full (every gene, regardless of ``gene_batch_size``)
        NB/Poisson log-likelihood only -- excludes the Normal log-prior
        and gene hyperpriors."""
        p = self.p
        mu_gk = self.log_mu_gk.exp()
        if self.HAS_DISPERSION:
            phi_gk = self.log_phi_gk.exp()
            return _likelihood.negbin_loglik(p, mu_gk, phi_gk, self.group_idx, self.Y_ng)
        return _likelihood.poisson_loglik(p, mu_gk, self.group_idx, self.Y_ng)

    def log_joint(self) -> torch.Tensor:
        """NB/Poisson log-likelihood + Normal(0, sigma2) log-prior on w +
        gene hyperpriors.

        When ``gene_batch_size`` is set, a random subset of genes is
        drawn fresh each call (via ``self._gene_batch_generator``); the
        likelihood term is computed on that subset and rescaled by
        ``G / gene_batch_size``, and the gene-hyperprior term is
        restricted to the same subset.
        """
        w_log_prior = torch.distributions.Normal(0.0, self.sigma2**0.5).log_prob(self.w).sum()

        if self.gene_batch_size is None:
            return self._full_log_joint(w_log_prior)

        gene_idx = torch.randperm(self.G, generator=self._gene_batch_generator)[: self.gene_batch_size]
        p = self.p
        mu_gk_full = self.log_mu_gk.exp()
        mu_gk = mu_gk_full[gene_idx]
        Y_ng = self.Y_ng[:, gene_idx]
        scale = self.G / self.gene_batch_size
        if self.HAS_DISPERSION:
            phi_gk_full = self.log_phi_gk.exp()
            phi_gk = phi_gk_full[gene_idx]
            loglik = scale * _likelihood.negbin_loglik(p, mu_gk, phi_gk, self.group_idx, Y_ng)
        else:
            loglik = scale * _likelihood.poisson_loglik(p, mu_gk, self.group_idx, Y_ng)

        hyperprior_log_prob = self.mu_hyperprior.log_prob(mu_gk_full)[gene_idx].sum()
        if self.HAS_DISPERSION:
            hyperprior_log_prob = hyperprior_log_prob + self.phi_hyperprior.log_prob(phi_gk_full)[gene_idx].sum()

        return loglik + w_log_prior + hyperprior_log_prob

    def _full_log_joint(self, w_log_prior: torch.Tensor) -> torch.Tensor:
        loglik = self.data_loglik()
        hyperprior_log_prob = self.mu_hyperprior.log_prob(self.log_mu_gk.exp()).sum()
        if self.HAS_DISPERSION:
            hyperprior_log_prob = hyperprior_log_prob + self.phi_hyperprior.log_prob(self.log_phi_gk.exp()).sum()
        return loglik + w_log_prior + hyperprior_log_prob

    def full_log_joint(self) -> torch.Tensor:
        """Full, unbatched log-joint (every gene), regardless of
        ``gene_batch_size``. Equivalent to :meth:`log_joint` when
        ``gene_batch_size`` is ``None``.
        """
        w_log_prior = torch.distributions.Normal(0.0, self.sigma2**0.5).log_prob(self.w).sum()
        return self._full_log_joint(w_log_prior)


def run_map_optimization(
    model: _StippleBasisModel,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
    convergence_tol: float = 1e-4,
    convergence_window: int = 20,
    verbose_print_interval: int = 5,
    verbose: bool = False,
    debug: bool = False,
    message_fn: Callable[[str], None] | None = None,
) -> tuple[list[float], bool, float]:
    """Full-batch Adam optimization of ``model.log_joint()``.

    Parameters
    ----------
    model
        A constructed, not-yet-fit :class:`_StippleBasisModel`.
    n_iterations
        Maximum number of Adam iterations.
    learning_rate
    convergence_tol
        Convergence is checked via the relative change, over a
        ``convergence_window``-iteration lag, in the full (unbatched)
        objective (:meth:`_StippleBasisModel.full_log_joint`), evaluated
        every ``convergence_window`` iterations.
    verbose_print_interval
        Print (or call ``message_fn`` with) progress every this many
        iterations, if ``verbose``.
    debug
        If True, prints/reports every iteration regardless of
        ``verbose_print_interval``.
    message_fn
        If given, called with each progress message instead of ``print``.

    Returns
    -------
    (history, converged, runtime_seconds)
        ``history`` is the per-iteration objective value
        (``model.log_joint()``, not the loss); ``converged`` is whether the
        tolerance was reached before ``n_iterations``; a
        :class:`UserWarning` is issued (not raised) if not.
    """
    start = time.time()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    history: list[float] = []
    eval_history: list[float] = []
    converged = False
    for it in range(n_iterations):
        optimizer.zero_grad()
        loss = -model.log_joint()
        loss.backward()
        optimizer.step()

        obj_val = -loss.item()
        history.append(obj_val)
        if (verbose and it % verbose_print_interval == 0) or debug:
            msg = f"iteration {it}: objective = {obj_val:.4f}"
            if message_fn is not None:
                message_fn(msg)
            else:
                print(msg, flush=True)

        if it % convergence_window == 0:
            with torch.no_grad():
                eval_val = model.full_log_joint().item()
            eval_history.append(eval_val)
            if len(eval_history) >= 2:
                prev = eval_history[-2]
                rel_change = abs(eval_val - prev) / (abs(prev) + 1e-8)
                if rel_change < convergence_tol:
                    converged = True
                    break

    runtime_seconds = time.time() - start
    model.eval()

    if not converged:
        warnings.warn(
            f"MAP optimization did not converge within {n_iterations} iterations "
            f"(tol={convergence_tol:.2e}, window={convergence_window})",
            stacklevel=2,
        )

    return history, converged, runtime_seconds
