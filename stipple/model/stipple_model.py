"""Spatial radial-basis-function model of per-location RNA capture
efficiency (see :class:`StippleModel`).
"""

from __future__ import annotations

import resource
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.special import xlogy

from . import basis as _basis
from . import data as _data
from . import likelihood as _likelihood
from .model import _StippleBasisModel, run_map_optimization
from .utils import resolve_n_knots_target

CONVERGENCE_WINDOW: int = 20
VERBOSE_PRINT_INTERVAL: int = 5
_LIKELIHOODS = ("negbinomial", "poisson")
_DEFAULT_FINE_RESOLUTION: dict[str, Any] = {
    "n_knots_target": 100, "radius_overlap_factor": 1.5, "basis_kind": "bisquare",
}


def _ensure_raw_layer(adata: AnnData) -> None:
    """Snapshot ``adata.X`` into ``adata.layers["raw"]`` if it doesn't
    already exist. No-op if ``"raw"`` is already present.

    Parameters
    ----------
    adata
    """
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()
        warnings.warn(
            "adata.layers['raw'] did not exist -- adding it now as a snapshot of "
            "adata.X at fit time, so stipple.plotting functions have a stable "
            "reference to the counts this model was fit on, independent of any "
            "later changes to adata.X. Access it directly via adata.layers['raw'] "
            "if needed.",
            UserWarning, stacklevel=3,
        )
# resource.getrusage(...).ru_maxrss units are platform-dependent: bytes on
# macOS, kilobytes on Linux -- this divisor converts either to MB.
_RSS_TO_MB: float = 1e6 if sys.platform == "darwin" else 1e3


class StippleModel:
    """Spatial basis-function model of per-location RNA capture efficiency.

    Parameters
    ----------
    adata
        AnnData with ``adata.X`` (n x G non-negative integer counts, dense
        or sparse), ``adata.obsm['spatial']`` (n x 2, forming a regular
        hexagonal or square grid), and ``adata.obs[group_key]``
        (categorical, no missing values).
    group_key
        Column of ``adata.obs`` giving each location's group label.
    known_capture_rate
        Known/assumed mean capture rate in ``(0, 1)``; ``mean_i(p_i) ==
        known_capture_rate`` exactly for any fitted ``w``.
    likelihood
        ``"negbinomial"`` (default) or ``"poisson"``.
    sigma2
        Fixed prior variance for each basis weight ``w_k ~ Normal(0,
        sigma2)``. Not estimated from data.
    fine_resolution, coarse_resolution
        Each is a dict configuring one basis resolution -- ``{"n_knots_target":
        int (required), "radius_overlap_factor": float (default 1.5),
        "basis_kind": "bisquare"|"gaussian"|"wendland" (default "bisquare")}``
        -- or ``None`` to omit that resolution. At least one of the two
        must be given. ``fine_resolution`` defaults to a single resolution
        (``n_knots_target=100``) when both are omitted. Passing both
        builds a two-resolution basis; passing only ``coarse_resolution``
        fits a single coarse-only basis.

        ``fine_resolution``'s (or a single resolution's) ``n_knots_target``
        below 50% of ``n`` is raised to that floor, with a warning (see
        :func:`stipple.model.resolve_n_knots_target`); no upper clamp.
        ``coarse_resolution`` is exempt from the lower bound.
    resolution_weights
        Only meaningful when both resolutions are given: length-2 list of
        non-negative weights summing to 1, one per resolution (``fine``
        first). ``None`` (default) splits evenly.
    gene_batch_size
        If given (and ``< G``), each training iteration's objective uses a
        random subset of this many genes instead of all ``G``. ``None``
        (default) uses every gene, every iteration. Only affects
        training -- every ``get_*``/``write_*`` method and :meth:`summary`
        always reflect the full-gene fit.
    gene_batch_seed
        Seeds the gene-batch sampling sequence when ``gene_batch_size`` is
        set. Irrelevant otherwise.
    num_threads
        Passed to ``torch.set_num_threads``.
    verbose
        If True, :meth:`message` prints progress during preprocessing and
        fitting.

    Raises
    ------
    ValueError
        If ``known_capture_rate`` is not in ``(0, 1)``; if ``likelihood``
        is not recognized; if ``group_key`` is not in ``adata.obs``; if a
        given ``fine_resolution``/``coarse_resolution`` dict is missing
        ``n_knots_target`` or has an unrecognized ``basis_kind``; or if
        :func:`stipple.model.data.preprocess` / :func:`stipple.model.grid.check_grid`
        rejects the input.
    """

    def __init__(
        self,
        adata: AnnData,
        known_capture_rate: float,
        group_key: str,
        likelihood: str = "negbinomial",
        sigma2: float = 1.0,
        fine_resolution: dict[str, Any] | None = None,
        coarse_resolution: dict[str, Any] | None = None,
        resolution_weights: list[float] | None = None,
        gene_batch_size: int | None = None,
        gene_batch_seed: int = 0,
        num_threads: int = 1,
        verbose: bool = False,
    ) -> None:
        if not 0.0 < known_capture_rate < 1.0:
            raise ValueError(f"known_capture_rate must be in (0, 1), got {known_capture_rate}")
        if likelihood not in _LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {_LIKELIHOODS}, got {likelihood!r}")
        if group_key not in adata.obs.columns:
            raise ValueError(f"group_key {group_key!r} not found in adata.obs")

        if fine_resolution is None and coarse_resolution is None:
            fine_resolution = _DEFAULT_FINE_RESOLUTION
        raw_resolutions = [(fine_resolution, False), (coarse_resolution, True)]
        resolutions: list[dict[str, Any]] = []
        is_coarse_flags: list[bool] = []
        for cfg, is_coarse in raw_resolutions:
            if cfg is None:
                continue
            resolved = _basis.resolve_resolution_config(cfg)
            if resolved["basis_kind"] not in _basis._BASIS_KINDS:
                raise ValueError(f"basis_kind must be one of {_basis._BASIS_KINDS}, got {resolved['basis_kind']!r}")
            resolutions.append(resolved)
            is_coarse_flags.append(is_coarse)

        self.verbose = verbose
        self.likelihood_name = likelihood
        self.sigma2 = float(sigma2)
        self.known_capture_rate = known_capture_rate
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        self._debug: bool = False
        self._debug_start: float | None = None
        self._debug_prev_time: float | None = None

        self.message("preprocessing data")
        self.group_key = group_key
        self.data = _data.preprocess(adata, group_key)
        self.message(
            f"detected {self.data.grid_info.grid_type} grid, "
            f"spacing={self.data.grid_info.spacing:.4g}, "
            f"{self.data.n} locations, {self.data.G} genes, {self.data.K} groups"
        )

        for cfg, is_coarse in zip(resolutions, is_coarse_flags):
            cfg["n_knots_target"] = resolve_n_knots_target(
                cfg["n_knots_target"], self.data.n, enforce_min=not is_coarse
            )
        self.resolutions = resolutions

        B_np, meta = _basis.make_multires_basis(self.data.coords_raw, resolutions, resolution_weights)
        self.knots = meta["knots_per_resolution"]  # list[np.ndarray], length = len(resolutions)
        self.radius = meta["radius_per_resolution"]  # list[float]
        self.M_per_resolution = meta["M_per_resolution"]  # list[int]
        self.resolution_weights = meta["resolution_weights"]  # list[float]
        B = torch.as_tensor(B_np, dtype=torch.float64)
        resolution_desc = ", ".join(
            f"{cfg['basis_kind']}:{m}" for cfg, m in zip(resolutions, self.M_per_resolution)
        )
        self.message(f"built basis: {len(resolutions)} resolution(s), M={B.shape[1]} total ({resolution_desc})")

        self.model = _StippleBasisModel(
            B,
            self.data.Y_gn,
            self.data.group_idx,
            self.data.G,
            self.data.K,
            self.data.n,
            self.data.mu_hat,
            self.data.phi_hat,
            self.sigma2,
            known_capture_rate,
            likelihood=likelihood,
            gene_batch_size=gene_batch_size,
            gene_batch_seed=gene_batch_seed,
        )

        self._fitted = False
        self._objective_history: list[float] = []
        self._converged: bool | None = None
        self._runtime_seconds: float | None = None

    def message(self, msg: str) -> None:
        """Print ``msg`` prefixed with ``"[stipple] "`` if ``self.verbose``.
        If debug mode is active (see :meth:`fit`'s ``debug`` argument),
        also prints a timing/memory/fitted-field diagnostic line.

        Parameters
        ----------
        msg
        """
        if self.verbose:
            print(f"[stipple] {msg}", flush=True)
        if self._debug:
            now = time.time()
            elapsed_str = f"{now - self._debug_start:.2f}s" if self._debug_start is not None else "n/a"
            iter_dt_str = f"{now - self._debug_prev_time:.3f}s" if self._debug_prev_time is not None else "n/a"
            self._debug_prev_time = now
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_TO_MB
            with torch.no_grad():
                mean_p_hat = float(self.model.p.mean().item())
                w_hat_std = float(self.model.w.std().item())
            print(
                f"[stipple debug] {msg} | iter_dt={iter_dt_str} elapsed={elapsed_str} "
                f"M={self.model.M} rss_mb={rss_mb:.1f} mean_p_hat={mean_p_hat:.4f} w_hat_std={w_hat_std:.4f}",
                flush=True,
            )

    def fit(
        self,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        convergence_tol: float = 1e-4,
        debug: bool = False,
    ) -> None:
        """Fit the model with Adam, full-batch, maximizing the penalized
        MAP log-joint.

        Parameters
        ----------
        n_iterations
            Maximum number of Adam iterations.
        learning_rate
        convergence_tol
            Convergence is checked via the relative change in the
            objective over a :data:`CONVERGENCE_WINDOW`-iteration lag; if
            below this tolerance, training stops early.
        debug
            If True, prints a timing/memory diagnostic and the current
            fitted-field summary (``mean_p_hat``, ``w_hat_std``) every
            iteration, via :meth:`message` -- independent of
            ``verbose``/:data:`VERBOSE_PRINT_INTERVAL`.
        """
        self._debug = debug
        self._debug_start = time.time()
        self._debug_prev_time = None

        history, converged, runtime_seconds = run_map_optimization(
            self.model,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            convergence_tol=convergence_tol,
            convergence_window=CONVERGENCE_WINDOW,
            verbose_print_interval=VERBOSE_PRINT_INTERVAL,
            verbose=self.verbose,
            debug=debug,
            message_fn=self.message,
        )

        self._objective_history = history
        self._converged = converged
        self._fitted = True
        self._runtime_seconds = runtime_seconds
        self._debug = False
        self._debug_prev_time = None

    def fit_anndata(
        self,
        adata: AnnData,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        convergence_tol: float = 1e-4,
        debug: bool = False,
        copy: bool = False,
    ) -> AnnData | None:
        """Fit the model and write every result into ``adata``.

        Convenience wrapper: ``self.fit(...)`` followed by
        ``self.set_anndata_results(adata, copy=copy)``.

        Parameters
        ----------
        adata
            Must have the same locations/genes, in the same order, as the
            data this model was constructed with (see
            :meth:`set_anndata_results`).
        n_iterations, learning_rate, convergence_tol, debug
            Passed to :meth:`fit`.
        copy
            Scanpy convention. If False (default), mutate ``adata`` in
            place and return None. If True, operate on and return a copy,
            leaving ``adata`` untouched.

        Returns
        -------
        AnnData or None
            The updated copy if ``copy=True``, else None.
        """
        self.fit(
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            convergence_tol=convergence_tol,
            debug=debug,
        )
        return self.set_anndata_results(adata, copy=copy)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model has not been fit; call .fit() first")

    def get_capture_rates(self) -> np.ndarray:
        """Estimated capture rate ``p_hat_i`` for every location, shape ``(n,)``.

        No posterior standard deviation is returned: ``w`` is a
        deterministic MAP point estimate (not a variationally-distributed
        quantity), so ``p_i`` has no analogous posterior uncertainty to
        report.
        """
        self._check_fitted()
        with torch.no_grad():
            return self.model.p.numpy()

    def get_group_assignment(self) -> np.ndarray:
        """Group assignment for every location, shape ``(n,)``, values in
        ``[0, K)`` (``self.data.group_labels[group_idx[i]]`` recovers the
        original label)."""
        self._check_fitted()
        return self.data.group_idx.numpy()

    def get_group_expression(self) -> np.ndarray:
        """Estimated ``mu_gk``, shape ``(G, K)``."""
        self._check_fitted()
        with torch.no_grad():
            return self.model.log_mu_gk.exp().numpy()

    def get_dispersion(self) -> np.ndarray:
        """Estimated ``phi_gk``, shape ``(G, K)``.

        Raises
        ------
        ValueError
            If the model was fit with the Poisson likelihood (no
            dispersion parameter).
        """
        self._check_fitted()
        if not self.model.HAS_DISPERSION:
            raise ValueError(
                f"get_dispersion() is only available for the 'negbinomial' likelihood, "
                f"this model was fit with {self.likelihood_name!r}"
            )
        with torch.no_grad():
            return self.model.log_phi_gk.exp().numpy()

    def get_fitted_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-location, per-gene fitted mean and variance implied by the
        MAP fit.

        ``fitted_mean[i, g] = mu_hat_gk[g, k(i)] * p_hat_i`` — the
        estimated group-level expression scaled by that location's
        estimated capture rate. ``fitted_variance`` is the corresponding
        likelihood variance (``mu`` for Poisson; ``mu + mu**2/phi`` for
        NegBin).

        Returns
        -------
        fitted_mean, fitted_variance : np.ndarray, each shape (n, G)
        """
        self._check_fitted()
        p_hat = self.get_capture_rates()  # (n,)
        mu_gk_hat = self.get_group_expression()  # (G, K)
        group_idx = self.data.group_idx.numpy()  # (n,)
        mu_ng = mu_gk_hat[:, group_idx].T * p_hat[:, None]  # (n, G)

        if self.model.HAS_DISPERSION:
            phi_gk_hat = self.get_dispersion()
            phi_ng = phi_gk_hat[:, group_idx].T  # (n, G)
            var_ng = mu_ng + mu_ng**2 / phi_ng
        else:
            var_ng = mu_ng

        return mu_ng, var_ng

    def get_residuals(self, type: str = "pearson") -> np.ndarray:
        """Pearson or deviance residuals at the MAP ``p_hat_i``,
        ``mu_hat_gk`` (and ``phi_hat_gk`` for NegBin).

        Parameters
        ----------
        type
            ``"pearson"`` or ``"deviance"``.

        Returns
        -------
        np.ndarray, shape (G, n)
        """
        self._check_fitted()
        if type not in ("pearson", "deviance"):
            raise ValueError(f"type must be 'pearson' or 'deviance', got {type!r}")

        mu_ng, var_ng = self.get_fitted_values()  # (n, G) each
        Y_ng = self.model.Y_ng.numpy()  # (n, G)

        if self.model.HAS_DISPERSION:
            phi_gk_hat = self.get_dispersion()
            group_idx = self.data.group_idx.numpy()
            phi_ng = phi_gk_hat[:, group_idx].T  # (n, G)
        else:
            phi_ng = None

        if type == "pearson":
            resid_ng = (Y_ng - mu_ng) / np.sqrt(var_ng)
        else:
            if phi_ng is not None:
                deviance = 2.0 * (
                    xlogy(Y_ng, Y_ng / mu_ng) - (Y_ng + phi_ng) * np.log((Y_ng + phi_ng) / (mu_ng + phi_ng))
                )
            else:
                deviance = 2.0 * (xlogy(Y_ng, Y_ng / mu_ng) - (Y_ng - mu_ng))
            deviance = np.clip(deviance, 0.0, None)
            resid_ng = np.sign(Y_ng - mu_ng) * np.sqrt(deviance)

        return resid_ng.T  # (G, n)

    def get_basis_params(self) -> dict[str, Any]:
        """Basis configuration and fitted-weight summary.

        Every per-resolution field is a list, in the same (fine-first)
        order as ``resolutions`` was resolved in the constructor -- length
        1 for a single-resolution basis, length 2 for a fine+coarse basis.

        Returns
        -------
        dict
            ``n_resolutions``, ``basis_kind`` (list[str]),
            ``n_knots_target`` (list[int]), ``radius_overlap_factor``
            (list[float]), ``radius`` (list[float], actual radius used per
            resolution), ``M_per_resolution`` (list[int]),
            ``resolution_weights`` (list[float]), ``M`` (int, total basis
            functions across all resolutions), ``sigma2`` (the fixed prior
            variance), and fitted-weight summary statistics
            (``w_hat_mean``, ``w_hat_std``, ``w_hat_min``, ``w_hat_max``).
        """
        self._check_fitted()
        with torch.no_grad():
            w = self.model.w.numpy()
        return {
            "n_resolutions": len(self.resolutions),
            "basis_kind": [cfg["basis_kind"] for cfg in self.resolutions],
            "n_knots_target": [cfg["n_knots_target"] for cfg in self.resolutions],
            "radius_overlap_factor": [cfg["radius_overlap_factor"] for cfg in self.resolutions],
            "radius": list(self.radius),
            "M_per_resolution": list(self.M_per_resolution),
            "resolution_weights": list(self.resolution_weights),
            "sigma2": self.sigma2,
            "M": self.model.M,
            "w_hat_mean": float(w.mean()),
            "w_hat_std": float(w.std()),
            "w_hat_min": float(w.min()),
            "w_hat_max": float(w.max()),
        }

    def write_obs(self, adata: AnnData) -> None:
        """Write estimated capture rates into ``adata.obs``, in place.

        Adds ``adata.obs["stipple_p_hat"]`` (see :meth:`get_capture_rates`).
        Also snapshots ``adata.X`` into ``adata.layers["raw"]`` the first
        time results are written for this ``adata`` (no-op if already
        present) -- see :func:`_ensure_raw_layer`.

        Parameters
        ----------
        adata
            Must have the same number of locations (``adata.n_obs``), in
            the same order, as the data this model was fit on.
        """
        self._check_fitted()
        if adata.n_obs != self.data.n:
            raise ValueError(
                f"adata.n_obs ({adata.n_obs}) does not match the number of "
                f"locations this model was fit on ({self.data.n})"
            )
        _ensure_raw_layer(adata)
        adata.obs["stipple_p_hat"] = self.get_capture_rates()

    def write_obsm(self, adata: AnnData) -> None:
        """Write per-location, per-gene fitted values into ``adata.obsm``, in place.

        Adds ``adata.obsm["stipple_fitted_mean"]`` and
        ``adata.obsm["stipple_fitted_variance"]`` (each shape ``(n, G)``,
        see :meth:`get_fitted_values`) -- the mean/variance implied by
        the MAP fit at each location, depending on that location's
        estimated capture rate ``p_hat_i``.

        Parameters
        ----------
        adata
            Must have the same number of locations and genes, in the
            same order, as the data this model was fit on.
        """
        self._check_fitted()
        if adata.n_obs != self.data.n:
            raise ValueError(
                f"adata.n_obs ({adata.n_obs}) does not match the number of "
                f"locations this model was fit on ({self.data.n})"
            )
        if adata.n_vars != self.data.G:
            raise ValueError(
                f"adata.n_vars ({adata.n_vars}) does not match the number of "
                f"genes this model was fit on ({self.data.G})"
            )
        fitted_mean, fitted_variance = self.get_fitted_values()
        adata.obsm["stipple_fitted_mean"] = fitted_mean
        adata.obsm["stipple_fitted_variance"] = fitted_variance

    def write_varm(self, adata: AnnData) -> None:
        """Write gene-by-group parameter estimates into ``adata.varm``, in place.

        Adds ``adata.varm["stipple_mu_gk"]`` (``G x K``, see
        :meth:`get_group_expression`) and, for the negative-binomial
        likelihood, ``adata.varm["stipple_phi_gk"]`` (see :meth:`get_dispersion`).

        Parameters
        ----------
        adata
            Must have the same number of genes (``adata.n_vars``), in the
            same order, as the data this model was fit on.
        """
        self._check_fitted()
        if adata.n_vars != self.data.G:
            raise ValueError(
                f"adata.n_vars ({adata.n_vars}) does not match the number of "
                f"genes this model was fit on ({self.data.G})"
            )
        adata.varm["stipple_mu_gk"] = self.get_group_expression()
        if self.model.HAS_DISPERSION:
            adata.varm["stipple_phi_gk"] = self.get_dispersion()

    def write_uns(self, adata: AnnData) -> None:
        """Write scalar/short model parameters into ``adata.uns["stipple"]``, in place.

        Excludes anything shaped like a per-location (length ``n``) array
        -- those belong in ``adata.obs`` (see :meth:`write_obs`) -- but
        keeps short per-group values such as ``group_labels`` (length
        ``K``) and the objective trajectory (length = iterations run, not
        ``n``).
        """
        self._check_fitted()
        basis_params = self.get_basis_params()
        p_hat = self.get_capture_rates()
        adata.uns["stipple"] = {
            "likelihood": self.likelihood_name,
            "n_resolutions": basis_params["n_resolutions"],
            "basis_kind": basis_params["basis_kind"],
            "n_knots_target": basis_params["n_knots_target"],
            "M": basis_params["M"],
            "M_per_resolution": basis_params["M_per_resolution"],
            "radius": basis_params["radius"],
            "radius_overlap_factor": basis_params["radius_overlap_factor"],
            "resolution_weights": basis_params["resolution_weights"],
            "sigma2": basis_params["sigma2"],
            "w_hat_mean": basis_params["w_hat_mean"],
            "w_hat_std": basis_params["w_hat_std"],
            "known_capture_rate": self.known_capture_rate,
            "mean_capture_rate_hat": float(np.mean(p_hat)),
            "grid_type": self.data.grid_info.grid_type,
            "grid_spacing": self.data.grid_info.spacing,
            "group_key": self.group_key,  # adata.obs column name -- lets stipple.plotting
                                            # resolve each location's group without a live model
            "group_labels": list(self.data.group_labels),  # length K -- short, kept
            "n_locations": self.data.n,
            "n_genes": self.data.G,
            "n_groups": self.data.K,
            "converged": self._converged,
            "n_iterations_run": len(self._objective_history),
            "final_objective": self._objective_history[-1] if self._objective_history else float("nan"),
            "objective_history": np.asarray(self._objective_history),  # length = iterations run, not n
            "runtime_seconds": self._runtime_seconds,
        }

    def set_anndata_results(self, adata: AnnData, copy: bool = False) -> AnnData | None:
        """Write every fit result into ``adata``.

        Convenience wrapper around :meth:`write_obs`, :meth:`write_obsm`,
        :meth:`write_varm`, and :meth:`write_uns`. Returning the target
        (when ``copy=True``) lets a future ``fit_anndata`` chain
        ``self.fit(...); return self.set_anndata_results(adata, copy=copy)``.

        Parameters
        ----------
        adata
            Must have the same locations/genes, in the same order, as the
            data this model was fit on (see :meth:`write_obs`,
            :meth:`write_obsm`, :meth:`write_varm`).
        copy
            Scanpy convention. If False (default), mutate ``adata`` in
            place and return None. If True, operate on and return a
            copy, leaving ``adata`` untouched.

        Returns
        -------
        AnnData or None
            The updated copy if ``copy=True``, else None.
        """
        self._check_fitted()
        target = adata.copy() if copy else adata
        self.write_obs(target)
        self.write_obsm(target)
        self.write_varm(target)
        self.write_uns(target)
        return target if copy else None

    def write_gene_params_csv(self, path: str) -> None:
        """Write this fitted model's gene-level parameters to a CSV at
        ``path``, in long/tidy format: one row per ``(gene, group,
        parameter)``. ``mu_gk`` is always written (``parameter="mu"``);
        ``phi_gk`` is added (``parameter="dispersion"``) for the
        negative-binomial likelihood.

        Parameters
        ----------
        path
            Output CSV path.
        """
        self._check_fitted()

        mu_gk = self.get_group_expression()  # (G, K)
        phi_gk = self.get_dispersion() if self.model.HAS_DISPERSION else None

        rows = []
        for g, gene in enumerate(self.data.gene_names):
            for k, group in enumerate(self.data.group_labels):
                rows.append({"gene": gene, "group": group, "parameter": "mu", "estimated": float(mu_gk[g, k])})
                if phi_gk is not None:
                    rows.append(
                        {"gene": gene, "group": group, "parameter": "dispersion", "estimated": float(phi_gk[g, k])}
                    )

        pd.DataFrame(rows).to_csv(path, index=False)

    def write_observation_csv(self, adata: AnnData, path: str) -> None:
        """Write this fitted model's per-observation results to a CSV at
        ``path`` -- one row per observation: ``obs_id``, spatial
        coordinates, group assignment, fitted capture rate, and a handful
        of model-level constants repeated on every row.

        Parameters
        ----------
        adata
            Must have the same number of locations, in the same order, as
            the data this model was fit on.
        path
            Output CSV path.
        """
        self._check_fitted()
        if adata.n_obs != self.data.n:
            raise ValueError(
                f"adata.n_obs ({adata.n_obs}) does not match the number of "
                f"locations this model was fit on ({self.data.n})"
            )

        p_hat = self.get_capture_rates()
        basis_params = self.get_basis_params()
        group_idx = self.data.group_idx.numpy()
        group = np.asarray(self.data.group_labels, dtype=object)[group_idx]

        data = {
            "obs_id": adata.obs_names.to_numpy(),
            "x": self.data.coords_raw[:, 0],
            "y": self.data.coords_raw[:, 1],
            "group": group,
            "group_idx": group_idx,
            "p_hat": p_hat,
            "n_resolutions": basis_params["n_resolutions"],
            "basis_kind": "+".join(basis_params["basis_kind"]),
            "M": basis_params["M"],
            "sigma2": basis_params["sigma2"],
            "likelihood": self.likelihood_name,
            "known_capture_rate": self.known_capture_rate,
            "converged": self._converged,
        }
        pd.DataFrame(data).to_csv(path, index=False)

    def summary(self) -> None:
        """Print a formatted summary of the fitted model."""
        self._check_fitted()
        p_hat = self.get_capture_rates()
        basis_params = self.get_basis_params()
        final_objective = self._objective_history[-1] if self._objective_history else float("nan")

        resolution_desc = ", ".join(
            f"{kind}(M={m}, target={t}, radius={r:.4g})"
            for kind, m, t, r in zip(
                basis_params["basis_kind"], basis_params["M_per_resolution"],
                basis_params["n_knots_target"], basis_params["radius"],
            )
        )

        print("[stipple] StippleModel summary")
        print(f"  likelihood: {self.likelihood_name}")
        print(f"  sigma2: {basis_params['sigma2']:.4f}")
        print(f"  basis: {basis_params['n_resolutions']} resolution(s), M={basis_params['M']} total -- {resolution_desc}")
        print(f"  genes (G): {self.data.G}, locations (n): {self.data.n}, groups (K): {self.data.K}")
        print(f"  known capture rate: {self.known_capture_rate:.4f}")
        print(f"  estimated mean capture rate: {float(np.mean(p_hat)):.4f}")
        print(f"  w_hat: mean={basis_params['w_hat_mean']:.4f}, std={basis_params['w_hat_std']:.4f}")
        print(f"  final objective: {final_objective:.4f}")
        print(f"  converged: {self._converged}")
        print(f"  runtime: {self._runtime_seconds:.2f}s" if self._runtime_seconds is not None else "  runtime: n/a")
