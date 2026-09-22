"""Diagnostic plotting utilities for a fitted StippleModel model, read from an
AnnData object.

Spatial (:mod:`stipple.plotting.spatial`): plot_capture_rate_surface,
plot_spatial_residuals, plot_residual_variogram, plot_fitted_vs_raw_surface
(raw counts vs. fitted expected counts vs. their difference),
plot_fitted_parameter_surface (fitted mu_gk/phi_gk, painted spatially by
each location's group).
Fit diagnostics (:mod:`stipple.plotting.diagnostics`): plot_residuals_vs_fitted,
plot_count_distribution, plot_objective_trace (the MAP-fit objective
trajectory), plot_qq_residuals (Dunn-Smyth residuals against N(0,1), gene
required).
Exploratory, pre-fit (:mod:`stipple.plotting.exploratory`):
plot_mean_variance_by_space, plot_quantile_outlier_score -- only these two
read raw ``adata.X``/``adata.obsm["spatial"]`` directly and work before any
model has been fit; every other function below requires
``model.set_anndata_results(adata)`` to have been run first.

Every function takes a plain ``adata`` object as its first argument, never
the live ``StippleModel`` model -- an AnnData is what actually gets saved/reloaded
across sessions, while a fitted Python model object does not survive
serialization. Reads fitted results from the standard slots
``set_anndata_results``/``write_obs``/``write_obsm``/``write_varm``/
``write_uns`` write into ``adata`` (see ``stipple.model.StippleModel``'s own docstrings for
exactly which slot each of those methods fills), and raw counts from
``adata.layers["raw"]`` (auto-snapshotted from ``adata.X`` at the first
``write_obs`` call, falling back to ``adata.X`` with a warning otherwise).

Shared parameters, common to (almost) every function below:

  gene          : int, str (a name from adata.var_names), a list/tuple
                  mixing either, or None (pools/sums every gene). A name
                  not found there is dropped with a warning rather than
                  raised; ValueError if nothing valid remains. A list/tuple
                  renders one panel per gene as a grid -- return value is
                  then a 2D Axes array. spatial.py/diagnostics.py only.
                  plot_fitted_parameter_surface requires an explicit gene
                  (None raises -- pooling mu_gk/phi_gk across genes has no
                  clean spatial interpretation).
  residual_type : "pearson" (default) or "deviance" -- matches
                  ``StippleModel.get_residuals``'s own ``type`` parameter's
                  accepted values (though not its parameter name).
                  plot_spatial_residuals, plot_residual_variogram,
                  plot_residuals_vs_fitted only. plot_spatial_residuals also
                  accepts "dunn_smyth" (randomized quantile residuals --
                  exactly N(0, 1) under a correctly-specified model, unlike
                  "pearson"/"deviance"; see
                  ``stipple.utils.compute_dunn_smyth_residuals`` and each
                  function's own ``random_state`` parameter).
                  plot_qq_residuals has no ``residual_type`` parameter -- it
                  always uses Dunn-Smyth residuals (the only type that's
                  exactly N(0, 1), making its reference line/curve a real
                  test) and requires an explicit ``gene`` (pooling breaks
                  the exact N(0, 1) guarantee -- see its own docstring).
  ax            : a matplotlib Axes, or an array of Axes for multi-panel
                  plots. None (default) creates and returns a new one.
                  return_fig=True + a caller-supplied ax raises ValueError.
  size          : scatter marker size (matplotlib's `s`), default 1.5.
                  None falls back to that call's own internal default;
                  scatter_kwargs overrides it again if given.
  figsize       : passed to plt.subplots() when ax is None.
  title, xlabel, ylabel : default auto-generates today's text; an
                  explicit string uses it verbatim; explicit None removes
                  it. Multi-panel calls apply a literal override
                  identically to every panel.
  font_sizes    : optional dict, any of "title" (plot title), "axis_title"
                  (xlabel/ylabel, and a colorbar's own label), "tick_label"
                  (tick numbers on any axis, including a colorbar's).
                  Missing keys/None (default) leave matplotlib's own
                  rcParams default untouched.
  tick_kwargs   : optional dict -> ax.tick_params (and a colorbar's own
                  ticks, where present) -- e.g. length=, width=. Applied
                  after font_sizes["tick_label"], so it can override
                  labelsize too.
  cmap          : colormap for any color-mapped dimension (spatial only).
  origin        : "upper" (default) or "lower". Spatial scatter panels only.
  *_kwargs      : one named dict per distinct matplotlib call a function
                  makes (default None == {}) -- e.g. scatter_kwargs ->
                  ax.scatter, hist_kwargs -> ax.hist, line_kwargs/
                  refline_kwargs -> ax.plot, cbar_kwargs -> plt.colorbar.
                  Each function's own docstring states which call each
                  dict forwards to (these vary per function, unlike
                  font_sizes/tick_kwargs above, which are identical
                  everywhere). No function accepts **kwargs.
  save_source_csv : optional path -- writes the plotted data to CSV.
  return_fig    : False (default) returns Axes/array-of-Axes; True
                  returns the owning Figure instead.
  save          : optional path -- fig.savefig(save).
  show          : True (default) calls plt.show().

Works for both likelihoods (negbinomial/poisson) and any basis
configuration (basis kind, single- or multi-resolution) -- none of these
choices change any plotting function's signature or behavior, only what's
available in ``adata`` (e.g. no ``adata.varm["stipple_phi_gk"]`` for a
Poisson fit).

When generating many plots in a loop, matplotlib holds each created
figure open in memory until it's closed -- past its default of 20 open
figures it warns about memory use. Close each figure once you're done
with it, e.g. ``plt.close(ax.get_figure())`` (or ``ax[0].get_figure()``
for multi-panel plots).
"""
from __future__ import annotations

from .diagnostics import (
    plot_count_distribution, plot_objective_trace, plot_qq_residuals, plot_residuals_vs_fitted,
)
from .exploratory import plot_mean_variance_by_space, plot_quantile_outlier_score
from .spatial import (
    plot_capture_rate_surface, plot_fitted_parameter_surface, plot_fitted_vs_raw_surface,
    plot_residual_variogram, plot_spatial_residuals,
)

__all__ = [
    "plot_capture_rate_surface",
    "plot_spatial_residuals",
    "plot_residual_variogram",
    "plot_fitted_vs_raw_surface",
    "plot_fitted_parameter_surface",
    "plot_residuals_vs_fitted",
    "plot_count_distribution",
    "plot_objective_trace",
    "plot_qq_residuals",
    "plot_mean_variance_by_space",
    "plot_quantile_outlier_score",
]
