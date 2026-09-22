"""Fit-quality / distributional diagnostics: plot_residuals_vs_fitted,
plot_count_distribution, plot_objective_trace, plot_qq_residuals.

Every function here requires ``model.set_anndata_results(adata)`` to have
been run first. See :mod:`stipple.plotting` for the parameter conventions
shared across the whole package.
"""
from __future__ import annotations

import numpy as np

from ._utils import (
    _UNSET, _apply_label, _apply_tick_params, _finalize, _gene_label, _get_coords, _get_counts,
    _grid_dims, _merge_scatter_kwargs, _normalize_genes, _residuals,
    _select_genes, _fitted_mean_variance, _write_source_csv,
)
from .spatial import _spatial_residuals_panel


def plot_residuals_vs_fitted(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    residual_type: str = "pearson",
    ax=None,
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    refline_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Scatter of per-location residuals against the fitted mean count.

    Parameters
    ----------
    residual_type
        ``"pearson"`` (default) or ``"deviance"``.
    scatter_kwargs
        Forwarded to ``ax.scatter`` (default alpha=0.6).
    refline_kwargs
        Forwarded to ``ax.axhline`` (the y=0 reference line).
    """
    import matplotlib.pyplot as plt

    ax_given = ax is not None
    genes = _normalize_genes(gene, adata)

    if genes is not None and len(genes) > 1:
        n_rows, n_cols = _grid_dims(len(genes))
        if ax is None:
            _, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize or (5 * n_cols, 5 * n_rows), dpi=300, squeeze=False
            )
            ret = axes
        else:
            axes = np.atleast_2d(ax)
            ret = ax
        flat = axes.reshape(-1)
        if flat.size < len(genes):
            raise ValueError(f"ax provides {flat.size} panel(s) but {len(genes)} genes were requested.")
        dfs = [
            _residuals_vs_fitted_panel(
                adata, g, residual_type, a, size, title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, refline_kwargs,
            )
            for a, g in zip(flat, genes)
        ]
        for a in flat[len(genes):]:
            a.set_visible(False)
        if save_source_csv is not None:
            _write_source_csv(save_source_csv, dfs)
    else:
        single_gene = genes[0] if genes is not None else None
        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (6, 6), dpi=300)
        df = _residuals_vs_fitted_panel(
            adata, single_gene, residual_type, ax, size, title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, refline_kwargs,
        )
        if save_source_csv is not None:
            _write_source_csv(save_source_csv, [df])
        ret = ax

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _residuals_vs_fitted_panel(
    adata, gene, residual_type, ax, size, title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, refline_kwargs
):
    import pandas as pd

    fitted, _ = _fitted_mean_variance(adata, gene=gene)
    resid = _residuals(adata, gene=gene, residual_type=residual_type)

    kw = _merge_scatter_kwargs({"s": 8, "alpha": 0.6}, size, scatter_kwargs)
    ax.scatter(fitted, resid, **kw)
    rkw = {"color": "k", "ls": "--", "lw": 1, "alpha": 0.7}
    rkw.update(refline_kwargs or {})
    ax.axhline(0.0, **rkw)
    _apply_label(ax, "xlabel", xlabel, "fitted mean count", font_sizes)
    _apply_label(ax, "ylabel", ylabel, f"{residual_type} residual", font_sizes)
    ax.set_box_aspect(1)
    _apply_label(ax, "title", title, f"Residuals vs. fitted ({_gene_label(gene, adata.var_names)})", font_sizes)
    _apply_tick_params(ax, font_sizes, tick_kwargs)

    return pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names),
        "fitted": fitted,
        "type": residual_type,
        "residual": resid,
    })


def plot_count_distribution(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    by_group: bool = False,
    bins: int | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
    fitted_line_kwargs: dict | None = None,
    moment_match_line_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Histogram of observed counts with the fitted count distribution and
    a moment-matched distribution overlaid as lines -- one row per gene
    selected. The "fitted" line is the mixture-PMF over each location's
    own fitted (mean, variance); the "moment-matched" line treats the
    observed counts as IID draws with plug-in sample mean/variance.

    Parameters
    ----------
    by_group
        If True, renders one column per group (requires
        ``adata.uns["stipple"]["group_key"]``), shape ``(n_genes,
        n_groups)``.
    bins
        ``None`` (default) bins at integer counts; an int uses that many
        bins instead.
    hist_kwargs
        Forwarded to ``ax.hist`` (observed counts).
    fitted_line_kwargs, moment_match_line_kwargs
        Forwarded to ``ax.plot`` for the fitted-mixture / moment-matched
        PMF lines.
    """
    import matplotlib.pyplot as plt

    ax_given = ax is not None
    likelihood = adata.uns["stipple"]["likelihood"]
    genes = _normalize_genes(gene, adata)
    gene_rows = genes if genes is not None else [None]
    n_rows = len(gene_rows)

    if by_group:
        group_key = adata.uns["stipple"].get("group_key")
        if group_key is None:
            raise ValueError(
                "by_group=True requires adata.uns['stipple']['group_key'] -- "
                "re-run model.set_anndata_results(adata) to populate it."
            )
        group_labels_all = np.asarray(adata.obs[group_key])
        group_labels = adata.uns["stipple"]["group_labels"]
    else:
        group_labels_all = None
        group_labels = [None]
    n_cols = len(group_labels)

    if ax is None:
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (5 * n_cols, 4 * n_rows), dpi=300, squeeze=False
        )
        ret = axes
    else:
        axes = ax
        ret = ax

    axes_rows = np.atleast_2d(axes)
    if axes_rows.shape[0] < n_rows or axes_rows.shape[1] < n_cols:
        raise ValueError(f"ax must provide at least {n_rows} row(s) x {n_cols} column(s) for {n_rows} gene(s).")

    dfs = [
        _count_distribution_row(
            adata, g, row[:n_cols], group_labels, group_labels_all, likelihood, bins,
            title, xlabel, ylabel, font_sizes, tick_kwargs, hist_kwargs, fitted_line_kwargs,
            moment_match_line_kwargs,
        )
        for row, g in zip(axes_rows, gene_rows)
    ]
    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _count_distribution_row(
    adata, gene, row_axes, group_labels, group_labels_all, likelihood, bins,
    title, xlabel, ylabel, font_sizes, tick_kwargs, hist_kwargs, fitted_line_kwargs,
    moment_match_line_kwargs,
):
    import pandas as pd
    from scipy.stats import nbinom, poisson

    observed_all = _select_genes(_get_counts(adata), gene)
    mean_all, var_all = _fitted_mean_variance(adata, gene=gene)
    subtitle = f" ({_gene_label(gene, adata.var_names)})"

    dfs = []
    for ax, glabel in zip(row_axes, group_labels):
        mask = slice(None) if glabel is None else group_labels_all == glabel
        group_suffix = "" if glabel is None else f" | {glabel}"

        y = observed_all[mask]
        mu = mean_all[mask]
        var = var_all[mask]
        max_count = int(y.max()) if y.size else 0

        hist_bins = bins if bins is not None else np.arange(0, max_count + 2) - 0.5
        h_kw = {"density": True, "alpha": 0.7, "color": "#3A7CA5", "edgecolor": "white", "label": "observed"}
        h_kw.update(hist_kwargs or {})
        ax.hist(y, bins=hist_bins, **h_kw)

        k = np.arange(0, max_count + 1)
        if not mu.size:
            pmf = np.zeros_like(k, dtype=float)
        elif likelihood == "poisson":
            pmf = poisson.pmf(k[:, None], mu[None, :]).mean(axis=1)
        else:
            phi = np.clip(mu**2 / np.maximum(var - mu, 1e-6), 1e-3, 1e6)
            p = phi / (phi + mu)
            pmf = nbinom.pmf(k[:, None], phi[None, :], p[None, :]).mean(axis=1)

        kw = {"color": "k", "ls": "--", "lw": 1.5, "marker": "o", "markersize": 3, "label": "fitted"}
        kw.update(fitted_line_kwargs or {})
        ax.plot(k, pmf, **kw)

        if y.size:
            mm_mean = float(y.mean())
            mm_var = float(y.var(ddof=1)) if y.size > 1 else 0.0
            if likelihood == "poisson":
                mm_pmf = poisson.pmf(k, mm_mean)
            else:
                mm_phi = np.clip(mm_mean**2 / max(mm_var - mm_mean, 1e-6), 1e-3, 1e6)
                mm_p = mm_phi / (mm_phi + mm_mean)
                mm_pmf = nbinom.pmf(k, mm_phi, mm_p)
        else:
            mm_pmf = np.zeros_like(k, dtype=float)

        mm_kw = {"color": "#D9534F", "ls": "-.", "lw": 1.5, "marker": "s", "markersize": 3, "label": "moment-matched"}
        mm_kw.update(moment_match_line_kwargs or {})
        ax.plot(k, mm_pmf, **mm_kw)

        _apply_label(ax, "xlabel", xlabel, "count", font_sizes)
        _apply_label(ax, "ylabel", ylabel, "density", font_sizes)
        ax.legend(fontsize=8)
        _apply_label(ax, "title", title, "Count distribution" + subtitle + group_suffix, font_sizes)
        _apply_tick_params(ax, font_sizes, tick_kwargs)

        dfs.append(pd.DataFrame({
            "gene": _gene_label(gene, adata.var_names),
            "group": "all" if glabel is None else glabel,
            "kind": "observed",
            "count": y,
            "density": np.nan,
        }))
        dfs.append(pd.DataFrame({
            "gene": _gene_label(gene, adata.var_names),
            "group": "all" if glabel is None else glabel,
            "kind": "fitted_pmf",
            "count": k,
            "density": pmf,
        }))
        dfs.append(pd.DataFrame({
            "gene": _gene_label(gene, adata.var_names),
            "group": "all" if glabel is None else glabel,
            "kind": "moment_matched_pmf",
            "count": k,
            "density": mm_pmf,
        }))

    return pd.concat(dfs, ignore_index=True)


def plot_qq_residuals(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...],
    random_state=None,
    ax=None,
    bins: int = 40,
    cmap: str = "RdBu_r",
    origin: str = "upper",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    qq_scatter_kwargs: dict | None = None,
    qq_refline_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
    hist_refline_kwargs: dict | None = None,
    spatial_scatter_kwargs: dict | None = None,
    spatial_cbar_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """QQ-plot and histogram of Dunn-Smyth (randomized quantile) residuals
    against the standard normal distribution -- one row of 3 panels
    (QQ-plot, histogram, spatial scatter) per gene in ``gene``.

    Parameters
    ----------
    gene
        Required (an int, str, or list/tuple); ``None`` is not supported.
        A list/tuple renders one row per gene, shape ``(n_genes, 3)``.
    random_state
        Seeds the randomization draw. ``None`` draws fresh randomness
        each call.
    cmap, origin, size, spatial_scatter_kwargs, spatial_cbar_kwargs
        Forwarded to the spatial panel only (see
        :func:`stipple.plotting.plot_spatial_residuals`).
    qq_scatter_kwargs
        Forwarded to ``ax.scatter`` (QQ points, default alpha=0.6).
    qq_refline_kwargs, hist_refline_kwargs
        Forwarded to ``ax.plot`` (the y=x and N(0,1) reference curves).
    hist_kwargs
        Forwarded to ``ax.hist``.
    """
    import matplotlib.pyplot as plt

    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")
    if gene is None:
        raise ValueError("gene must be provided (an int, str, or list/tuple of them) -- pooled gene=None QQ-plots are not supported.")

    ax_given = ax is not None
    genes = _normalize_genes(gene, adata)
    gene_rows = genes
    n_rows = len(gene_rows)
    n_cols = 3
    coords = _get_coords(adata)

    if ax is None:
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize or (6 * n_cols, 6 * n_rows), dpi=300, squeeze=False
        )
        ret = axes
    else:
        axes = ax
        ret = ax

    axes_rows = np.atleast_2d(axes)
    if axes_rows.shape[0] < n_rows or axes_rows.shape[1] < n_cols:
        raise ValueError(f"ax must provide at least {n_rows} row(s) x {n_cols} columns for {n_rows} gene(s).")

    dfs = [
        _qq_residuals_row(
            adata, coords, g, random_state, row[:n_cols], bins, cmap, origin, size, title, xlabel, ylabel,
            font_sizes, tick_kwargs, qq_scatter_kwargs, qq_refline_kwargs, hist_kwargs, hist_refline_kwargs,
            spatial_scatter_kwargs, spatial_cbar_kwargs,
        )
        for row, g in zip(axes_rows, gene_rows)
    ]
    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _qq_residuals_row(
    adata, coords, gene, random_state, row_axes, bins, cmap, origin, size, title, xlabel, ylabel,
    font_sizes, tick_kwargs, qq_scatter_kwargs, qq_refline_kwargs, hist_kwargs, hist_refline_kwargs,
    spatial_scatter_kwargs, spatial_cbar_kwargs,
):
    import pandas as pd
    from scipy.stats import norm

    residual_type = "dunn_smyth"

    r = np.asarray(_residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state), dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    subtitle = f" ({_gene_label(gene, adata.var_names, none_label='all genes pooled')})"

    # --- QQ-plot: sorted residuals vs. theoretical N(0,1) quantiles ---
    ax_qq = row_axes[0]
    sorted_r = np.sort(r)
    theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    qq_kw = _merge_scatter_kwargs({"alpha": 0.6}, size, qq_scatter_kwargs)
    ax_qq.scatter(theoretical, sorted_r, **qq_kw)
    lo = float(min(theoretical[0], sorted_r[0])) if n else -1.0
    hi = float(max(theoretical[-1], sorted_r[-1])) if n else 1.0
    qrl_kw = {"color": "k", "ls": "--", "lw": 1, "alpha": 0.7}
    qrl_kw.update(qq_refline_kwargs or {})
    ax_qq.plot([lo, hi], [lo, hi], **qrl_kw)
    _apply_label(ax_qq, "xlabel", xlabel, "Theoretical N(0, 1) quantiles", font_sizes)
    _apply_label(ax_qq, "ylabel", ylabel, "Sample quantiles", font_sizes)
    ax_qq.set_box_aspect(1)
    _apply_label(ax_qq, "title", title, f"{residual_type.capitalize()} residual QQ-plot" + subtitle, font_sizes)
    _apply_tick_params(ax_qq, font_sizes, tick_kwargs)

    # --- Histogram with N(0,1) density overlay ---
    ax_hist = row_axes[1]
    h_kw = {"bins": bins, "density": True, "alpha": 0.7, "color": "#3A7CA5", "edgecolor": "white"}
    h_kw.update(hist_kwargs or {})
    ax_hist.hist(r, **h_kw)
    xlo, xhi = ax_hist.get_xlim()
    grid = np.linspace(xlo, xhi, 200)
    hrl_kw = {"color": "k", "ls": "--", "lw": 1.5, "label": "N(0, 1)"}
    hrl_kw.update(hist_refline_kwargs or {})
    ax_hist.plot(grid, norm.pdf(grid), **hrl_kw)
    _apply_label(ax_hist, "xlabel", xlabel, f"{residual_type} residual", font_sizes)
    _apply_label(ax_hist, "ylabel", ylabel, "Density", font_sizes)
    ax_hist.set_box_aspect(1)
    ax_hist.legend(fontsize=8)
    _apply_label(ax_hist, "title", title, "Residual histogram" + subtitle, font_sizes)
    _apply_tick_params(ax_hist, font_sizes, tick_kwargs)

    # sorted_r/theoretical fully determine the QQ-plot; the histogram is
    # recomputable from the same residuals (plus its own `bins` argument),
    # so one column of unsorted residuals is enough source data for both.
    dfs = [pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names, none_label="all genes pooled"),
        "type": residual_type,
        "residual": r,
    })]

    # --- Spatial panel: only when a specific gene (not the pooled
    # gene=None aggregate) was selected -- see docstring. ---
    if coords is not None:
        df_spatial = _spatial_residuals_panel(
            adata, coords, gene, residual_type, random_state, row_axes[2], cmap, origin, size,
            title, xlabel, ylabel, font_sizes, tick_kwargs, spatial_scatter_kwargs, spatial_cbar_kwargs,
        )
        dfs.append(df_spatial)

    return pd.concat(dfs, ignore_index=True)


def plot_objective_trace(
    adata,
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    converged_marker_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Penalized MAP log-joint value vs. Adam iteration, from
    ``adata.uns["stipple"]["objective_history"]`` (see
    :meth:`stipple.model._StippleBasisModel.log_joint`).

    A monotonically flattening curve indicates the fit is converging; a
    curve that's still moving steeply at the last iteration indicates
    ``n_iterations`` was too low. Like ``spammonod``'s own L-BFGS/Laplace
    fits, this is a MAP-only optimization trace, not a variational ELBO --
    there is no latent variable being integrated over in the basis-function
    model (see ``adr/0005-replace-gp-spatial-field-with-radial-basis-capture-model.md``).

    A dashed vertical line marks the last iteration if
    ``adata.uns["stipple"]["converged"]`` is True.

    line_kwargs -> ax.plot (the objective trace itself).
    converged_marker_kwargs -> ax.axvline (the convergence marker).
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    ax_given = ax is not None
    meta = adata.uns["stipple"]
    history = np.asarray(meta["objective_history"], dtype=float)
    iterations = np.arange(history.size)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 6), dpi=300)

    kw = {"color": "tab:blue"}
    kw.update(line_kwargs or {})
    ax.plot(iterations, history, **kw)

    if meta.get("converged") and iterations.size:
        mkw = {"color": "k", "ls": "--", "lw": 1, "alpha": 0.7, "label": "converged"}
        mkw.update(converged_marker_kwargs or {})
        ax.axvline(iterations[-1], **mkw)
        ax.legend(fontsize=8)

    _apply_label(ax, "xlabel", xlabel, "iteration", font_sizes)
    _apply_label(ax, "ylabel", ylabel, "objective (penalized log-joint)", font_sizes)
    ax.set_box_aspect(1)
    _apply_label(ax, "title", title, "Objective convergence trace", font_sizes)
    _apply_tick_params(ax, font_sizes, tick_kwargs)

    if save_source_csv is not None:
        df = pd.DataFrame({"iteration": iterations, "objective": history})
        _write_source_csv(save_source_csv, [df])

    return _finalize(ax, ax_given=ax_given, return_fig=return_fig, save=save, show=show)
