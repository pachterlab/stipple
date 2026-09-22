"""Spatially-mapped diagnostics: plot_capture_rate_surface,
plot_spatial_residuals, plot_residual_variogram, plot_fitted_vs_raw_surface,
plot_fitted_parameter_surface.

Every function here requires ``model.set_anndata_results(adata)`` to have
been run first. See :mod:`stipple.plotting` for the parameter conventions
shared across the whole package.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from ._utils import (
    _UNSET, _apply_label, _apply_tick_params, _colorbar, _finalize, _fitted_mean_variance,
    _gene_label, _get_coords, _get_counts, _grid_dims, _group_idx, _merge_scatter_kwargs,
    _naive_residuals, _normalize_genes, _residuals, _select_genes, _write_source_csv,
)


def plot_capture_rate_surface(
    adata,
    ax=None,
    cmap: str = "viridis",
    origin: str = "upper",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Spatial scatter of the fitted capture rate ``p_hat`` at each location.

    Reads ``adata.obs["stipple_p_hat"]`` (written by
    ``model.write_obs(adata)``/``set_anndata_results``).

    scatter_kwargs -> ax.scatter. cbar_kwargs -> plt.colorbar.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")

    ax_given = ax is not None
    coords = _get_coords(adata)
    p_hat = np.asarray(adata.obs["stipple_p_hat"])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 6), dpi=300)

    kw = _merge_scatter_kwargs({}, size, scatter_kwargs)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=p_hat, cmap=cmap, **kw)
    _colorbar(sc, ax, "p_hat", cbar_kwargs, font_sizes, tick_kwargs)
    _apply_label(ax, "xlabel", xlabel, "x", font_sizes)
    _apply_label(ax, "ylabel", ylabel, "y", font_sizes)
    ax.set_aspect("equal", adjustable="box")   # render x/y at true scale
    if origin == "upper":
        ax.invert_yaxis()
    _apply_label(ax, "title", title, "Fitted capture rate surface", font_sizes)
    _apply_tick_params(ax, font_sizes, tick_kwargs)

    if save_source_csv is not None:
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "p_hat": p_hat})
        _write_source_csv(save_source_csv, [df])

    return _finalize(ax, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def plot_spatial_residuals(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    residual_type: str = "pearson",
    random_state=None,
    compare_naive: bool = False,
    ax=None,
    cmap: str = "RdBu_r",
    origin: str = "upper",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Scatter of per-location residuals at their spatial coordinates.

    Parameters
    ----------
    residual_type
        ``"pearson"`` (default), ``"deviance"``, or ``"dunn_smyth"``.
    random_state
        Seeds ``"dunn_smyth"``'s randomization draw; ignored otherwise.
    compare_naive
        If True, renders 2 panels per gene: the fitted model's residuals
        (left) and residuals against a naive, spatially-unaware
        moment-matched model (right), sharing one color scale. Multi-gene
        selections then return shape ``(n_genes, 2)``; a single gene
        returns ``(1, 2)``, not a bare Axes.
    scatter_kwargs
        Forwarded to ``ax.scatter`` (default vmin/vmax span the residuals
        symmetrically about 0).
    cbar_kwargs
        Forwarded to ``plt.colorbar``.
    """
    import matplotlib.pyplot as plt

    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")

    ax_given = ax is not None
    coords = _get_coords(adata)
    genes = _normalize_genes(gene, adata)

    if compare_naive:
        return _plot_spatial_residuals_compare(
            adata, coords, genes, residual_type, random_state, ax, cmap, origin, size, figsize,
            title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
            save_source_csv, return_fig, save, show,
        )

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
            _spatial_residuals_panel(
                adata, coords, g, residual_type, random_state, a, cmap, origin, size,
                title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
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
        df = _spatial_residuals_panel(
            adata, coords, single_gene, residual_type, random_state, ax, cmap, origin, size,
            title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
        )
        if save_source_csv is not None:
            _write_source_csv(save_source_csv, [df])
        ret = ax

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _spatial_residuals_panel(
    adata, coords, gene, residual_type, random_state, ax, cmap, origin, size,
    title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
):
    import pandas as pd

    resid = _residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state)
    vmax = float(np.max(np.abs(resid))) if resid.size else 1.0
    kw = _merge_scatter_kwargs({"vmin": -vmax, "vmax": vmax}, size, scatter_kwargs)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=resid, cmap=cmap, **kw)
    _colorbar(sc, ax, f"{residual_type} residual", cbar_kwargs, font_sizes, tick_kwargs)
    _apply_label(ax, "xlabel", xlabel, "x", font_sizes)
    _apply_label(ax, "ylabel", ylabel, "y", font_sizes)
    ax.set_aspect("equal", adjustable="box")   # render x/y at true scale
    if origin == "upper":
        ax.invert_yaxis()
    _apply_label(ax, "title", title, f"Spatial residuals ({_gene_label(gene, adata.var_names)})", font_sizes)
    _apply_tick_params(ax, font_sizes, tick_kwargs)

    return pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "type": residual_type,
        "residual": resid,
    })


def _plot_spatial_residuals_compare(
    adata, coords, genes, residual_type, random_state, ax, cmap, origin, size, figsize,
    title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
    save_source_csv, return_fig, save, show,
):
    import matplotlib.pyplot as plt

    ax_given = ax is not None
    gene_rows = genes if genes is not None else [None]
    n_rows = len(gene_rows)

    if ax is None:
        _, axes = plt.subplots(n_rows, 2, figsize=figsize or (12, 5 * n_rows), dpi=300, squeeze=False)
        ret = axes
    else:
        axes = ax
        ret = ax

    axes_rows = np.atleast_2d(axes)
    if axes_rows.shape[0] < n_rows or axes_rows.shape[1] < 2:
        raise ValueError(f"ax must provide at least {n_rows} row(s) x 2 columns for {n_rows} gene(s).")

    dfs = [
        _spatial_residuals_compare_row(
            adata, coords, g, residual_type, random_state, row[:2], cmap, origin, size,
            title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
        )
        for row, g in zip(axes_rows, gene_rows)
    ]
    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _spatial_residuals_compare_row(
    adata, coords, gene, residual_type, random_state, row_axes, cmap, origin, size,
    title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
):
    import pandas as pd

    fitted_resid = _residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state)
    naive_resid = _naive_residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state)
    both = np.concatenate([fitted_resid, naive_resid])
    vmax = float(np.max(np.abs(both))) if both.size else 1.0
    suffix = f" ({_gene_label(gene, adata.var_names)})"

    for a, label, resid in zip(row_axes, ("Fitted model", "Naive (IID) model"), (fitted_resid, naive_resid)):
        kw = _merge_scatter_kwargs({"vmin": -vmax, "vmax": vmax}, size, scatter_kwargs)
        sc = a.scatter(coords[:, 0], coords[:, 1], c=resid, cmap=cmap, **kw)
        _colorbar(sc, a, f"{residual_type} residual", cbar_kwargs, font_sizes, tick_kwargs)
        _apply_label(a, "xlabel", xlabel, "x", font_sizes)
        _apply_label(a, "ylabel", ylabel, "y", font_sizes)
        a.set_aspect("equal", adjustable="box")   # render x/y at true scale
        if origin == "upper":
            a.invert_yaxis()
        _apply_label(a, "title", title, f"{label} spatial residuals" + suffix, font_sizes)
        _apply_tick_params(a, font_sizes, tick_kwargs)

    return pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names),
        "x": np.tile(coords[:, 0], 2),
        "y": np.tile(coords[:, 1], 2),
        "type": residual_type,
        "model": ["fitted"] * len(fitted_resid) + ["naive"] * len(naive_resid),
        "residual": both,
    })


_DIFF_PANEL_TITLES = {
    "raw": "Raw - fitted",
    "pearson": "Pearson residuals",
    "deviance": "Deviance residuals",
    "dunn_smyth": "Dunn-Smyth residuals",
}


def plot_fitted_vs_raw_surface(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    residual_type: str = "raw",
    ax=None,
    cmap: str = "viridis",
    origin: str = "upper",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Three-panel spatial surface -- raw counts, fitted expected counts,
    and a third panel showing either their raw difference or a residual --
    one row per gene selected.

    Parameters
    ----------
    residual_type
        Controls the third panel: ``"raw"`` (default, unstandardized
        ``raw - fitted``), ``"pearson"``, ``"deviance"``, or
        ``"dunn_smyth"``.
    scatter_kwargs, cbar_kwargs
        Forwarded to ``ax.scatter``/``plt.colorbar``, shared across all 3
        panels in a row.
    """
    import matplotlib.pyplot as plt

    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")
    if residual_type not in _DIFF_PANEL_TITLES:
        raise ValueError(f"residual_type must be one of {tuple(_DIFF_PANEL_TITLES)}, got {residual_type!r}")

    ax_given = ax is not None
    coords = _get_coords(adata)
    genes = _normalize_genes(gene, adata)
    gene_rows = genes if genes is not None else [None]
    n_rows = len(gene_rows)

    if ax is None:
        _, axes = plt.subplots(n_rows, 3, figsize=figsize or (18, 6 * n_rows), dpi=300, squeeze=False)
        ret = axes
    else:
        axes = ax
        ret = ax

    axes_rows = np.atleast_2d(axes)
    if axes_rows.shape[0] < n_rows or axes_rows.shape[1] < 3:
        raise ValueError(f"ax must provide at least {n_rows} row(s) x 3 columns for {n_rows} gene(s).")

    dfs = [
        _fitted_vs_raw_row(
            adata, coords, g, residual_type, row[:3], cmap, origin, size, title, xlabel, ylabel,
            font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
        )
        for row, g in zip(axes_rows, gene_rows)
    ]
    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _fitted_vs_raw_row(
    adata, coords, gene, residual_type, row_axes, cmap, origin, size, title, xlabel, ylabel,
    font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
):
    import pandas as pd

    raw = _select_genes(_get_counts(adata), gene)
    fitted, _ = _fitted_mean_variance(adata, gene=gene)
    if residual_type == "raw":
        diff = raw - fitted
    else:
        diff = _residuals(adata, gene=gene, residual_type=residual_type)

    shared = np.concatenate([raw, fitted])
    shared_vmin = float(shared.min()) if shared.size else 0.0
    shared_vmax = float(shared.max()) if shared.size else 1.0

    panels = [
        ("Raw counts", raw, cmap, shared_vmin, shared_vmax),
        ("Fitted expected counts", fitted, cmap, shared_vmin, shared_vmax),
        (_DIFF_PANEL_TITLES[residual_type], diff, "RdBu_r", None, None),
    ]
    suffix = f" ({_gene_label(gene, adata.var_names)})"
    for a, (panel_title, values, this_cmap, vmin, vmax) in zip(row_axes, panels):
        if this_cmap == "RdBu_r":
            dmax = float(np.max(np.abs(values))) if values.size else 1.0
            defaults = {"vmin": -dmax, "vmax": dmax}
        else:
            defaults = {"vmin": vmin, "vmax": vmax}
        vkw = _merge_scatter_kwargs(defaults, size, scatter_kwargs)
        sc = a.scatter(coords[:, 0], coords[:, 1], c=values, cmap=this_cmap, **vkw)
        _colorbar(sc, a, panel_title, cbar_kwargs, font_sizes, tick_kwargs)
        _apply_label(a, "xlabel", xlabel, "x", font_sizes)
        _apply_label(a, "ylabel", ylabel, "y", font_sizes)
        a.set_aspect("equal", adjustable="box")   # render x/y at true scale
        if origin == "upper":
            a.invert_yaxis()
        _apply_label(a, "title", title, panel_title + suffix, font_sizes)
        _apply_tick_params(a, font_sizes, tick_kwargs)

    return pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "raw": raw,
        "fitted": fitted,
        "diff": diff,
    })


def plot_residual_variogram(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    residual_type: str = "dunn_smyth",
    random_state=None,
    n_bins: int = 15,
    compare_naive: bool = False,
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
    naive_line_kwargs: dict | None = None,
    halflag_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Empirical semivariogram of residuals vs. pairwise distance. A
    dashed vertical reference line marks half the max pairwise distance;
    only trust the curve's shape up to that line.

    Parameters
    ----------
    random_state
        Seeds ``residual_type="dunn_smyth"``'s randomization draw.
    compare_naive
        If True, overlays a second line: the residual variogram of a
        naive, spatially-unaware moment-matched model.
    line_kwargs
        Forwarded to ``ax.plot`` for the fitted-model line.
    naive_line_kwargs
        Forwarded to ``ax.plot`` for the ``compare_naive`` line.
    halflag_kwargs
        Forwarded to the half-max-distance ``ax.axvline``.
    """
    import matplotlib.pyplot as plt

    ax_given = ax is not None
    coords = _get_coords(adata)
    dists = pdist(coords)
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
            _residual_variogram_panel(
                adata, dists, g, residual_type, random_state, n_bins, compare_naive, a, title, xlabel, ylabel,
                font_sizes, tick_kwargs, line_kwargs, naive_line_kwargs, halflag_kwargs,
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
        df = _residual_variogram_panel(
            adata, dists, single_gene, residual_type, random_state, n_bins, compare_naive, ax, title, xlabel, ylabel,
            font_sizes, tick_kwargs, line_kwargs, naive_line_kwargs, halflag_kwargs,
        )
        if save_source_csv is not None:
            _write_source_csv(save_source_csv, [df])
        ret = ax

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _semivariance(resid: np.ndarray, dists: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    sq_diffs = pdist(resid[:, None], metric="sqeuclidean")
    semivariance = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if np.any(mask):
            semivariance[b] = 0.5 * sq_diffs[mask].mean()
    return semivariance


def _residual_variogram_panel(
    adata, dists, gene, residual_type, random_state, n_bins, compare_naive, ax, title, xlabel, ylabel,
    font_sizes, tick_kwargs, line_kwargs, naive_line_kwargs, halflag_kwargs,
):
    import pandas as pd

    max_dist = float(dists.max())
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    bin_edges[-1] *= 1.0 + 1e-9  # include the max-distance pair
    bin_idx = np.clip(np.digitize(dists, bin_edges) - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    resid = _residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state)
    semivariance = _semivariance(resid, dists, bin_idx, n_bins)

    kw = {"marker": "o", "label": "fitted"}
    kw.update(line_kwargs or {})
    ax.plot(bin_centers, semivariance, **kw)

    dfs = [pd.DataFrame({
        "gene": _gene_label(gene, adata.var_names),
        "model": "fitted",
        "distance": bin_centers,
        "semivariance": semivariance,
    })]

    if compare_naive:
        naive_resid = _naive_residuals(adata, gene=gene, residual_type=residual_type, random_state=random_state)
        naive_semivariance = _semivariance(naive_resid, dists, bin_idx, n_bins)
        nkw = {"color": "#D9534F", "ls": "-.", "marker": "s", "label": "naive (IID)"}
        nkw.update(naive_line_kwargs or {})
        ax.plot(bin_centers, naive_semivariance, **nkw)
        dfs.append(pd.DataFrame({
            "gene": _gene_label(gene, adata.var_names),
            "model": "naive",
            "distance": bin_centers,
            "semivariance": naive_semivariance,
        }))

    hkw = {"color": "k", "ls": "--", "lw": 1, "alpha": 0.7}
    hkw.update(halflag_kwargs or {})
    ax.axvline(max_dist / 2.0, **hkw)
    _apply_label(ax, "xlabel", xlabel, "distance", font_sizes)
    _apply_label(ax, "ylabel", ylabel, "semivariance", font_sizes)
    ax.set_box_aspect(1)
    _apply_label(ax, "title", title, f"Residual variogram ({_gene_label(gene, adata.var_names)})", font_sizes)
    ax.set_xlim(left=0.0)   # origin always in view, even though bin_centers > 0
    ax.set_ylim(bottom=0.0)   # y range always includes 0, whatever the top autoscales to
    _apply_tick_params(ax, font_sizes, tick_kwargs)
    if compare_naive:
        ax.legend(fontsize=8)

    return pd.concat(dfs, ignore_index=True)


def plot_fitted_parameter_surface(
    adata,
    gene: int | str | list[int | str] | tuple[int | str, ...],
    ax=None,
    cmap: str = "viridis",
    origin: str = "upper",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Spatial surface of fitted per-gene, per-group parameters -- every
    location colored by its own group's value.

    Parameters
    ----------
    gene
        Required (``None`` has no clean per-group interpretation here).
        A list/tuple renders one row per gene, shape ``(n_genes,
        n_panels)``. Negative-binomial fits get 2 panels per gene
        (``mu_gk``, ``phi_gk``); Poisson fits get 1 (``mu_gk`` only).
    scatter_kwargs, cbar_kwargs
        Forwarded to ``ax.scatter``/``plt.colorbar``, shared across every
        panel.
    """
    import matplotlib.pyplot as plt

    if gene is None:
        raise ValueError(
            "gene is required for plot_fitted_parameter_surface -- pooling mu_gk/phi_gk "
            "across genes has no clean spatial interpretation."
        )
    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")

    ax_given = ax is not None
    coords = _get_coords(adata)
    group_idx = _group_idx(adata)
    mu_gk = np.asarray(adata.varm["stipple_mu_gk"])   # (G, K)
    phi_gk = np.asarray(adata.varm["stipple_phi_gk"]) if "stipple_phi_gk" in adata.varm else None

    genes = _normalize_genes(gene, adata)
    n_rows = len(genes)
    n_cols = 1 if phi_gk is None else 2

    if ax is None:
        _, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (6 * n_cols, 6 * n_rows), dpi=300, squeeze=False)
        ret = axes
    else:
        axes = np.atleast_2d(ax)
        ret = ax
    if axes.shape[0] < n_rows or axes.shape[1] < n_cols:
        raise ValueError(f"ax must provide at least {n_rows} row(s) x {n_cols} column(s) for {n_rows} gene(s).")

    dfs = [
        _parameter_surface_row(
            adata, coords, group_idx, g, row[:n_cols], mu_gk, phi_gk, cmap, origin, size,
            title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
        )
        for row, g in zip(axes, genes)
    ]
    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ret, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def _parameter_surface_row(
    adata, coords, group_idx, gene, row_axes, mu_gk, phi_gk, cmap, origin, size,
    title, xlabel, ylabel, font_sizes, tick_kwargs, scatter_kwargs, cbar_kwargs,
):
    import pandas as pd

    gene_name = str(adata.var_names[gene])
    panels = [("mu_gk", mu_gk)] if phi_gk is None else [("mu_gk", mu_gk), ("phi_gk", phi_gk)]

    dfs = []
    for a, (name, values) in zip(row_axes, panels):
        values_gk = values[gene, :]         # (K,)
        per_location = values_gk[group_idx]  # (n,) -- broadcast each location's group value
        kw = _merge_scatter_kwargs({}, size, scatter_kwargs)
        sc = a.scatter(coords[:, 0], coords[:, 1], c=per_location, cmap=cmap, **kw)
        _colorbar(sc, a, name, cbar_kwargs, font_sizes, tick_kwargs)
        _apply_label(a, "xlabel", xlabel, "x", font_sizes)
        _apply_label(a, "ylabel", ylabel, "y", font_sizes)
        a.set_aspect("equal", adjustable="box")   # render x/y at true scale
        if origin == "upper":
            a.invert_yaxis()
        _apply_label(a, "title", title, f"Fitted {name} surface ({gene_name})", font_sizes)
        _apply_tick_params(a, font_sizes, tick_kwargs)
        dfs.append(pd.DataFrame({
            "gene": gene_name, "x": coords[:, 0], "y": coords[:, 1], "parameter": name, "value": per_location,
        }))

    return pd.concat(dfs, ignore_index=True)
