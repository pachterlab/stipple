"""Pre-fit, raw-data-only diagnostics: plot_mean_variance_by_space,
plot_quantile_outlier_score.

Unlike every other module in :mod:`stipple.plotting`, these two only ever
read ``adata.X``/``adata.obsm["spatial"]`` directly -- no ``StippleModel`` fit or
``set_anndata_results`` call is required (or expected) before calling
them; they're meant as an exploratory first look at the raw data.
"""
from __future__ import annotations

import numpy as np

from ._utils import (
    _UNSET, _apply_label, _apply_tick_params, _colorbar, _finalize, _get_coords,
    _merge_scatter_kwargs, _write_source_csv,
)


def _get_raw_counts(adata) -> np.ndarray:
    """adata.X, densified if sparse -- no adata.layers["raw"] fallback
    logic (unlike stipple.plotting._utils._get_counts): these functions are
    explicitly pre-fit, so there is no snapshot to prefer over adata.X."""
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def plot_mean_variance_by_space(
    adata,
    n_quantiles: int = 4,
    ax=None,
    cmap: str = "viridis",
    size: float | None = 1.5,
    figsize: tuple[float, float] | None = None,
    title: str | None = _UNSET,
    xlabel: str | None = _UNSET,
    ylabel: str | None = _UNSET,
    font_sizes: dict | None = None,
    tick_kwargs: dict | None = None,
    scatter_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    refline_kwargs: dict | None = None,
    save_source_csv: str | None = None,
    return_fig: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """Per-gene mean-variance relationship, computed within spatial quantile
    bins and faceted two ways: one panel colored by x-coordinate quantile,
    one by y-coordinate quantile.

    scatter_kwargs -> ax.scatter. cbar_kwargs -> plt.colorbar.
    refline_kwargs -> ax.plot (the Poisson var=mean diagonal guide line).
    title/xlabel/ylabel, if given, apply identically to both panels.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    ax_given = ax is not None
    Y = _get_raw_counts(adata)
    coords = _get_coords(adata)
    G = Y.shape[1]

    def _binned_mean_var(coord_col: int):
        edges = np.quantile(coords[:, coord_col], np.linspace(0.0, 1.0, n_quantiles + 1))
        edges[-1] *= 1.0 + np.sign(edges[-1]) * 1e-9 if edges[-1] != 0 else 1e-9
        bin_idx = np.clip(np.digitize(coords[:, coord_col], edges) - 1, 0, n_quantiles - 1)

        means, variances, bins_ = [], [], []
        for b in range(n_quantiles):
            mask = bin_idx == b
            if mask.sum() < 2:
                continue
            for g in range(G):
                y = Y[mask, g]
                means.append(y.mean())
                variances.append(y.var(ddof=1))
                bins_.append(b)
        return np.array(means), np.array(variances), np.array(bins_)

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=figsize or (12, 6), dpi=300)
    axes = np.atleast_1d(ax)

    dfs = []
    for a, coord_col, label in zip(axes, (0, 1), ("x", "y")):
        means, variances, bins_ = _binned_mean_var(coord_col)
        skw = _merge_scatter_kwargs({}, size, scatter_kwargs)
        sc = a.scatter(means, variances, c=bins_, cmap=cmap, **skw)
        finite = np.concatenate([means[means > 0], variances[variances > 0]])
        if finite.size:
            lo, hi = finite.min(), finite.max()
            rkw = {"color": "k", "ls": "--", "lw": 1, "alpha": 0.5, "label": "Poisson (var=mean)"}
            rkw.update(refline_kwargs or {})
            a.plot([lo, hi], [lo, hi], **rkw)
        a.set_xscale("log")
        a.set_yscale("log")
        _apply_label(a, "xlabel", xlabel, "mean count", font_sizes)
        _apply_label(a, "ylabel", ylabel, "variance", font_sizes)
        a.set_box_aspect(1)
        _apply_label(a, "title", title, f"Mean-variance, colored by {label}-quantile", font_sizes)
        a.legend(fontsize=8)
        _colorbar(sc, a, f"{label} quantile bin", cbar_kwargs, font_sizes, tick_kwargs)
        _apply_tick_params(a, font_sizes, tick_kwargs)
        dfs.append(pd.DataFrame({
            "facet": label,
            "quantile_bin": bins_,
            "mean": means,
            "variance": variances,
        }))

    if save_source_csv is not None:
        _write_source_csv(save_source_csv, dfs)

    return _finalize(ax, ax_given=ax_given, return_fig=return_fig, save=save, show=show)


def plot_quantile_outlier_score(
    adata,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
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
    """Exploratory, pre-fit spatial score for how many genes are jointly
    unusually high or low at each location: each gene contributes +1 at a
    location above its own ``upper_quantile``, -1 below its own
    ``lower_quantile``, 0 otherwise, averaged across genes into one score
    ``s_i`` per location.

    Parameters
    ----------
    lower_quantile, upper_quantile
        Per-gene, across-location percentile cutoffs, ``0 <=
        lower_quantile < upper_quantile <= 1``.
    scatter_kwargs
        Forwarded to ``ax.scatter`` (default vmin/vmax span ``s_i``
        symmetrically about 0).
    cbar_kwargs
        Forwarded to ``plt.colorbar``.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if origin not in ("lower", "upper"):
        raise ValueError(f"origin must be 'lower' or 'upper', got {origin!r}")
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError(
            "lower_quantile must be < upper_quantile, both in [0, 1]; "
            f"got lower_quantile={lower_quantile}, upper_quantile={upper_quantile}"
        )

    ax_given = ax is not None
    Y = _get_raw_counts(adata)
    coords = _get_coords(adata)

    q_lo = np.quantile(Y, lower_quantile, axis=0)   # (G,)
    q_hi = np.quantile(Y, upper_quantile, axis=0)   # (G,)
    signed_indicator = (Y > q_hi[None, :]).astype(float) - (Y < q_lo[None, :]).astype(float)
    s_i = signed_indicator.mean(axis=1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (6, 6), dpi=300)

    vmax = float(np.max(np.abs(s_i))) if s_i.size else 1.0
    kw = _merge_scatter_kwargs({"vmin": -vmax, "vmax": vmax}, size, scatter_kwargs)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=s_i, cmap=cmap, **kw)
    _colorbar(sc, ax, "quantile outlier score $s_i$", cbar_kwargs, font_sizes, tick_kwargs)
    _apply_label(ax, "xlabel", xlabel, "x", font_sizes)
    _apply_label(ax, "ylabel", ylabel, "y", font_sizes)
    ax.set_aspect("equal", adjustable="box")   # render x/y at true scale
    if origin == "upper":
        ax.invert_yaxis()
    _apply_label(
        ax, "title", title,
        f"Quantile outlier score (pre-fit, {int(lower_quantile*100)}/{int(upper_quantile*100)} pct)",
        font_sizes,
    )
    _apply_tick_params(ax, font_sizes, tick_kwargs)

    if save_source_csv is not None:
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "score": s_i})
        _write_source_csv(save_source_csv, [df])

    return _finalize(ax, ax_given=ax_given, return_fig=return_fig, save=save, show=show)
