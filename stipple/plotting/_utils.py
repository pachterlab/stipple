"""Private helpers for stipple.plotting -- not part of the public API."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import xlogy

_UNSET = object()   # sentinel: "not provided, use the auto-generated default" -- distinct
                     # from an explicit None, which means "remove this title/label".

_RESIDUAL_TYPES = ("pearson", "deviance", "dunn_smyth")


def _get_coords(adata) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for spatial coordinates.")
    return np.asarray(adata.obsm["spatial"], dtype=float)


def _get_counts(adata) -> np.ndarray:
    """Raw (n, G) count matrix, densified if sparse. Reads
    ``adata.layers["raw"]`` if present, else falls back to ``adata.X``
    with a warning.
    """
    import scipy.sparse as sp

    if "raw" in adata.layers:
        Y = adata.layers["raw"]
    else:
        warnings.warn(
            "adata.layers['raw'] not found -- falling back to adata.X, which may have "
            "been modified since this model was fit. Re-run model.set_anndata_results(adata) "
            "(or model.write_obs(adata)) to add the layers['raw'] safeguard.",
            UserWarning, stacklevel=3,
        )
        Y = adata.X
    if sp.issparse(Y):
        Y = Y.toarray()
    return np.asarray(Y, dtype=float)


def _select_genes(Y: np.ndarray, gene) -> np.ndarray:
    """Sum Y's (n, G) columns selected by gene into an (n,) array -- None
    sums all columns, an int or a list/tuple of ints sums just those."""
    return Y.sum(axis=1) if gene is None else Y[:, np.atleast_1d(gene)].sum(axis=1)


def _gene_label(gene, var_names=None, none_label: str = "total counts") -> str:
    """Human-readable gene-selection label for plot titles."""
    if gene is None:
        return none_label
    idx = np.atleast_1d(gene)
    if var_names is not None:
        names = [str(var_names[int(i)]) for i in idx]
        return names[0] if idx.size == 1 else f"genes {names}"
    if idx.size == 1:
        return f"gene {int(idx[0])}"
    return f"genes {[int(i) for i in idx]}"


def _normalize_genes(gene, adata) -> list[int] | None:
    """``None`` -> ``None`` (pool every gene). An int/str, or a list/tuple
    mixing either, -> a list of int indices, resolved against
    ``adata.var_names``. A string not found is dropped with a warning;
    raises ``ValueError`` if every requested gene ends up dropped.
    """
    if gene is None:
        return None
    var_names = list(adata.var_names)
    # Not np.atleast_1d(gene): a numpy array unifies mixed int/str lists to
    # a single dtype (every int silently becomes its string repr),
    # corrupting a mixed [0, "gene_name"] list -- plain list()/[gene]
    # preserves each element's own type.
    raw = list(gene) if isinstance(gene, (list, tuple, np.ndarray)) else [gene]
    resolved = []
    for g in raw:
        if isinstance(g, (str, np.str_)):
            name = str(g)
            if name not in var_names:
                warnings.warn(f"gene {name!r} not found in adata.var_names -- omitting.", UserWarning, stacklevel=3)
                continue
            resolved.append(var_names.index(name))
        else:
            resolved.append(int(g))
    if not resolved:
        raise ValueError("No valid genes remained in `gene` after resolving names -- see warnings above.")
    return resolved


def _fitted_mean_variance(adata, gene: int | list[int] | tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-location (mean, variance) of the fitted response, read from
    ``adata.obsm["stipple_fitted_mean"]``/``["stipple_fitted_variance"]`` --
    gene=None sums across all genes, an int/list/tuple sums just that
    subset (same convention as :func:`_select_genes`/:func:`_get_counts`).
    """
    if "stipple_fitted_mean" not in adata.obsm:
        raise RuntimeError(
            "adata.obsm['stipple_fitted_mean'] not found -- run "
            "model.set_anndata_results(adata) (or model.write_obsm(adata)) first."
        )
    mean_ng = np.asarray(adata.obsm["stipple_fitted_mean"])
    var_ng = np.asarray(adata.obsm["stipple_fitted_variance"])
    return _select_genes(mean_ng, gene), _select_genes(var_ng, gene)


def _group_idx(adata) -> np.ndarray:
    """Per-location integer group index in ``[0, K)``, resolved from
    ``adata.obs[adata.uns["stipple"]["group_key"]]`` against
    ``adata.uns["stipple"]["group_labels"]`` -- shared by
    ``_residuals(residual_type="deviance")`` and
    ``diagnostics.plot_count_distribution(by_group=True)``.
    """
    meta = adata.uns["stipple"]
    group_key = meta.get("group_key")
    if group_key is None:
        raise ValueError(
            "adata.uns['stipple']['group_key'] is missing -- re-run "
            "model.set_anndata_results(adata) to populate it."
        )
    label_to_idx = {label: i for i, label in enumerate(meta["group_labels"])}
    return np.array([label_to_idx[g] for g in adata.obs[group_key]])


def _dunn_smyth_residuals(adata, gene: int | list[int] | tuple[int, ...] | None = None, random_state=None) -> np.ndarray:
    """Randomized quantile (Dunn-Smyth) residuals for the gene selection
    given. ``gene=None`` (or a multi-gene selection) sums raw counts and
    fitted (mean, variance) across the selected genes first, then draws
    one residual per location.
    """
    observed = _select_genes(_get_counts(adata), gene)
    mean, var = _fitted_mean_variance(adata, gene=gene)
    likelihood = adata.uns["stipple"]["likelihood"]
    return _residual_stat(observed, mean, var, likelihood, "dunn_smyth", random_state)


def _residual_stat(
    observed: np.ndarray, mean: np.ndarray, var: np.ndarray, likelihood: str, residual_type: str, random_state=None,
) -> np.ndarray:
    """Residual math shared by every (observed, mean, variance) triple.

    Parameters
    ----------
    residual_type
        ``"pearson"``: ``(observed - mean) / sqrt(var)``. ``"deviance"``:
        NegBin dispersion recovered from ``(mean, var)``. ``"dunn_smyth"``:
        randomized quantile residual; ``random_state`` seeds the draw.
    """
    if residual_type == "dunn_smyth":
        from ..utils import _randomized_quantile_residuals

        rng = np.random.default_rng(random_state)
        return _randomized_quantile_residuals(observed, mean, var, likelihood, rng)

    if residual_type == "pearson":
        return (observed - mean) / np.sqrt(np.maximum(var, 1e-12))

    if likelihood == "negbinomial":
        phi = np.clip(mean**2 / np.maximum(var - mean, 1e-6), 1e-3, 1e6)
        deviance = 2.0 * (
            xlogy(observed, observed / mean) - (observed + phi) * np.log((observed + phi) / (mean + phi))
        )
    else:
        deviance = 2.0 * (xlogy(observed, observed / mean) - (observed - mean))
    deviance = np.clip(deviance, 0.0, None)
    return np.sign(observed - mean) * np.sqrt(deviance)


def _residuals(
    adata, gene: int | list[int] | tuple[int, ...] | None = None, residual_type: str = "pearson", random_state=None,
) -> np.ndarray:
    """Per-location residuals against the fitted mean, matching
    ``StippleModel.get_residuals`` exactly for a single gene.
    ``gene=None`` (or a multi-gene selection) sums raw counts and fitted
    (mean, variance) across the selected genes first, then computes one
    residual from that aggregate.
    """
    if residual_type not in _RESIDUAL_TYPES:
        raise ValueError(f"residual_type must be one of {_RESIDUAL_TYPES}, got {residual_type!r}")

    observed = _select_genes(_get_counts(adata), gene)
    mean, var = _fitted_mean_variance(adata, gene=gene)
    return _residual_stat(observed, mean, var, adata.uns["stipple"]["likelihood"], residual_type, random_state)


def _moment_matched_mean_variance(adata, gene: int | list[int] | tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Naive IID/moment-matched (mean, variance): the sample mean and
    variance of the observed counts themselves, broadcast to every
    location -- the same plug-in estimate
    :func:`stipple.plotting.diagnostics.plot_count_distribution`'s
    "moment-matched" line uses, ignoring per-location fitted means and any
    spatial smoothing entirely. This is the "no spatial model at all" null
    that :func:`_naive_residuals` compares the fit against.
    """
    observed = _select_genes(_get_counts(adata), gene)
    n = observed.size
    mm_mean = float(observed.mean()) if n else 0.0
    mm_var = float(observed.var(ddof=1)) if n > 1 else 0.0
    return np.full(n, mm_mean), np.full(n, mm_var)


def _naive_residuals(
    adata, gene: int | list[int] | tuple[int, ...] | None = None, residual_type: str = "pearson", random_state=None,
) -> np.ndarray:
    """Same residual computation as :func:`_residuals`, but against the
    naive moment-matched (IID, spatially-unaware) model from
    :func:`_moment_matched_mean_variance` instead of the fitted spatial
    model -- i.e. every location is scored against the identical pooled
    (mean, variance), so any remaining spatial structure in these
    residuals reflects real spatial signal in the data, not model
    artifacts. Comparing this against :func:`_residuals`'s output is the
    "did the spatial smoothing actually help" check: a fitted-model
    variogram/residual map that looks no flatter than this naive one means
    the smoothing isn't earning its keep.
    """
    if residual_type not in _RESIDUAL_TYPES:
        raise ValueError(f"residual_type must be one of {_RESIDUAL_TYPES}, got {residual_type!r}")

    observed = _select_genes(_get_counts(adata), gene)
    mean, var = _moment_matched_mean_variance(adata, gene=gene)
    return _residual_stat(observed, mean, var, adata.uns["stipple"]["likelihood"], residual_type, random_state)


def _grid_dims(n_panels: int, max_cols: int = 5) -> tuple[int, int]:
    """(n_rows, n_cols) for an n_panels-panel grid, as square as possible,
    capped at max_cols columns."""
    n_cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n_panels)))))
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


_FONT_SIZE_KEYS = {"title": "title", "xlabel": "axis_title", "ylabel": "axis_title"}


def _apply_label(ax, kind: str, value, default: str, font_sizes: dict | None = None) -> None:
    """Resolve and apply a title/xlabel/ylabel: value is _UNSET (use
    `default`), None (remove -- explicitly blank it out, since `ax` may be
    caller-supplied and reused, so a stale title must actually be cleared
    rather than just left untouched), or a literal string (use verbatim).

    kind : "title", "xlabel", or "ylabel".
    font_sizes : optional dict with "title"/"axis_title" keys (see
        font_sizes in the plotting module docstring) -- fontsize=None (the
        default if the relevant key is absent) leaves matplotlib's own
        default untouched.
    """
    resolved = default if value is _UNSET else (value if value is not None else "")
    fontsize = (font_sizes or {}).get(_FONT_SIZE_KEYS[kind])
    getattr(ax, f"set_{kind}")(resolved, fontsize=fontsize)


def _apply_tick_params(ax, font_sizes: dict | None = None, tick_kwargs: dict | None = None) -> None:
    """Apply tick-label fontsize (font_sizes["tick_label"]) and/or tick
    length/width/etc. (tick_kwargs, forwarded to ax.tick_params -- highest
    precedence, can override labelsize too) to both major and minor
    ticks. No-op if neither is given, leaving matplotlib's own defaults
    untouched."""
    kw = {}
    tick_label_size = (font_sizes or {}).get("tick_label")
    if tick_label_size is not None:
        kw["labelsize"] = tick_label_size
    kw.update(tick_kwargs or {})
    if kw:
        ax.tick_params(which="both", **kw)


def _colorbar(mappable, ax, default_label: str, cbar_kwargs: dict | None = None,
              font_sizes: dict | None = None, tick_kwargs: dict | None = None):
    """plt.colorbar(mappable, ...), sized to match ax's own height exactly
    (via mpl_toolkits.axes_grid1's divider, robust to ax.set_aspect("equal")
    -- plt.colorbar(ax=ax)'s default shrink otherwise leaves it taller
    than an equal-aspect panel), with cbar_kwargs merged over
    {"label": default_label} and font_sizes/tick_kwargs applied to the
    colorbar's own label and tick labels the same way as any other axis."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar_kw = {"label": default_label}
    cbar_kw.update(cbar_kwargs or {})
    label = cbar_kw.pop("label")
    cbar = plt.colorbar(mappable, cax=cax, **cbar_kw)
    cbar.set_label(label, fontsize=(font_sizes or {}).get("axis_title"))
    _apply_tick_params(cbar.ax, font_sizes, tick_kwargs)
    return cbar


def _merge_scatter_kwargs(defaults: dict, size, scatter_kwargs: dict | None) -> dict:
    """Merge a scatter call's own internal defaults, the top-level `size`
    param (sets `s`, lowest-to-highest: internal defaults -> size ->
    scatter_kwargs), and a caller-supplied scatter_kwargs dict (highest
    precedence, can override anything including `s`)."""
    kw = dict(defaults)
    if size is not None:
        kw["s"] = size
    kw.update(scatter_kwargs or {})
    return kw


def _write_source_csv(path, dfs) -> None:
    import pandas as pd
    pd.concat(dfs, ignore_index=True).to_csv(path, index=False)


def _finalize(ret, *, ax_given: bool, return_fig: bool, save: str | None, show: bool):
    """Shared return-path handler for return_fig/save/show, called as the
    last step of every public plotting function.

    ret : the value the function would otherwise return (a bare Axes, or
        an array of Axes for a multi-panel grid).
    ax_given : whether the caller supplied their own `ax` -- return_fig=True
        is disallowed in that case (the caller already owns the Figure).
    return_fig : if True, return the owning Figure instead of `ret`.
    save : if given, a file path to save the figure to (fig.savefig(save)).
    show : if True, call plt.show().
    """
    import matplotlib.pyplot as plt

    axes = np.atleast_1d(ret)
    fig = axes.ravel()[0].get_figure()

    if save is not None:
        fig.savefig(save)
    if show:
        plt.show()

    if return_fig:
        if ax_given:
            raise ValueError(
                "return_fig=True is not supported when ax is supplied by the caller -- "
                "the caller already owns the Figure; use ax.get_figure() directly."
            )
        return fig
    return ret
