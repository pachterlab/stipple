"""Extract and validate model inputs from an AnnData object (see
:func:`preprocess`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData

from . import grid, priors

DTYPE = torch.float64
DENSE_WARN_ENTRIES: int = 5_000_000  # warn before densifying a count matrix larger than this


@dataclass
class PreprocessedData:
    """Result of :func:`preprocess`."""

    Y_gn: torch.Tensor  # (G, n), float dtype, integer-valued
    coords_raw: np.ndarray  # (n, 2)
    coords_transformed: np.ndarray  # (n, 2): int (u, v) for hex; raw Cartesian for square
    inverse_transform: Callable[[np.ndarray], np.ndarray]
    group_idx: torch.Tensor  # (n,) long, in [0, K)
    group_labels: list[str]  # length K
    gene_names: list[str]  # length G
    grid_info: grid.GridInfo
    mu_hat: torch.Tensor  # (G,)
    phi_hat: torch.Tensor  # (G,)
    n: int
    G: int
    K: int


def _validate_counts(X: np.ndarray | sp.spmatrix) -> None:
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if values.size == 0:
        return
    if np.any(values < 0):
        raise ValueError("adata.X contains negative values; expected non-negative integer counts")
    if not np.allclose(values, np.round(values)):
        raise ValueError("adata.X contains non-integer values; expected integer counts")


def _extract_counts(adata: AnnData) -> np.ndarray | sp.spmatrix:
    """Return ``adata.X`` (dense or sparse, unmodified), validated as
    non-negative and integer-valued."""
    X = adata.X
    _validate_counts(X)
    return X


def _extract_coords(adata: AnnData) -> np.ndarray:
    """``adata.obsm['spatial']``, validated shape ``(n, 2)``."""
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"adata.obsm['spatial'] must have shape (n, 2), got {coords.shape}")
    return coords


def _encode_groups(adata: AnnData, group_key: str) -> tuple[np.ndarray, list[str]]:
    """Integer-encode ``adata.obs[group_key]`` via ``pd.factorize(sort=True)``.

    Raises
    ------
    ValueError
        If ``group_key`` is missing from ``adata.obs``, or the column
        contains missing values.
    """
    if group_key not in adata.obs.columns:
        raise ValueError(f"group_key {group_key!r} not found in adata.obs")
    column = adata.obs[group_key]
    if column.isna().any():
        raise ValueError(f"adata.obs[{group_key!r}] contains missing values")
    codes, uniques = pd.factorize(column, sort=True)
    group_idx = codes.astype(np.int64)
    group_labels = [str(u) for u in uniques]
    return group_idx, group_labels


def preprocess(adata: AnnData, group_key: str) -> PreprocessedData:
    """Validate and extract all preprocessed model inputs from ``adata``.

    Extracts the count matrix (dense or sparse), spatial coordinates,
    and group labels; validates every input; computes plug-in gene-level
    estimates (:func:`stipple.model.priors.estimate_plugins`); detects and
    transforms the spatial grid (:func:`stipple.model.grid.check_grid`,
    :func:`stipple.model.grid.to_grid_coordinates`).

    Parameters
    ----------
    adata
        AnnData with ``adata.X`` (n x G non-negative integer counts,
        dense or sparse), ``adata.obsm['spatial']`` (n x 2), and
        ``adata.obs[group_key]`` (categorical, no missing values).
    group_key
        Column of ``adata.obs`` giving each location's group label.

    Returns
    -------
    PreprocessedData

    Raises
    ------
    ValueError
        If ``group_key`` is not in ``adata.obs``; if any group label is
        missing; if counts contain negative or non-integer values; if
        ``adata.obsm['spatial']`` is not shape ``(n, 2)`` or its length
        does not match ``adata.n_obs``; or if
        :func:`stipple.model.grid.check_grid` cannot recognize the grid.
    """
    X = _extract_counts(adata)
    coords_raw = _extract_coords(adata)
    group_idx_np, group_labels = _encode_groups(adata, group_key)

    n, G = X.shape
    if coords_raw.shape[0] != n:
        raise ValueError(
            f"adata.X has {n} locations but adata.obsm['spatial'] has {coords_raw.shape[0]}"
        )
    K = len(group_labels)

    plugins = priors.estimate_plugins(X)

    grid_info = grid.check_grid(coords_raw)
    coords_transformed, inverse_transform = grid.to_grid_coordinates(coords_raw, grid_info)

    if n * G > DENSE_WARN_ENTRIES:
        warnings.warn(
            f"materializing a dense {n}x{G} count matrix ({n * G} entries) for the "
            "likelihood -- this may use significant memory for large gene panels",
            stacklevel=2,
        )
    X_dense = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X, dtype=float)
    Y_gn = torch.tensor(X_dense.T, dtype=DTYPE)  # (G, n)

    gene_names = [str(g) for g in adata.var_names] if adata.var_names is not None else [
        f"gene_{i}" for i in range(G)
    ]

    return PreprocessedData(
        Y_gn=Y_gn,
        coords_raw=coords_raw,
        coords_transformed=coords_transformed,
        inverse_transform=inverse_transform,
        group_idx=torch.as_tensor(group_idx_np, dtype=torch.long),
        group_labels=group_labels,
        gene_names=gene_names,
        grid_info=grid_info,
        mu_hat=torch.as_tensor(plugins.mu_hat, dtype=DTYPE),
        phi_hat=torch.as_tensor(plugins.phi_hat, dtype=DTYPE),
        n=n,
        G=G,
        K=K,
    )
