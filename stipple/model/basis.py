"""Radial-basis-function (RBF) spatial basis construction for the
capture-efficiency field (see :class:`stipple.model._StippleBasisModel`).

Builds a compactly-supported radial basis ``B`` (n, M), normalized so
each row sums to 1, from which ``u = B @ w`` gives the logit-scale
capture field.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

_BASIS_KINDS = ("bisquare", "gaussian", "wendland")


def resolve_resolution_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults for an omitted ``radius_overlap_factor``/
    ``basis_kind`` in a resolution config dict.

    Parameters
    ----------
    cfg
        Dict with required key ``n_knots_target``, optional
        ``radius_overlap_factor``/``basis_kind``.

    Raises
    ------
    ValueError
        If ``cfg`` is missing ``n_knots_target``.
    """
    if "n_knots_target" not in cfg:
        raise ValueError("a resolution config dict must include 'n_knots_target'")
    return {
        "n_knots_target": cfg["n_knots_target"],
        "radius_overlap_factor": cfg.get("radius_overlap_factor", 1.5),
        "basis_kind": cfg.get("basis_kind", "bisquare"),
    }


def _basis_kernel(D: np.ndarray, radius: np.ndarray, kind: str, **kind_kwargs) -> np.ndarray:
    """Dispatch on `kind` to evaluate a compactly-supported radial basis
    kernel given precomputed pairwise distances `D` (n, M) and per-knot
    `radius` (M,). Every kind is exactly 0 for d >= radius.

    Parameters
    ----------
    kind:
        `"bisquare"`: Tukey's biweight, `(1 - (d/r)^2)^2` for `d < r` --
            C^1 at the boundary (value and derivative both -> 0 as d -> r).
        `"gaussian"`: RBF `exp(-0.5*(d/sigma)^2)` with
            `sigma = radius / gaussian_sigma_factor` (default 3.0),
            hard-truncated at `d < radius`. Only *weakly* compact: this
            truncation introduces a small (~1% at the default factor)
            value discontinuity at the cutoff, so unlike bisquare/wendland
            it is not C^0-continuous at the support boundary.
        `"wendland"`: Wendland's compactly-supported C2 function,
            `(1-d/r)_+^4 * (4*d/r + 1)`.

    Returns
    -------
    (n, M) array, 0 wherever d_ik >= radius_k.
    """
    ratio = D / radius[None, :]
    if kind == "bisquare":
        raw = (1.0 - ratio**2) ** 2
    elif kind == "gaussian":
        sigma_factor = float(kind_kwargs.get("gaussian_sigma_factor", 3.0))
        raw = np.exp(-0.5 * (ratio * sigma_factor) ** 2)
    elif kind == "wendland":
        return np.where(ratio < 1.0, (1.0 - ratio) ** 4 * (4.0 * ratio + 1.0), 0.0)
    else:
        raise ValueError(f"Unknown basis kind {kind!r}; choose one of {_BASIS_KINDS}.")
    return np.where(D < radius[None, :], raw, 0.0)


def radial_basis(coords: np.ndarray, knots: np.ndarray, radius, kind: str = "bisquare", **kind_kwargs) -> np.ndarray:
    """Evaluate raw (unnormalized) radial basis functions at coords.

    Parameters
    ----------
    coords:
        (n, 2) array of spatial locations.
    knots:
        (M, 2) array of knot center locations.
    radius:
        Scalar or (M,) array of basis radii, one per knot.
    kind:
        `"bisquare"` (default), `"gaussian"`, or `"wendland"` -- see
        `_basis_kernel`'s docstring.

    Returns
    -------
    (n, M) dense array where entry [i, k] is the raw basis value at
    (coords[i], knots[k]), 0 if d_ik >= radius_k regardless of kind.
    """
    if kind not in _BASIS_KINDS:
        raise ValueError(f"Unknown basis kind {kind!r}; choose one of {_BASIS_KINDS}.")
    coords = np.asarray(coords, dtype=float)
    knots = np.asarray(knots, dtype=float)
    radius_arr = np.broadcast_to(np.asarray(radius, dtype=float), (knots.shape[0],))
    D = cdist(coords, knots)  # (n, M)
    return _basis_kernel(D, radius_arr, kind, **kind_kwargs)


def normalize_basis(raw_basis: np.ndarray) -> np.ndarray:
    """Normalize raw basis values into a partition of unity (rows sum to 1).

    Raises
    ------
    ValueError:
        If any row sums to 0, i.e. a location is not covered by any
        knot's radius. Names the offending location index rather than
        silently dividing by zero.
    """
    raw_basis = np.asarray(raw_basis, dtype=float)
    row_sums = raw_basis.sum(axis=1)

    uncovered = np.flatnonzero(row_sums == 0.0)
    if uncovered.size > 0:
        raise ValueError(
            f"Location index(es) {uncovered.tolist()} are not covered by "
            "any knot's radius (row sum is 0); cannot normalize."
        )

    return raw_basis / row_sums[:, None]


def make_hex_knot_grid(
    coords: np.ndarray,
    n_knots_target: int,
    radius_overlap_factor: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Generate a hexagonal grid of knots covering the extent of coords.

    Parameters
    ----------
    coords:
        (n, 2) array of all spot locations; used to determine the
        bounding area to cover.
    n_knots_target:
        Approximate desired number of knots. The actual count may differ
        slightly to fit a hex grid.
    radius_overlap_factor:
        Multiplier (>1) applied to the knot spacing to set each knot's
        radius, so neighboring basis functions overlap enough for a
        smooth partition-of-unity -- this is the tunable "overlap radius".

    Returns
    -------
    (knots, radius): knots is an (M, 2) array of knot centers; radius is
    the scalar radius (spacing * radius_overlap_factor) shared by every
    knot.
    """
    coords = np.asarray(coords, dtype=float)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = max(x_max - x_min, 1e-12)
    height = max(y_max - y_min, 1e-12)
    area = width * height

    spacing = np.sqrt(2.0 * area / (np.sqrt(3.0) * max(n_knots_target, 1)))
    row_spacing = spacing * np.sqrt(3.0) / 2.0
    radius = spacing * radius_overlap_factor

    n_rows = int(np.ceil((y_max - y_min) / row_spacing)) + 1
    n_cols = int(np.ceil((x_max - x_min) / spacing)) + 2

    row_idx = np.arange(n_rows)
    col_idx = np.arange(n_cols)
    ys = y_min + row_idx * row_spacing
    x_offsets = np.where(row_idx % 2 == 1, spacing / 2.0, 0.0)
    xs = (x_min + col_idx * spacing)[None, :] + x_offsets[:, None]
    ys_grid = np.broadcast_to(ys[:, None], (n_rows, n_cols))

    knots = np.stack([xs.ravel(), ys_grid.ravel()], axis=1)

    mask = (
        (knots[:, 0] >= x_min - radius) & (knots[:, 0] <= x_max + radius)
        & (knots[:, 1] >= y_min - radius) & (knots[:, 1] <= y_max + radius)
    )
    knots = knots[mask]

    return knots, float(radius)


def make_multires_basis(
    coords: np.ndarray,
    resolutions: list[dict],
    resolution_weights: list[float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Combine R independent knot-grid resolutions (e.g. one fine, one
    coarse) into a single (n, M_total) basis matrix whose rows still sum
    to exactly 1. Each resolution is built and row-normalized
    independently, then scaled by its own ``resolution_weights[r]`` entry
    and concatenated column-wise.

    Parameters
    ----------
    coords:
        (n, 2) array of spatial locations.
    resolutions:
        List of R dicts, each with keys ``n_knots_target`` (required),
        ``radius_overlap_factor`` (default 1.5), ``basis_kind`` (default
        ``"bisquare"``) -- e.g. a fine resolution (large
        ``n_knots_target``, small ``radius_overlap_factor``) and a coarse
        one (small ``n_knots_target``, large ``radius_overlap_factor``).
    resolution_weights:
        Length-R list of non-negative weights summing to 1, one per
        resolution. ``None`` (default) splits equally (``1/R`` each).

    Returns
    -------
    (B_combined, meta): ``B_combined`` is the (n, M_total) dense array;
    ``meta`` has keys ``knots_per_resolution``, ``radius_per_resolution``,
    ``basis_kind_per_resolution``, ``resolution_weights``,
    ``M_per_resolution`` (all length-R lists).
    """
    R = len(resolutions)
    if resolution_weights is None:
        resolution_weights = [1.0 / R] * R
    if len(resolution_weights) != R:
        raise ValueError(f"resolution_weights has length {len(resolution_weights)}, expected {R}")
    if not np.isclose(sum(resolution_weights), 1.0):
        raise ValueError(f"resolution_weights must sum to 1, got {sum(resolution_weights)}")

    blocks, knots_per_resolution, radius_per_resolution, basis_kind_per_resolution = [], [], [], []
    for r, cfg in enumerate(resolutions):
        n_knots_target = cfg["n_knots_target"]
        radius_overlap_factor = cfg.get("radius_overlap_factor", 1.5)
        basis_kind = cfg.get("basis_kind", "bisquare")
        knots, radius = make_hex_knot_grid(coords, n_knots_target, radius_overlap_factor)
        raw_basis = radial_basis(coords, knots, radius, kind=basis_kind)
        blocks.append(normalize_basis(raw_basis) * resolution_weights[r])
        knots_per_resolution.append(knots)
        radius_per_resolution.append(radius)
        basis_kind_per_resolution.append(basis_kind)

    B_combined = np.concatenate(blocks, axis=1)
    meta = {
        "knots_per_resolution": knots_per_resolution,
        "radius_per_resolution": radius_per_resolution,
        "basis_kind_per_resolution": basis_kind_per_resolution,
        "resolution_weights": resolution_weights,
        "M_per_resolution": [b.shape[1] for b in blocks],
    }
    return B_combined, meta
