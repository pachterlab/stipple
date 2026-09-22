"""Detect and transform regular spatial grids (hexagonal or square).

The typical input is ``adata.obsm["spatial"]``, an ``(n_points, 2)`` array
of spot/cell coordinates. :func:`check_grid` determines whether those
coordinates form a regular hexagonal or square lattice, and
:func:`to_grid_coordinates` maps them onto a rectilinear index grid.
:func:`stipple.model._StippleBasisModel` fits directly on the raw coordinates
(see :attr:`stipple.model.data.PreprocessedData.coords_raw`), not this transformed
grid -- it's retained for ``grid_info`` (grid type/spacing metadata) and
for any grid-aware plotting or analysis that wants it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

SQRT2 = np.sqrt(2.0)


class _SquareInverseTransform:
    """Identity inverse transform for a square grid (cast to float) --
    a module-level class, not a closure, so :func:`to_grid_coordinates`'s
    result stays picklable (a function object defined inside another
    function cannot be pickled; an instance of a module-level class can)."""

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(t, dtype=float)


class _HexInverseTransform:
    """Inverse transform for a hexagonal grid: maps oblique integer
    ``(u, v)`` lattice coordinates back to the original Cartesian spatial
    coordinates via the lattice's own ``origin``/``basis`` -- a
    module-level class (not a closure over :func:`to_grid_coordinates`'s
    locals) so it stays picklable; see :class:`_SquareInverseTransform`.
    """

    def __init__(self, origin: np.ndarray, basis: np.ndarray) -> None:
        self.origin = origin
        self.basis = basis

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.origin + t @ self.basis.T


@dataclass
class GridInfo:
    """Result of :func:`check_grid`."""

    grid_type: str
    spacing: float
    n_interior: int
    n_boundary: int
    warning: str | None = None


def _mode(counts: np.ndarray) -> int:
    values, freq = np.unique(counts, return_counts=True)
    return int(values[np.argmax(freq)])


def check_grid(
    coords: np.ndarray,
    spacing_tol: float = 0.10,
    shell_tol: float = 0.15,
    min_interior_fraction: float = 0.05,
    verbose: bool = False,
) -> GridInfo:
    """Detect whether ``coords`` form a regular hexagonal or square grid.

    The nearest-neighbor distance distribution is checked for a single,
    tightly concentrated mode (indicating uniform spacing), then each
    point's neighbor shells are counted to classify it as hexagonal
    (6 equidistant neighbors) or square (4 neighbors at distance ``d``
    plus 4 diagonal neighbors at ``d * sqrt(2)``). Points matching the
    full expected pattern are "interior"; the rest are "boundary".

    Parameters
    ----------
    coords
        Array of shape ``(n_points, 2)``, e.g. ``adata.obsm["spatial"]``.
    spacing_tol
        Maximum allowed relative spread (5th-95th percentile range over
        the median) of nearest-neighbor distances before the spacing is
        flagged as not tightly concentrated.
    shell_tol
        Relative tolerance used when counting neighbors within a
        distance shell.
    min_interior_fraction
        Minimum fraction of points that must match the full interior
        neighbor pattern before the grid is accepted as regular.
    verbose
        If True, print the detected grid type, spacing, and interior /
        boundary point counts.

    Returns
    -------
    GridInfo
        Detected grid type, spacing ``delta``, interior/boundary point
        counts, and an optional warning message.

    Raises
    ------
    ValueError
        If ``coords`` does not form a recognizable regular grid.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"expected coords of shape (n_points, 2), got {coords.shape}")
    n = coords.shape[0]
    if n < 7:
        raise ValueError("need at least 7 points to assess grid regularity")

    tree = cKDTree(coords)
    k = min(13, n)
    dists, _ = tree.query(coords, k=k)
    nn_dist = dists[:, 1]

    delta = float(np.median(nn_dist))
    spread = (np.percentile(nn_dist, 95) - np.percentile(nn_dist, 5)) / delta
    warning_msg = None
    if spread > spacing_tol:
        warning_msg = (
            "nearest-neighbor distances are not tightly concentrated "
            f"(5th-95th percentile spread is {spread:.1%} of the median "
            "spacing); the grid may be irregular or jittered"
        )

    def shell_count(lo: float, hi: float) -> np.ndarray:
        return np.sum((dists[:, 1:] >= lo) & (dists[:, 1:] <= hi), axis=1)

    shell1 = shell_count(delta * (1 - shell_tol), delta * (1 + shell_tol))
    shell1_mode = _mode(shell1)

    if shell1_mode == 6:
        grid_type = "hexagonal"
        interior_mask = shell1 == 6
    elif shell1_mode == 4:
        shell2 = shell_count(
            delta * SQRT2 * (1 - shell_tol), delta * SQRT2 * (1 + shell_tol)
        )
        if _mode(shell2) == 4:
            grid_type = "square"
            interior_mask = (shell1 == 4) & (shell2 == 4)
        else:
            grid_type = None
            interior_mask = np.zeros(n, dtype=bool)
    else:
        grid_type = None
        interior_mask = np.zeros(n, dtype=bool)

    n_interior = int(interior_mask.sum())
    n_boundary = n - n_interior
    interior_fraction = n_interior / n

    if grid_type is None or interior_fraction < min_interior_fraction:
        raise ValueError(
            "coordinates do not form a recognizable regular hexagonal or "
            f"square grid: the most common neighbor count is {shell1_mode} "
            f"and only {interior_fraction:.1%} of points match a "
            "consistent interior pattern. Check adata.obsm['spatial'] for "
            "irregular spacing, jitter, or missing spots."
        )

    if warning_msg is None and interior_fraction < 0.5:
        warning_msg = (
            f"only {interior_fraction:.1%} of points match the full "
            f"interior neighbor pattern expected for a {grid_type} grid; "
            "the grid may be small or have many missing spots"
        )

    if verbose:
        msg = (
            f"[stipple.model.grid] detected {grid_type} grid: spacing delta = "
            f"{delta:.4g}, {n_interior} interior points, {n_boundary} "
            "boundary points"
        )
        if warning_msg:
            msg += f"\n[stipple.model.grid] WARNING: {warning_msg}"
        print(msg)

    if warning_msg is not None:
        warnings.warn(warning_msg, stacklevel=2)

    return GridInfo(
        grid_type=grid_type,
        spacing=delta,
        n_interior=n_interior,
        n_boundary=n_boundary,
        warning=warning_msg,
    )


def _circular_mean(angles: np.ndarray, period: float) -> float:
    scale = 2 * np.pi / period
    c = np.mean(np.cos(angles * scale))
    s = np.mean(np.sin(angles * scale))
    return float(np.mod(np.arctan2(s, c) / scale, period))


def _hex_basis_vectors(
    coords: np.ndarray, tree: cKDTree, delta: float, shell_tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the two lattice basis vectors (60 degrees apart) of a
    hexagonal point set, robust to the grid's global orientation."""
    lo, hi = delta * (1 - shell_tol), delta * (1 + shell_tol)
    pairs = tree.query_pairs(r=hi, output_type="ndarray")
    vec = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    dist = np.linalg.norm(vec, axis=1)
    vec = vec[dist >= lo]

    # The six hexagonal neighbor directions are spaced 60 degrees apart,
    # so folding every direction (and its reverse) modulo 60 degrees
    # collapses them onto a single reference angle.
    angles = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), np.pi / 3)
    theta0 = _circular_mean(angles, np.pi / 3)
    e1 = delta * np.array([np.cos(theta0), np.sin(theta0)])
    e2 = delta * np.array([np.cos(theta0 + np.pi / 3), np.sin(theta0 + np.pi / 3)])
    return e1, e2


N_ROT_HEX = 6


def _rotate_hex(uv: np.ndarray, r: int) -> np.ndarray:
    """Rotate oblique integer hex lattice ``(u, v)`` points by ``60 * r``
    degrees, ``r % 6``, via the cube-coordinate cyclic rotation
    ``(x, y, z) -> (-z, -x, -y)`` (``x = u``, ``z = v``, ``y = -x - z``).

    Parameters
    ----------
    uv : np.ndarray, shape (n, 2)
    r
        Rotation count, taken mod 6.
    """
    x = uv[:, 0].astype(np.int64).copy()
    z = uv[:, 1].astype(np.int64).copy()
    y = -x - z
    for _ in range(r % 6):
        x, y, z = -z, -x, -y
    return np.stack([x, z], axis=1)


def _cube_rotate_vector(vec: np.ndarray, r: int) -> np.ndarray:
    """Same cube-coordinate rotation as :func:`_rotate_hex`, applied to a
    single 2-vector rather than an ``(n, 2)`` array -- used to rotate a
    basis vector (as opposed to lattice coordinates) by ``60 * r`` degrees.
    """
    x, z = float(vec[0]), float(vec[1])
    y = -x - z
    for _ in range(r % 6):
        x, y, z = -z, -x, -y
    return np.array([x, z])


def _bbox_cells(coords_int: np.ndarray) -> int:
    span0 = int(coords_int[:, 0].max() - coords_int[:, 0].min() + 1)
    span1 = int(coords_int[:, 1].max() - coords_int[:, 1].min() + 1)
    return span0 * span1


def _reduce_hex_coords(coords_int: np.ndarray) -> tuple[np.ndarray, int]:
    """Try all 6 equally-valid hex basis choices (via 60-degree rotation of
    the already-computed ``(u, v)`` array) and keep whichever minimizes the
    bounding-box cell count.

    Parameters
    ----------
    coords_int : np.ndarray, shape (n, 2)

    Returns
    -------
    (coords_reduced, r_star)
        ``coords_reduced`` is ``coords_int`` under the chosen basis (same
        row order); ``r_star`` is which of the 6 rotations was kept.
    """
    best_cells = None
    best_r = 0
    best_coords = coords_int
    for r in range(N_ROT_HEX):
        candidate = _rotate_hex(coords_int, r)
        cells = _bbox_cells(candidate)
        if best_cells is None or cells < best_cells:
            best_cells, best_r, best_coords = cells, r, candidate
    return best_coords, best_r


def to_grid_coordinates(
    coords: np.ndarray,
    grid_info: GridInfo,
    shell_tol: float = 0.15,
    residual_tol: float = 0.25,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Transform spatial coordinates onto a regular index grid.

    For hexagonal grids, points are mapped to oblique integer
    coordinates ``(u, v)`` along the lattice's own basis vectors, which
    turns the hex lattice into a rectilinear index grid. For square grids
    the Cartesian coordinates already form a rectilinear grid and are used
    directly.

    Parameters
    ----------
    coords
        Array of shape ``(n_points, 2)``, matching the coordinates
        passed to :func:`check_grid`.
    grid_info
        The :class:`GridInfo` returned by :func:`check_grid` for
        ``coords``.
    shell_tol
        Relative tolerance used when identifying first-shell hexagonal
        neighbors while estimating the lattice basis vectors.
    residual_tol
        Relative-to-spacing residual above which a point's rounded
        ``(u, v)`` is considered a poor fit to the estimated lattice.

    Returns
    -------
    transformed
        Integer ``(u, v)`` grid coordinates for hexagonal grids, or the
        original Cartesian coordinates for square grids.
    inverse_transform
        Callable mapping transformed coordinates back to the original
        spatial coordinate space.
    """
    coords = np.asarray(coords, dtype=float)

    if grid_info.grid_type == "square":
        return coords.copy(), _SquareInverseTransform()

    if grid_info.grid_type != "hexagonal":
        raise ValueError(f"unrecognized grid type {grid_info.grid_type!r}")

    tree = cKDTree(coords)
    e1, e2 = _hex_basis_vectors(coords, tree, grid_info.spacing, shell_tol)
    basis = np.column_stack([e1, e2])
    basis_inv = np.linalg.inv(basis)

    origin = coords[0].copy()
    uv_real = (coords - origin) @ basis_inv.T
    uv = np.round(uv_real).astype(int)

    reconstructed = origin + uv @ basis.T
    residual = np.linalg.norm(coords - reconstructed, axis=1) / grid_info.spacing
    bad_fraction = float(np.mean(residual > shell_tol + residual_tol))
    if bad_fraction > 0.05:
        warnings.warn(
            "hexagonal-to-oblique coordinate transform has larger-than-"
            f"expected residuals for {bad_fraction:.1%} of points; the "
            "detected lattice orientation may be inaccurate",
            stacklevel=2,
        )

    # Rotating uv by r_star is equivalent to re-deriving it from basis
    # rotated by -r_star, so the inverse transform is rebuilt from that
    # rotated basis to stay consistent with the reduced coordinates.
    uv, r_star = _reduce_hex_coords(uv)
    if r_star != 0:
        e1_r = _cube_rotate_vector(np.array([1.0, 0.0]), -r_star) @ basis.T
        e2_r = _cube_rotate_vector(np.array([0.0, 1.0]), -r_star) @ basis.T
        basis = np.column_stack([e1_r, e2_r])

    return uv, _HexInverseTransform(origin, basis)
