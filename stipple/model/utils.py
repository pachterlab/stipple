"""Public utilities for :mod:`stipple.model` -- currently basis-size
resolution/clamping (see :func:`resolve_n_knots_target`)."""
from __future__ import annotations

import warnings

MIN_KNOTS_RATIO: float = 0.5  # n_knots_target below this * n gives poor recovery -- see resolve_n_knots_target


def resolve_n_knots_target(n_knots_target: int, n: int, enforce_min: bool = True) -> int:
    """Clamp a user-supplied ``n_knots_target`` into a validated range,
    warning if it had to be adjusted. No upper clamp.

    Parameters
    ----------
    n_knots_target
        User-supplied target knot count.
    n
        Number of locations (``PreprocessedData.n``).
    enforce_min
        If True (default), warn and raise ``n_knots_target`` to
        ``MIN_KNOTS_RATIO * n`` if it's below that. Set False for a
        resolution meant to stay deliberately coarse regardless of ``n``
        (e.g. ``coarse_resolution`` in a multi-resolution basis) --
        forcing a coarse layer up to fine-layer density would defeat the
        point of having one.

    Returns
    -------
    int
        The (possibly adjusted) ``n_knots_target``.
    """
    resolved = n_knots_target
    if enforce_min:
        min_knots = int(round(MIN_KNOTS_RATIO * n))
        if resolved < min_knots:
            warnings.warn(
                f"n_knots_target={n_knots_target} is below {MIN_KNOTS_RATIO:.0%} of n={n} "
                f"({min_knots}) -- recovery of the capture-rate field depends on the M/n ratio, "
                f"not on n_knots_target's absolute value, and a too-small basis relative to n "
                f"gives poor recovery. Raising n_knots_target to {min_knots}.",
                stacklevel=3,
            )
            resolved = min_knots
    return resolved
