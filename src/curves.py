"""Boundary curves and the quantities derived directly from them.

A curve is any callable `gamma(t) -> (3,) array` on t in [0, 1), closed by
gamma(1) = gamma(0). Curves are expected to live inside the unit cube; the
solver treats the domain as a 3-torus, so a curve may also wrap the domain (see
`helicoid`), in which case it closes only up to a lattice translation.
"""

import numpy as np


def discretize(gamma, num_points: int = 512) -> np.ndarray:
    """Sample a curve into a closed polyline of shape (num_points, 3)."""
    t = np.linspace(0.0, 1.0, num_points, endpoint=False)
    return np.asarray([np.asarray(gamma(ti), dtype=float) for ti in t])


def as_polylines(gamma, num_points: int = 512) -> list:
    """Normalize any boundary specification into a list of closed polylines.

    Accepts a callable, an (M, 3) array, or a sequence of either. The list form
    is what makes multi-component boundaries work: three separate rings must be
    deposited as three closed loops, not as one curve that teleports between
    them, which would lay down spurious current along the connecting jumps.
    """
    if callable(gamma):
        return [discretize(gamma, num_points)]
    if isinstance(gamma, (list, tuple)):
        out = []
        for component in gamma:
            out.extend(as_polylines(component, num_points))
        return out
    arr = np.asarray(gamma, dtype=float)
    if arr.ndim == 3:
        return [np.asarray(component, dtype=float) for component in arr]
    return [arr]


def total_area_vector(polylines) -> np.ndarray:
    """Projected area vector of a whole multi-component boundary."""
    return sum(area_vector(p) for p in polylines)


def segments(points: np.ndarray):
    """Midpoints and edge vectors of a closed polyline.

    Returns (midpoints, deltas), both (M, 3), with the closing segment included.
    """
    nxt = np.roll(points, -1, axis=0)
    return 0.5 * (points + nxt), nxt - points


def area_vector(points: np.ndarray) -> np.ndarray:
    """Projected area vector A = (1/2) * closed integral of gamma x d(gamma).

    Wang & Chern eq. (18). These are the harmonic coordinates that pin down which
    cohomology class -- i.e. which of the topologically distinct spanning
    surfaces on T^3 -- the solver converges to.

    Second-order accurate: uses segment midpoints rather than endpoints.
    """
    mid, delta = segments(points)
    return 0.5 * np.cross(mid, delta).sum(axis=0)


def tangent(t: float, gamma, h: float = 1e-3) -> np.ndarray:
    """Fourth-order central-difference tangent of a parameterized curve."""
    return (-gamma(t + 2 * h) + 8 * gamma(t + h) - 8 * gamma(t - h) + gamma(t - 2 * h)) / (12 * h)


# -- curve library ---------------------------------------------------------


def ellipse(t, a=0.3, b=0.3, center=(0.5, 0.5, 0.5), normal="z"):
    """Planar ellipse. With a == b the exact minimal surface is a flat disc of
    area pi*a^2, which is what `tests/test_plateau.py` checks against."""
    center = np.asarray(center, dtype=float)
    u, v = a * np.cos(2 * np.pi * t), b * np.sin(2 * np.pi * t)
    offsets = {"x": (0.0, u, v), "y": (v, 0.0, u), "z": (u, v, 0.0)}
    return center + np.asarray(offsets[normal])


def circle(t, radius=0.3, center=(0.5, 0.5, 0.5), normal="z"):
    return ellipse(t, a=radius, b=radius, center=center, normal=normal)


def helicoid(t, num_turns=1, radius=0.3, center=(0.5, 0.5, 0.0)):
    """Boundary of a periodic helicoid: a helix wrapping the domain once in z.

    This is the paper's validation case (their Fig. 13). It does not close in
    R^3, only on the torus, so `area_vector` is not meaningful for it -- pass the
    cohomology class explicitly when solving.
    """
    center = np.asarray(center, dtype=float)
    angle = 2 * np.pi * t * num_turns
    return center + np.asarray([radius * np.cos(angle), radius * np.sin(angle), t])


def trefoil(t, scale=0.12, center=(0.5, 0.5, 0.5)):
    """Trefoil knot -- a non-trivial test whose minimal surface is a Seifert-like
    spanning surface rather than anything disc-shaped."""
    center = np.asarray(center, dtype=float)
    a = 2 * np.pi * t
    return center + scale * np.asarray(
        [
            np.sin(a) + 2 * np.sin(2 * a),
            np.cos(a) - 2 * np.cos(2 * a),
            -np.sin(3 * a),
        ]
    )


def polygon(t, vertices):
    """Closed polygon through `vertices`, parameterized by arc length."""
    verts = np.asarray(vertices, dtype=float)
    closed = np.vstack([verts, verts[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s = np.mod(t, 1.0) * cum[-1]
    i = min(int(np.searchsorted(cum, s, side="right") - 1), len(verts) - 1)
    local = (s - cum[i]) / lengths[i]
    return closed[i] + local * (closed[i + 1] - closed[i])


def triangle(t, vertices=((0.3, 0.3, 0.5), (0.7, 0.35, 0.5), (0.5, 0.7, 0.5))):
    return polygon(t, vertices)


def borromean_rings(radius=0.28, center=(0.5, 0.5, 0.5), num_points=512) -> list:
    """Three ellipses inscribed in mutually orthogonal golden rectangles.

    The classic Borromean configuration and the paper's cover figure: the rings
    are pairwise *unlinked* (each pair has linking number zero) yet the triple
    cannot be separated, so the minimal surface spanning them is genuinely
    three-dimensional rather than three separate discs.

    Unlike the other entries here this returns a **list of three polylines**, not
    a callable -- see `as_polylines`. Passing it to `solve_plateau` works
    directly.
    """
    center = np.asarray(center, dtype=float)
    a = radius
    b = radius / 1.6180339887  # golden ratio
    t = np.linspace(0.0, 1.0, num_points, endpoint=False)
    u, v = a * np.cos(2 * np.pi * t), b * np.sin(2 * np.pi * t)
    zero = np.zeros_like(u)
    return [
        center + np.stack([u, v, zero], axis=-1),
        center + np.stack([zero, u, v], axis=-1),
        center + np.stack([v, zero, u], axis=-1),
    ]
