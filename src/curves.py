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


def borromean_rings(t, radius=0.22, offset=0.09, center=(0.5, 0.5, 0.5)):
    """Three linked ellipses in mutually orthogonal planes, traversed in sequence.

    The paper's cover figure. Returned as a single parameterized curve with three
    components; the connecting jumps are degenerate and contribute no area.
    """
    center = np.asarray(center, dtype=float)
    branch = int(np.mod(t, 1.0) * 3) % 3
    local = np.mod(t * 3, 1.0)
    u, v = radius * np.cos(2 * np.pi * local), radius * np.sin(2 * np.pi * local)
    if branch == 0:
        return center + np.asarray([u, v, offset])
    if branch == 1:
        return center + np.asarray([offset, u, v])
    return center + np.asarray([v, offset, u])
