"""Recovering a triangle mesh from the Dirac-delta 1-form eta.

Wang & Chern section 4.1. The optimizer eta = delta_Sigma is an impulse
concentrated on the minimal surface, not a mesh. To render it we find the 0-form
u whose differential best matches eta in least squares (their eq. 38),

    u = argmin ||d(u) - delta_Sigma||_L2,

which is the same normal-equation solve as the ADMM phi-step. Theory (their
eq. 40-41) says u then has a jump of exactly 1 across Sigma and is smooth
elsewhere, so an isosurface taken inside that jump contains Sigma.

An isosurface is always closed, so it necessarily extends past the intended
boundary Gamma. The excess is clipped using |eta|, which vanishes off the
surface.
"""

import numpy as np

from . import spectral


def level_set(grid, eta) -> np.ndarray:
    """The 0-form u with a unit jump across the surface. Mean-centred."""
    u = spectral.solve_phi(grid, eta)
    return u - u.mean()


def _jump_level(u, weights) -> float:
    """The |eta|-weighted median of u over the surface -- a starting estimate.

    u is only defined up to a constant, so the level has to come from the data.
    Unlike the midpoint of the global range this is not thrown off by the smooth
    far-field variation of u. It is only a seed for `_select_level`; see there
    for why the median alone is not good enough.
    """
    order = np.argsort(u.ravel())
    vals, w = u.ravel()[order], weights.ravel()[order]
    total = w.sum()
    if total <= 0:
        return float(np.median(u))
    return float(vals[np.searchsorted(np.cumsum(w), 0.5 * total)])


def _trilinear_sample(field, points):
    """Sample a scalar grid at fractional index coordinates, periodically."""
    n = field.shape[0]
    base = np.floor(points).astype(int)
    frac = points - base
    out = np.zeros(len(points))
    for corner in np.ndindex(2, 2, 2):
        weight = np.ones(len(points))
        for axis, c in enumerate(corner):
            weight *= frac[:, axis] if c else (1.0 - frac[:, axis])
        idx = tuple(np.mod(base[:, a] + corner[a], n) for a in range(3))
        out += weight * field[idx]
    return out


def extract_surface(solution, level: float = None, clip_fraction: float = 0.15,
                    level_search: int = 13):
    """Extract the minimal surface as a triangle mesh.

    Parameters
    ----------
    solution : PlateauSolution from `plateau.solve_plateau`.
    level : isovalue; when None it is chosen by `_select_level` to recover the
        most surface area.
    level_search : how many candidate isovalues to try when `level` is None.
    clip_fraction : drop triangles where |eta| falls below this fraction of its
        maximum. This is what turns the closed isosurface into a surface with
        boundary Gamma. Raise it if stray sheets survive, lower it if the surface
        is eaten away near its rim.

    Returns
    -------
    (vertices, faces) in world coordinates, and populates `solution.level_set`,
    `.vertices`, `.faces` in place.
    """
    grid = solution.grid
    eta = solution.eta
    magnitude = spectral.pointwise_norm(eta)

    u = level_set(grid, eta)
    if level is None:
        level = _select_level(u, magnitude, grid, clip_fraction, level_search,
                              target_area=solution.mass)

    verts, faces = _mesh_at_level(u, magnitude, grid, level, clip_fraction)

    solution.level_set = u
    solution.vertices = verts
    solution.faces = faces
    solution.level = level
    return verts, faces


def _mesh_at_level(u, magnitude, grid, level, clip_fraction):
    """Marching cubes at one isovalue, clipped to the support of eta."""
    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Surface extraction needs scikit-image: pip install scikit-image"
        ) from exc

    # Pad by one cell so marching cubes closes correctly across the periodic seam.
    padded = np.pad(u, 1, mode="wrap")
    try:
        verts, faces, _, _ = measure.marching_cubes(padded, level=level)
    except (ValueError, RuntimeError):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    verts -= 1.0  # undo the pad offset, back to index coordinates

    strength = _trilinear_sample(magnitude, verts)
    keep = strength >= clip_fraction * magnitude.max()

    # Keep only triangles whose three vertices all survive, then reindex.
    faces = faces[keep[faces].all(axis=1)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return verts[used] * grid.h, remap[faces]


def _select_level(u, magnitude, grid, clip_fraction, n_candidates=13,
                  target_area=None) -> float:
    """Pick the isovalue whose recovered area best matches the mass norm.

    For a disc, u jumps sharply across Sigma and barely varies along it, so any
    level inside the jump works and the weighted median is fine. For a
    topologically interesting surface that stops being true: u also drifts
    smoothly *along* the surface by an amount comparable to the unit jump, so a
    single isosurface only catches the part of Sigma where u happens to sit near
    that level. Measured against the mass norm, the weighted median recovers 102%
    of the area for a circle but only 74% for Borromean rings.

    The objective is |area - ||eta||_mass|, not "largest area". Those differ:
    maximizing area alone overshoots on a disc (109%), because there is always
    some level that sweeps up extra geometry. The mass norm is exactly what the
    surface area should equal, so it is the right target to aim at rather than a
    quantity to maximize. Clipping keeps the search honest -- sheets away from
    Sigma are removed before the area is measured.

    A single level remains a genuine limitation for complex topologies; the
    recovery ratio is asserted in `tests/test_plateau.py` so regressions show up.
    """
    support = magnitude > 0.2 * magnitude.max()
    if not support.any():
        return _jump_level(u, magnitude)

    candidates = np.quantile(u[support], np.linspace(0.15, 0.85, n_candidates))

    best_level, best_score = float(candidates[0]), np.inf
    for candidate in candidates:
        verts, faces = _mesh_at_level(u, magnitude, grid, float(candidate), clip_fraction)
        area = surface_area(verts, faces)
        score = -area if target_area is None else abs(area - target_area)
        if score < best_score:
            best_level, best_score = float(candidate), score
    return best_level


def surface_area(vertices, faces) -> float:
    """Total area of a triangle mesh -- an independent check on `solution.mass`."""
    if len(faces) == 0:
        return 0.0
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
