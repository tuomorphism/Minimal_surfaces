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
    """The isovalue sitting inside the jump.

    u is only defined up to a constant, so the level must be chosen from the
    data. On the surface u sweeps across the jump interval, so the |eta|-weighted
    median of u lands inside it -- and unlike the midpoint of the global range it
    is not thrown off by the smooth far-field variation of u.
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


def extract_surface(solution, level: float = None, clip_fraction: float = 0.15):
    """Extract the minimal surface as a triangle mesh.

    Parameters
    ----------
    solution : PlateauSolution from `plateau.solve_plateau`.
    level : isovalue; chosen from the jump interval when None.
    clip_fraction : drop triangles where |eta| falls below this fraction of its
        maximum. This is what turns the closed isosurface into a surface with
        boundary Gamma. Raise it if stray sheets survive, lower it if the surface
        is eaten away near its rim.

    Returns
    -------
    (vertices, faces) in world coordinates, and populates `solution.level_set`,
    `.vertices`, `.faces` in place.
    """
    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Surface extraction needs scikit-image: pip install scikit-image"
        ) from exc

    grid = solution.grid
    eta = solution.eta
    magnitude = spectral.pointwise_norm(eta)

    u = level_set(grid, eta)
    if level is None:
        level = _jump_level(u, magnitude)

    # Pad by one cell so marching cubes closes correctly across the periodic seam.
    padded = np.pad(u, 1, mode="wrap")
    verts, faces, _, _ = measure.marching_cubes(padded, level=level)
    verts -= 1.0  # undo the pad offset, back to index coordinates

    strength = _trilinear_sample(magnitude, verts)
    keep = strength >= clip_fraction * magnitude.max()

    # Keep only triangles whose three vertices all survive, then reindex.
    face_mask = keep[faces].all(axis=1)
    faces = faces[face_mask]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=int)
    remap[used] = np.arange(len(used))
    faces = remap[faces]
    verts = verts[used] * grid.h

    solution.level_set = u
    solution.vertices = verts
    solution.faces = faces
    return verts, faces


def surface_area(vertices, faces) -> float:
    """Total area of a triangle mesh -- an independent check on `solution.mass`."""
    if len(faces) == 0:
        return 0.0
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
