"""Building a feasible initial guess eta_0 from a boundary curve.

Implements Wang & Chern Algorithm 5, with three corrections (see README):

  * The Poisson solve uses the *positive* semi-definite Laplacian. Solving with
    the analyst's sign instead yields d(eta_0) = -delta_Gamma, i.e. the spanning
    surface with reversed orientation, which then fights the +A cohomology
    correction.
  * The cohomology adjustment is normalized. Their lines 12-18 accumulate
    `A - sum_v (eta_0)_{i,v}` over every vertex and add it to every vertex, with
    no division by |V|.
  * There is no separate "sharp" step. Their eq. (23) defines X = eta^sharp as a
    symmetric *average* of the two incident edges, but Algorithm 5 line 21 writes
    it as a difference -- these contradict, and the difference is a derivative,
    not an averaging. Working in collocated density units throughout makes the
    conversion the identity and the ambiguity disappears.
"""

from dataclasses import dataclass

import numpy as np

from . import curves, spectral


@dataclass
class InitialGuess:
    eta_0: np.ndarray  # (N,N,N,3) feasible 1-form, d(eta_0) = delta_Gamma
    delta_gamma: np.ndarray  # (N,N,N,3) Dirac-delta 2-form of the curve
    psi: np.ndarray  # (N,N,N,3) Biot-Savart potential
    area: np.ndarray  # (3,) projected area vector / cohomology class
    residual: float  # relative ||curl(eta_0) - delta_Gamma||, a feasibility check


def dirac_delta_curve(grid, points, sigma_cells: float = 1.0) -> np.ndarray:
    """Discretize the Dirac-delta 2-form delta_Gamma of a closed polyline.

    Deposits each segment's tangent vector onto the eight surrounding vertices
    with trilinear weights, giving the current density
    J(x) = integral over Gamma of delta^3(x - gamma) d(gamma).
    Its flux through any surface is the signed intersection number with Gamma,
    which is the defining property the paper states for delta_Gamma.

    This replaces the original per-face `SignedIntersection` scan, which looped
    over all N^3 cells x 3 faces and rescanned every curve segment inside each --
    around 5e7 segment tests at N=64. Deposition is O(number of segments).

    The 1/h^3 converts the deposited (dimensionless, DEC-integrated) intersection
    weights into a pointwise density, which is the convention used everywhere
    else in this codebase.
    """
    mid, delta = curves.segments(np.asarray(points, dtype=float))

    field = np.zeros((*grid.res, 3))
    coords = mid / grid.h
    base = np.floor(coords).astype(int)
    frac = coords - base

    for corner in np.ndindex(2, 2, 2):
        weight = np.ones(len(mid))
        for axis, c in enumerate(corner):
            weight *= frac[:, axis] if c else (1.0 - frac[:, axis])
        idx = tuple(np.mod(base[:, a] + corner[a], grid.resolution) for a in range(3))
        np.add.at(field, idx, weight[:, None] * delta / grid.cell_volume)

    # delta_Gamma of a closed curve satisfies d(delta_Gamma) = 0 exactly;
    # enforcing it discretely makes the Biot-Savart identity below exact.
    field = spectral.coclosed_project(grid, field)
    return spectral.mollify(grid, field, sigma_cells=sigma_cells)


def biot_savart(grid, delta_gamma: np.ndarray) -> tuple:
    """Find eta_tilde with d(eta_tilde) = delta_Gamma, via the Biot-Savart field.

    psi solves laplace_psd(psi) = delta_Gamma componentwise (their eq. 35), and
    eta_tilde = delta(psi). Then, since the Hodge Laplacian on 2-forms splits as
    laplace = d1 delta2 + delta3 d2,

        d1(delta2(psi)) = laplace(psi) - delta3(d2(psi)) = delta_Gamma

    because d2(delta_Gamma) = 0 after the projection above. This needs the
    positive semi-definite Laplacian; the analyst's sign gives -delta_Gamma,
    i.e. the spanning surface with reversed orientation.
    """
    psi = spectral.laplace_psd_inv(grid, delta_gamma)
    return spectral.delta2(grid, psi), psi


def compute_initial_guess(
    grid,
    gamma,
    num_points: int = None,
    sigma_cells: float = 1.0,
    area: np.ndarray = None,
) -> InitialGuess:
    """Curve -> feasible eta_0 satisfying both constraints of Problem 4.

    Parameters
    ----------
    gamma : callable(t) -> (3,), or an (M, 3) array of polyline points.
    num_points : curve samples; defaults to 8 per grid cell along the diagonal.
    sigma_cells : Gaussian mollification width for delta_Gamma, in grid cells.
    area : override the cohomology class. Required for curves that close only up
        to a lattice translation (e.g. `curves.helicoid`), where the projected
        area integral is not meaningful.
    """
    points = (
        np.asarray(gamma, dtype=float)
        if not callable(gamma)
        else curves.discretize(gamma, num_points or 8 * grid.resolution)
    )

    delta_gamma = dirac_delta_curve(grid, points, sigma_cells=sigma_cells)
    eta_tilde, psi = biot_savart(grid, delta_gamma)

    # Cohomology constraint (their eq. 37): integral of eta_0 ^ *dx_i = A_i.
    # In density units on the unit cube that is exactly mean(eta_0[..., i]) = A_i.
    # curl has zero mean by construction, so the harmonic part is just A itself.
    A = np.asarray(area, dtype=float) if area is not None else curves.area_vector(points)
    eta_0 = eta_tilde + A / grid.volume

    reconstructed = spectral.d1(grid, eta_0)
    scale = np.linalg.norm(delta_gamma)
    residual = float(np.linalg.norm(reconstructed - delta_gamma) / scale) if scale > 0 else 0.0

    return InitialGuess(
        eta_0=eta_0, delta_gamma=delta_gamma, psi=psi, area=A, residual=residual
    )
