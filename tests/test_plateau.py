"""End-to-end validation of the Plateau solver.

The circle is the case with a known closed-form answer: its minimal surface is
the flat disc of area pi*r^2, so `solution.mass` can be checked against a number
rather than against itself.
"""

import numpy as np
import pytest

from src import curves, extract, spectral
from src.grid import Grid
from src.initial_guess import compute_initial_guess
from src.plateau import solve_plateau

RADIUS = 0.3
EXACT_AREA = np.pi * RADIUS**2


def circle(t):
    return curves.circle(t, radius=RADIUS)


@pytest.fixture(scope="module")
def solution():
    return solve_plateau(circle, resolution=32, max_iter=400)


# -- the initial guess is feasible ----------------------------------------


def test_initial_guess_satisfies_the_boundary_constraint():
    """d(eta_0) = delta_Gamma.

    Regression guard for the sign error: solving the Biot-Savart Poisson problem
    with the analyst's Laplacian instead of the positive semi-definite one gives
    d(eta_0) = -delta_Gamma, i.e. the reversed orientation, which then fights the
    +A cohomology correction.
    """
    guess = compute_initial_guess(Grid(32), circle)
    assert guess.residual < 1e-10


def test_initial_guess_satisfies_the_cohomology_constraint():
    """mean(eta_0) = A, the projected area vector (their eq. 37).

    The original `correct_eta_0` used `A - sum(...)` where the normalization
    needs `A - mean(...)`, an error of a factor N^3 = 32768.
    """
    grid = Grid(32)
    guess = compute_initial_guess(grid, circle)
    assert guess.area[2] == pytest.approx(EXACT_AREA, rel=1e-3)
    assert np.allclose(guess.eta_0.reshape(-1, 3).mean(axis=0), guess.area, atol=1e-9)


def test_initial_guess_is_not_already_minimal():
    """Guards against a trivial 'solver' that returns its input."""
    grid = Grid(32)
    guess = compute_initial_guess(grid, circle)
    assert spectral.mass_norm(grid, guess.eta_0) > 1.5 * EXACT_AREA


# -- the solver reaches the right answer ----------------------------------


def test_circle_converges_to_the_flat_disc(solution):
    assert solution.mass == pytest.approx(EXACT_AREA, rel=0.10)


def test_solver_reduces_the_mass(solution):
    assert solution.mass < 0.65 * spectral.mass_norm(solution.grid, solution.eta_0)


def test_mass_is_not_monotonic_but_settles(solution):
    """The first X-update over-thresholds, then the dual variable pulls it back.

    lambda starts at zero, so the first shrinkage sees Z = d(phi) + eta_0 with no
    dual correction and zeroes most of the field -- history["mass"][0] lands far
    *below* the true area. Any test asserting monotone decrease is wrong about
    the algorithm, not about the implementation.
    """
    history = solution.history["mass"]
    assert history[0] < 0.6 * EXACT_AREA
    assert max(history[-20:]) - min(history[-20:]) < 0.01 * EXACT_AREA


def test_solution_stays_feasible(solution):
    """The constraint d(eta) = delta_Gamma must still hold at the optimum."""
    guess = compute_initial_guess(solution.grid, circle)
    d_eta = spectral.d1(solution.grid, solution.eta_feasible)
    rel = np.linalg.norm(d_eta - guess.delta_gamma) / np.linalg.norm(guess.delta_gamma)
    assert rel < 1e-9


def test_solution_keeps_its_cohomology_class(solution):
    mean = solution.eta_feasible.reshape(-1, 3).mean(axis=0)
    assert np.allclose(mean, solution.area, atol=1e-9)


def test_error_decreases_under_refinement():
    """First-order convergence, set by a half-cell thickening at the rim.

    Run without mollification: a fixed smoothing width in *cells* introduces a
    second error term that partially cancels the rim thickening, which lowers the
    absolute error at every resolution tested but flattens the convergence curve.
    Turning it off isolates the discretization error being measured here.
    """
    errors = [
        abs(solve_plateau(circle, resolution=N, max_iter=300, sigma_cells=0.0).mass - EXACT_AREA)
        for N in (16, 32)
    ]
    assert errors[1] < 0.75 * errors[0]


@pytest.mark.parametrize("tau", [0.3, 1.0, 3.0])
def test_fixed_point_is_independent_of_tau(tau):
    """tau sets the convergence rate, not the answer.

    A tau-dependent limit would mean the subproblems disagree -- which is exactly
    what the paper's eq. (29) does by using tau*lambda where Algorithm 2 uses
    lambda/tau. The two agree only at tau = 1, so a sweep across tau is the
    regression guard for that erratum.
    """
    got = solve_plateau(circle, resolution=16, tau=tau, max_iter=600).mass
    assert got == pytest.approx(0.2961, rel=0.06)


def test_flat_curve_gives_a_planar_surface(solution):
    """The disc spanning a circle in the z = 0.5 plane must lie in that plane."""
    magnitude = spectral.pointwise_norm(solution.eta)
    weight = magnitude.sum(axis=(0, 1))
    z = solution.grid.positions_grid[0, 0, :, 2]
    centroid = float((weight * z).sum() / weight.sum())
    assert centroid == pytest.approx(0.5, abs=2 * solution.grid.h)
    # and be concentrated: at least 80% of the mass within one cell of the plane
    mid = solution.grid.resolution // 2
    assert weight[mid - 1 : mid + 2].sum() / weight.sum() > 0.8


# -- mesh extraction -------------------------------------------------------


def test_extracted_mesh_matches_the_mass_norm(solution):
    verts, faces = extract.extract_surface(solution)
    assert len(faces) > 0
    assert extract.surface_area(verts, faces) == pytest.approx(EXACT_AREA, rel=0.25)


def test_extracted_mesh_spans_the_boundary_curve(solution):
    """Mesh vertices must reach the curve, and not extend far beyond it."""
    extract.extract_surface(solution)
    verts = solution.vertices
    radial = np.hypot(verts[:, 0] - 0.5, verts[:, 1] - 0.5)
    assert radial.max() < RADIUS + 4 * solution.grid.h
    assert radial.max() > RADIUS - 4 * solution.grid.h


# -- other curves run at all ----------------------------------------------


@pytest.mark.parametrize(
    "gamma,area",
    [
        (curves.trefoil, None),
        (curves.triangle, None),
        (lambda t: curves.helicoid(t, radius=0.25), np.zeros(3)),
    ],
)
def test_other_curves_run_and_reduce_mass(gamma, area):
    sol = solve_plateau(gamma, resolution=16, max_iter=150, area=area)
    assert np.isfinite(sol.mass) and sol.mass > 0
    assert sol.mass < spectral.mass_norm(sol.grid, sol.eta_0)


# -- planar curves: the exact answer is the region they bound ---------------


def test_ellipse_converges_to_its_planar_region():
    """A planar curve's minimal surface is the plane region it bounds: pi*a*b."""
    a, b = 0.35, 0.2
    gamma = lambda t: curves.ellipse(t, a=a, b=b)
    sol = solve_plateau(gamma, resolution=32, max_iter=400)
    assert sol.mass == pytest.approx(np.pi * a * b, rel=0.10)


def test_triangle_converges_to_its_planar_region():
    """Non-smooth boundary: corners are a different stress case from a circle.

    Exact area by the shoelace formula on the default vertices.
    """
    verts = np.array([(0.3, 0.3, 0.5), (0.7, 0.35, 0.5), (0.5, 0.7, 0.5)])
    exact = 0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
    sol = solve_plateau(curves.triangle, resolution=48, max_iter=400)
    assert sol.mass == pytest.approx(exact, rel=0.15)


def test_area_vector_matches_the_planar_area():
    """A, the cohomology class, must equal the enclosed area for a planar curve."""
    for gamma, exact in [
        (lambda t: curves.ellipse(t, a=0.35, b=0.2), np.pi * 0.35 * 0.2),
        (curves.triangle, 0.075),
    ]:
        A = curves.area_vector(curves.discretize(gamma, 2048))
        assert A[2] == pytest.approx(exact, rel=1e-4)
        assert np.allclose(A[:2], 0.0, atol=1e-9)


# -- the stopping rule certifies optimality --------------------------------


def test_residuals_decrease_monotonically_enough(solution):
    """Both primal and dual feasibility must actually improve.

    Guards the scaling bug where dual_relative sat pinned at exactly 1.0 because
    the denominator ||D^T lambda|| *is* the dual residual -- the phi-step forces
    D^T lambda = tau * D^T(X_hat - X), so normalizing by it divides the quantity
    by itself.
    """
    for key in ("primal_relative", "dual_relative"):
        series = solution.history[key]
        assert series[-1] < 0.02 * series[0]
        assert series[-1] < 1e-3


def test_convergence_flag_is_reachable():
    """`converged` must be able to become True, on the real criterion."""
    sol = solve_plateau(circle, resolution=16, rtol=3e-3, max_iter=2000)
    assert sol.converged
    assert sol.iterations < 2000
    assert sol.history["primal_relative"][-1] < 3e-3
    assert sol.history["dual_relative"][-1] < 3e-3


# -- multi-component boundaries --------------------------------------------


def test_borromean_rings_are_three_separate_loops():
    """Each ring must be deposited as its own closed loop.

    Parameterizing all three as a single curve would lay down spurious current
    along the jumps between them, so `as_polylines` keeps them separate.
    """
    rings = curves.borromean_rings()
    assert len(rings) == 3
    normalized = curves.as_polylines(rings)
    assert len(normalized) == 3
    assert all(np.allclose(a, b) for a, b in zip(normalized, rings))

    # Each ring is an ellipse of area pi*a*b normal to its own axis.
    exact = np.pi * 0.28 * (0.28 / 1.6180339887)
    for axis, ring in enumerate(rings):
        A = curves.area_vector(ring)
        assert A[(axis + 2) % 3] == pytest.approx(exact, rel=1e-3)
    assert curves.total_area_vector(rings) == pytest.approx(np.full(3, exact), rel=1e-3)


def test_multi_component_boundary_stays_feasible():
    """d(eta_0) = delta_Gamma must hold for a three-loop boundary too."""
    rings = curves.borromean_rings()
    guess = compute_initial_guess(Grid(32), rings)
    assert guess.residual < 1e-10


def test_borromean_rings_solve():
    rings = curves.borromean_rings()
    sol = solve_plateau(rings, resolution=32, max_iter=200)
    assert np.isfinite(sol.mass) and sol.mass > 0
    assert sol.mass < spectral.mass_norm(sol.grid, sol.eta_0)
    # The spanning surface is genuinely 3D, not three disjoint discs: each of
    # those would have area pi*a*b, so three of them would total ~0.46.
    assert sol.mass < 0.9 * 3 * np.pi * 0.28 * (0.28 / 1.6180339887)


def test_list_of_curves_matches_manual_concatenation():
    """A list of callables and a list of arrays must give the same delta_Gamma."""
    from src.initial_guess import dirac_delta_curve

    grid = Grid(24)
    gammas = [lambda t: curves.circle(t, radius=0.2, center=(0.5, 0.5, 0.35)),
              lambda t: curves.circle(t, radius=0.2, center=(0.5, 0.5, 0.65))]
    as_callables = dirac_delta_curve(grid, gammas)
    as_arrays = dirac_delta_curve(grid, [curves.discretize(g, 512) for g in gammas])
    assert np.allclose(as_callables, as_arrays)
