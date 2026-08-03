"""The discrete exterior calculus must be exact, not merely consistent.

The ADMM phi-step is an argmin, and it is only solved exactly when the Poisson
solver genuinely inverts d^T d for the same d used to form d(phi) afterwards.
The original code paired a central-difference d with a 7-point Poisson kernel,
leaving a 71% relative residual in that step; `test_phi_step_is_exact_projection`
is the regression guard for that.
"""

import numpy as np
import pytest

from src import spectral
from src.grid import Grid

RESOLUTIONS = [8, 16]


def inner(a, b):
    return float((a * b).sum())


@pytest.fixture(params=RESOLUTIONS)
def grid(request):
    return Grid(request.param)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def scalar(grid, rng):
    return rng.standard_normal(grid.res)


def vector(grid, rng):
    return rng.standard_normal((*grid.res, 3))


# -- the complex is exact --------------------------------------------------


def test_d_squared_is_zero(grid, rng):
    assert np.abs(spectral.d1(grid, spectral.d0(grid, scalar(grid, rng)))).max() < 1e-10
    assert np.abs(spectral.d2(grid, spectral.d1(grid, vector(grid, rng)))).max() < 1e-10


def test_codifferential_is_the_adjoint_of_d(grid, rng):
    phi, eta, omega, f = scalar(grid, rng), vector(grid, rng), vector(grid, rng), scalar(grid, rng)
    assert inner(spectral.d0(grid, phi), eta) == pytest.approx(
        inner(phi, spectral.delta1(grid, eta)), abs=1e-9
    )
    assert inner(spectral.d1(grid, eta), omega) == pytest.approx(
        inner(eta, spectral.delta2(grid, omega)), abs=1e-9
    )
    assert inner(spectral.d2(grid, omega), f) == pytest.approx(
        inner(omega, spectral.delta3(grid, f)), abs=1e-9
    )


def test_laplacian_factors_as_d_transpose_d(grid, rng):
    phi = scalar(grid, rng)
    got = spectral.delta1(grid, spectral.d0(grid, phi))
    want = spectral.laplace_psd(grid, phi)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_hodge_laplacian_acts_componentwise_on_2forms(grid, rng):
    omega = vector(grid, rng)
    got = spectral.d1(grid, spectral.delta2(grid, omega)) + spectral.delta3(
        grid, spectral.d2(grid, omega)
    )
    want = spectral.laplace_psd(grid, omega)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_laplacian_matches_the_papers_7point_kernel(grid):
    """Wang & Chern Algorithm 3 uses w = 4*sum sin^2(k_i/2).

    Their Poisson solver is correct; it is their eq. (23) midpoint rule for D
    that fails to pair with it. Forward differences restore the pairing.
    """
    h = grid.h
    want = sum(4.0 / h**2 * np.sin(k * h / 2.0) ** 2 for k in grid.k_space)
    assert np.abs(grid.laplace_symbol - want).max() / want.max() < 1e-12


def test_laplacian_kernel_is_constants_only(grid):
    """The midpoint rule additionally annihilates the checkerboard mode."""
    assert (grid.laplace_symbol <= 1e-12 * grid.laplace_symbol.max()).sum() == 1


# -- inverses and projections ---------------------------------------------


def test_phi_step_is_exact_projection(grid, rng):
    """d^T(d(solve_phi(Y)) - Y) == 0: the phi-subproblem is solved exactly."""
    Y = vector(grid, rng)
    residual = spectral.delta1(grid, spectral.d0(grid, spectral.solve_phi(grid, Y)) - Y)
    assert np.abs(residual).max() / np.abs(spectral.delta1(grid, Y)).max() < 1e-12


def test_laplace_inverse_roundtrips_off_the_kernel(grid, rng):
    phi = scalar(grid, rng)
    phi -= phi.mean()
    got = spectral.laplace_psd_inv(grid, spectral.laplace_psd(grid, phi))
    assert np.abs(got - phi).max() / np.abs(phi).max() < 1e-10


def test_coclosed_projection_kills_the_divergence(grid, rng):
    projected = spectral.coclosed_project(grid, vector(grid, rng))
    assert np.abs(spectral.d2(grid, projected)).max() < 1e-10


# -- locality, which is what makes the mass norm meaningful ----------------


def test_gradient_of_a_step_is_local():
    """Forward differences localize a jump to one cell.

    The minimizer is a Dirac-delta form on a surface, and the mass norm sums |X|
    without letting oscillations cancel, so a delocalized derivative inflates the
    objective. A spectral derivative rings across the whole axis and charges ~3.6x
    for the same jump, which is why it converges to the wrong minimum here.
    """
    grid = Grid(32)
    phi = (grid.positions_grid[..., 2] >= 0.5).astype(float)
    d_phi = spectral.d0(grid, phi)
    magnitude = spectral.pointwise_norm(d_phi)
    profile = magnitude[0, 0, :]
    assert (profile > 1e-8 * profile.max()).sum() == 2  # the two jumps of a periodic step
    assert spectral.mass_norm(grid, d_phi) == pytest.approx(2.0, rel=1e-9)


# -- norms and prox --------------------------------------------------------


def test_mass_norm_of_a_unit_sheet_is_its_area():
    grid = Grid(32)
    field = np.zeros((*grid.res, 3))
    field[:, :, 5, 2] = 1.0 / grid.h  # unit-mass sheet spread over one cell
    assert spectral.mass_norm(grid, field) == pytest.approx(1.0, rel=1e-12)


def test_shrink_thresholds_and_preserves_direction():
    z = np.zeros((2, 1, 1, 3))
    z[0, 0, 0] = [3.0, 4.0]+[0.0]  # |z| = 5
    z[1, 0, 0] = [0.3, 0.4, 0.0]  # |z| = 0.5
    out = spectral.shrink(z, 1.0)
    assert np.linalg.norm(out[0, 0, 0]) == pytest.approx(4.0)
    assert np.allclose(out[1, 0, 0], 0.0)
    assert np.allclose(out[0, 0, 0] / 4.0, z[0, 0, 0] / 5.0)
