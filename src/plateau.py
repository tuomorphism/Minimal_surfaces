"""Plateau's problem for an arbitrary closed curve in 3D.

Solves, for a boundary curve Gamma,

    minimize  ||eta||_mass   over   eta = eta_0 + d(phi),  phi a 0-form

by ADMM with Nesterov acceleration and restart (Goldstein et al. 2014,
Algorithm 8), following Wang & Chern (2021) Algorithm 1 with the corrections
documented in README.md.
"""

from dataclasses import dataclass, field

import numpy as np

from . import spectral
from .grid import Grid
from .initial_guess import compute_initial_guess


@dataclass
class PlateauSolution:
    eta: np.ndarray  # (N,N,N,3) optimal 1-form; the Dirac-delta form of the surface
    eta_feasible: np.ndarray  # eta_0 + d(phi); equals eta at convergence
    eta_0: np.ndarray  # the feasible initial guess it started from
    phi: np.ndarray  # (N,N,N) scalar potential, eta = eta_0 + grad(phi)
    grid: Grid
    area: np.ndarray  # projected area vector (cohomology class)
    mass: float  # ||eta||_mass -- the surface area
    converged: bool
    iterations: int
    history: dict = field(default_factory=dict)

    # Filled in by `extract_surface`; see extract.py.
    level_set: np.ndarray = None
    vertices: np.ndarray = None
    faces: np.ndarray = None
    level: float = None

    def __repr__(self):
        return (
            f"PlateauSolution(mass={self.mass:.5f}, iterations={self.iterations}, "
            f"converged={self.converged}, resolution={self.grid.resolution})"
        )


def optimality_residual(grid, X, floor_fraction=0.05):
    """Relative violation of the optimality condition of Wang & Chern Theorem 1.

    At the optimum there is a normalization xi of eta -- meaning xi = eta/|eta|
    wherever eta is nonzero -- that is coclosed: delta(xi) = 0. Equivalently the
    unit field along the surface is divergence-free, which is the differential
    statement that the surface has zero mean curvature.

    This is a genuine *optimality* measure. The paper's convergence criterion `c`
    only tracks how far the iterates moved, so it goes to zero when the method
    stalls just as surely as when it converges.

    Evaluated only where |eta| exceeds `floor_fraction` of its maximum, since
    xi is undefined off the surface.
    """
    norm = spectral.pointwise_norm(X)
    support = norm > floor_fraction * norm.max()
    if not support.any():
        return np.inf
    xi = np.where(support[..., None], X / np.maximum(norm, 1e-300)[..., None], 0.0)
    return float(np.abs(spectral.delta1(grid, xi))[support].mean() * grid.h)


def _auto_tau(grid, eta_0):
    """Pick the ADMM step size from the scale of the initial guess.

    The paper gives no guidance on tau, and it cannot be scale-free: the
    shrinkage threshold is 1/tau in absolute field units, while |eta_0| is set by
    the curve and the resolution. With tau fixed at 1 and a typical eta_0 whose
    bulk magnitude is well under 1, the very first X-update thresholds the entire
    field to zero.

    The notebook's `X_0 / X_0.max()` was patching exactly this, and made it
    worse: forcing max|X_0| = 1 means tau*|Z| > 1 holds nowhere, so the first
    iterate is identically zero.

    The problem is positively 1-homogeneous in eta_0, so only the ratio of the
    threshold to the field scale matters. Using the mean density mass/volume
    makes tau resolution-independent. tau sets the convergence *rate*, not the
    fixed point -- every value converges to the same answer, which is a useful
    check that the implementation is consistent -- but it changes how many
    iterations that takes by a wide margin. The constant is calibrated
    empirically against the circle (see tests/test_plateau.py).
    """
    mean_density = spectral.mass_norm(grid, eta_0) / grid.volume
    return 0.5 / max(mean_density, 1e-12)


def solve_plateau(
    gamma,
    resolution: int = 64,
    tau: float = None,
    rho: float = 0.999,
    max_iter: int = 1000,
    rtol: float = 1e-4,
    sigma_cells: float = 1.0,
    area: np.ndarray = None,
    num_points: int = None,
    callback=None,
    verbose: bool = False,
) -> PlateauSolution:
    """Compute the minimal surface spanning `gamma`.

    Parameters
    ----------
    gamma : callable(t) -> (3,) on [0,1), or an (M, 3) array of polyline points.
    resolution : grid points per axis.
    tau : ADMM step size; `None` selects it from the scale of eta_0.
    rho : acceleration/restart threshold in (0, 1).
    rtol : stop when the *relative* primal and dual residuals both fall below
        this. Deliberately not the paper's criterion `c`, which measures iterate
        movement and therefore also goes to zero when the method stalls.
        Residuals decay roughly like 1/k, so tightening this costs iterations
        steeply -- but the mass norm settles to four digits an order of magnitude
        earlier than the residuals do, so for area estimates a loose rtol is
        usually enough. `solution.history` records both if you need to check.
    sigma_cells : mollification width for delta_Gamma, in grid cells.
    area : override the cohomology class (required for domain-wrapping curves).

    Returns
    -------
    PlateauSolution, whose `.eta` is the Dirac-delta 1-form of the minimal
    surface. Pass it to `extract.extract_surface` for a triangle mesh.
    """
    grid = Grid(resolution)
    guess = compute_initial_guess(
        grid, gamma, num_points=num_points, sigma_cells=sigma_cells, area=area
    )
    X0 = guess.eta_0

    if tau is None:
        tau = _auto_tau(grid, X0)

    X = X0.copy()
    X_hat = X0.copy()
    lam = np.zeros_like(X0)
    lam_hat = np.zeros_like(X0)
    X_prev = X.copy()
    lam_prev = lam.copy()

    alpha = 1.0
    c = c_prev = np.inf
    phi = np.zeros(grid.res)
    history = {
        "mass": [],
        "criterion": [],
        "primal_residual": [],
        "dual_residual": [],
        "primal_relative": [],
        "dual_relative": [],
        "optimality": [],
    }
    converged = False
    iterations = 0

    for k in range(1, max_iter + 1):
        iterations = k
        if k > 1:
            X_prev, lam_prev, c_prev = X.copy(), lam.copy(), c

        # -- phi-update: argmin_phi <lam_hat, D phi> + (tau/2)||D phi - X_hat + X0||^2
        # Optimality is D^T D phi = D^T(X_hat - X0 - lam_hat/tau); solve_phi does
        # exactly that, and because D and the Laplacian share one symbol it is an
        # exact projection rather than an approximate one.
        Y = X_hat - X0 - lam_hat / tau
        phi = spectral.solve_phi(grid, Y)
        d_phi = spectral.d0(grid, phi)

        # -- X-update: pointwise shrinkage.
        # Z uses lam_hat/tau. Wang & Chern eq. (29) and Algorithm 4 print
        # tau*lam_hat, which contradicts their own Algorithm 2 (which uses
        # lam_hat/tau) and the derivation from their eq. (28): completing the
        # square on |X| - <lam,X> + (tau/2)|X - W|^2 gives Z = W + lam/tau. The
        # two agree only at tau = 1, which is why the error stayed hidden.
        Z = d_phi + X0 + lam_hat / tau
        X = spectral.shrink(Z, 1.0 / tau)

        # -- dual update
        primal = d_phi - X + X0
        lam = lam_hat + tau * primal

        c = (1.0 / tau) * spectral.l2_norm_sq(grid, lam - lam_hat) + tau * spectral.l2_norm_sq(
            grid, X - X_hat
        )

        # Optimality is certified by two conditions, not by the paper's `c`
        # (which measures iterate movement and so falls when the method stalls
        # just as surely as when it converges):
        #
        #   primal feasibility   D phi - X + X0 = 0
        #   dual feasibility     D^T lambda = 0     (phi carries no cost, so the
        #                                            KKT condition is exactly this)
        #
        # Note D^T lambda is not an independent quantity: the phi-step enforces
        # D^T(D phi - X_hat + X0 + lambda_hat/tau) = 0, which combined with the
        # lambda update gives D^T lambda = tau * D^T(X_hat - X). So this is the
        # usual ADMM dual residual, and normalizing it by ||D^T lambda|| would be
        # dividing the quantity by itself. Scale by h/||lambda|| instead, since
        # D ~ 1/h and lambda is O(1) (it converges to a field of unit vectors).
        dual = spectral.delta1(grid, lam)
        primal_norm = np.sqrt(spectral.l2_norm_sq(grid, primal))
        dual_norm = np.sqrt(spectral.l2_norm_sq(grid, dual))

        primal_scale = max(
            np.sqrt(spectral.l2_norm_sq(grid, X)), np.sqrt(spectral.l2_norm_sq(grid, X0)), 1e-30
        )
        dual_scale = max(np.sqrt(spectral.l2_norm_sq(grid, lam)) / grid.h, 1e-30)

        mass = spectral.mass_norm(grid, X)
        history["mass"].append(mass)
        history["criterion"].append(c)
        history["primal_residual"].append(primal_norm)
        history["dual_residual"].append(dual_norm)
        history["primal_relative"].append(primal_norm / primal_scale)
        history["dual_relative"].append(dual_norm / dual_scale)
        if callback is not None:
            callback(k, X, phi, history)

        if k == 1:
            # Paper Algorithm 1 lines 9-11. The notebook omitted this block and
            # initialized alpha = 0.1 outside the loop, making the first
            # extrapolation coefficient (alpha-1)/alpha_new negative.
            lam_hat, X_hat, alpha = lam.copy(), X.copy(), 1.0
            c_prev = c
        elif c < rho * c_prev:
            alpha_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha**2))
            beta = (alpha - 1.0) / alpha_new
            lam_hat = lam + beta * (lam - lam_prev)
            X_hat = X + beta * (X - X_prev)
            alpha = alpha_new
        else:
            lam_hat, X_hat, alpha = lam.copy(), X.copy(), 1.0
            c = c_prev / rho

        if verbose and (k % 25 == 0 or k == 1):
            print(
                f"  iter {k:4d}  mass={mass:.6f}  primal={history['primal_relative'][-1]:.2e}"
                f"  dual={history['dual_relative'][-1]:.2e}  c={c:.2e}"
            )

        # Stop on the actual optimality certificate, not on the paper's `c`.
        if history["primal_relative"][-1] < rtol and history["dual_relative"][-1] < rtol:
            converged = True
            break

    history["optimality"] = optimality_residual(grid, X)

    return PlateauSolution(
        # X is the shrinkage iterate: sharply supported, and the thing that
        # actually carries the mass norm. eta_feasible = eta_0 + d(phi) satisfies
        # the constraint exactly but is not thresholded; the two coincide at
        # convergence and their gap is the primal residual.
        eta=X,
        eta_feasible=X0 + spectral.d0(grid, phi),
        eta_0=X0,
        phi=phi,
        grid=grid,
        area=guess.area,
        mass=spectral.mass_norm(grid, X),
        converged=converged,
        iterations=iterations,
        history=history,
    )
