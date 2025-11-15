import numpy as np
from src.grid import Grid
from src.differential_form import DifferentialForm
from src.utils.numerical_methods import poisson_solve_v2 as poisson_solve
from src.utils.intersection_calculations import SignedIntersection
from src.utils.compute_area import compute_area_vector

def compute_initial_guess(grid: Grid, gamma: callable, alpha: float = 0.01) -> dict:
    """
    Compute the initial guess η₀ = ⋅d⋅ψ from a space curve γ using the Biot–Savart inverse.
    Returns:
        - 'delta_gamma': DifferentialForm, Hodge dual of δΓ (1-form)
        - 'psi': DifferentialForm, coexact 1-form solving ⋅d⋅ψ = δΓ
        - 'eta_tilde': DifferentialForm, η₀ = curl(ψ)
        - 'eta_0': DifferentialForm, corrected η₀ with cohomology constraints
    """
    star_delta_gamma = _compute_star_delta_gamma(grid, gamma)
    psi = _solve_biot_savart(grid, star_delta_gamma)
    eta_tilde = psi.exterior_derivative_fourier(alpha=alpha)
    eta_0 = _add_cohomology_constraints(grid, eta_tilde, gamma=gamma)

    return {
        'delta_gamma': star_delta_gamma,
        'psi': psi,
        'eta_tilde': eta_tilde,
        'eta_0': eta_0,
        'X_0': eta_0.field # Just obtain the raw field from eta_0.
    }

def _add_cohomology_constraints(grid: Grid, eta_0_tilde: DifferentialForm, gamma: callable) -> DifferentialForm:
    if eta_0_tilde.form_degree != 2:
        raise ValueError("Cohomology projection only defined for 2-forms")

    A = compute_area_vector(
        t_values=np.linspace(grid.bounds[0][0], grid.bounds[0][1], num=grid.res[0]),
        curve_function=gamma
    )

    values = np.mean(eta_0_tilde.field.reshape(-1, 3), axis=0)
    correction = np.broadcast_to(A - values, eta_0_tilde.field.shape)
    corrected_field = eta_0_tilde.field + correction
    return DifferentialForm(grid=grid, form_degree=2, field=corrected_field)

def _compute_star_delta_gamma(grid: Grid, gamma: callable) -> DifferentialForm:
    Nx, _, _ = grid.res
    h = grid.scale[0]  # Assume uniform spacing
    t_range = np.linspace(0, 1, num=Nx)
    curve_points = np.asarray([gamma(t) for t in t_range])

    field = np.zeros((*grid.res, 3), dtype=float)

    for xi, yi, zi in grid.get_flat_indices():
        x, y, z = grid.index_to_position([xi, yi, zi])
        for i in range(3):
            face = (np.array([x, y, z]), i, (h, h))
            flux = SignedIntersection(curve_points, face)
            field[xi, yi, zi, i] = flux

    return DifferentialForm(grid=grid, form_degree=1, field=field)

def _solve_biot_savart(grid: Grid, star_delta_gamma: DifferentialForm) -> DifferentialForm:
    if star_delta_gamma.form_degree != 1:
        raise ValueError("Expected a 1-form as the Hodge dual of δΓ")

    field = star_delta_gamma.field
    psi_components = [poisson_solve(field[..., i], h=grid.scale[0]) for i in range(3)]
    psi_field = np.stack(psi_components, axis=-1)
    return DifferentialForm(grid=grid, form_degree=1, field=psi_field)
