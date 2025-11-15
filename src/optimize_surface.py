import numpy as np
from src.differential_form import DifferentialForm
from src.grid import Grid

def optimize_phi(grid: Grid, lambda_hat: np.array, eta_0: DifferentialForm, eta: DifferentialForm, tau: float = 0.1) -> np.array:
    assert tau > 0, "hyperparameter tau should be positive."

    # From optimality condition, define helper form Y from which we compute the codifferential (codivergence)
    Y_field = eta.field - eta_0.field - (1 / tau) * lambda_hat
    D_Y = DifferentialForm(grid = grid, field = Y_field, form_degree = 1).codifferential(alpha = 0.001)

    # Solve poisson equation
    return eta.fft_poisson_solve(D_Y.field)

