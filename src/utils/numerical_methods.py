import numpy as np
from ..zero_form import ZeroForm

def numerical_tangent_function(t, f, h = 0.001):
    """
    Compute the 4th-order central difference approximation of the derivative of f at t.

    Parameters:
    - f: function, the function to differentiate
    - t: float, the point at which to compute the derivative
    - h: float, the step size

    Returns:
    - float, the approximate derivative at t
    """
    return (1 / (12 * h)) * (-f(t + 2 * h) + 8 * f(t + h) - 8 * f(t - h) + f(t - 2*h))

def poisson_solve_v2(phi: np.array, h) -> np.array:
    """
    Solves Poisson's equation Δu = Phi using FFT in a periodic domain.

    Args:
        Phi_form (ZeroForm): Object containing the scalar field (Nx, Ny, Nz)
                             and the grid spacing `h`.

    Returns:
        np.array: Solution to Poisson's equation.
    """

    Nx, Ny, Nz = phi.shape

    # Compute Fourier transform of Phi
    Phi_fft = np.fft.fftn(phi)
    
    # Compute Fourier-space frequencies
    kx = np.fft.fftfreq(Nx, d=h)
    ky = np.fft.fftfreq(Ny, d=h)
    kz = np.fft.fftfreq(Nz, d=h)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute the Laplacian in Fourier space
    laplacian_fourier = 4 / (h ** 2) * (np.sin(np.pi * KX / Nx) ** 2 +
                                        np.sin(np.pi * KY / Ny) ** 2 +
                                        np.sin(np.pi * KZ / Nz) ** 2)

    # Regularize the zero frequency component
    laplacian_fourier[0, 0, 0] = 1  # Prevent division by zero

    # Solve Poisson's equation in Fourier space
    result_fft = Phi_fft / (-laplacian_fourier + 1E-9)

    # Transform back into real space
    result = np.fft.ifftn(result_fft).real
    return result

def poisson_solve(Phi_form: ZeroForm) -> np.array:
    Phi = Phi_form.scalar_field
    h = Phi_form.h

    Nx, Ny, Nz = Phi.shape
    # Mapping into Fourier space:
    Phi_fft = np.fft.fftn(Phi)
    
    # Calculate the necessary frequencies for each dimension:
    K = np.asarray([[[[k_xi / Nx, k_yi / Ny, k_zi / Nz] for k_xi in range(Nx)] for k_yi in range(Ny)] for k_zi in range(Nz)])

    # Divisor with finite difference taken into account:
    sin_correction = np.sum(np.sin(np.pi * K) ** 2, axis = -1) * 4 / (h ** 2)
    sin_correction[0, 0, 0] = sin_correction[0, 0, 0] + 1E-9
    result = Phi_fft / (-sin_correction)

    # Transform back into original space
    result = np.fft.ifftn(result)
    return result

def closest_point_on_curve(p: np.array, curve_points: np.array) -> np.array:
    dists = np.linalg.norm(curve_points - p, axis=1)
    t_star = np.argmin(dists)
    return curve_points[t_star], t_star

def closest_point_on_surface(p: np.array, surface_points: np.array) -> np.array:
    """
    Mathematically the same computation for curves and surfaces, just want to emphasise the difference in arguments.
    """
    return closest_point_on_curve(p, curve_points=surface_points)
