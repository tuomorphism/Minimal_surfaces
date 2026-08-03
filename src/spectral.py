"""Discrete exterior calculus on the periodic grid, diagonalized by FFT.

The exterior derivative is the forward difference (symbol `grid.d_symbol`); the
codifferential is its exact adjoint. Everything is built from that one symbol, so
the complex is exact and the Laplacian factors properly:

    d(d(.)) == 0                                to machine precision
    delta_k == d_{k-1}^T                        exactly (adjointness)
    d0^T d0 == laplace_psd == 7-point stencil   to machine precision

Degrees are identified with array shapes, using the periodic identification
V ~ E ~ F that Wang & Chern describe on p.7: 0- and 3-forms are (N,N,N); 1- and
2-forms are (N,N,N,3).

Sign convention: `laplace_psd` is the *positive* semi-definite Laplacian d^T d,
symbol +sum|s_i|^2. This matches the paper's Algorithm 3, which divides by +w,
and is the convention the Biot-Savart step requires -- using the analyst's sign
instead flips the initial guess to d(eta_0) = -delta_Gamma.
"""

import numpy as np

_CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))  # (i, j, k) with levi_civita = +1


def _fft(field):
    return np.fft.fftn(field, axes=(0, 1, 2))


def _ifft(field_hat):
    return np.fft.ifftn(field_hat, axes=(0, 1, 2)).real


def _apply(symbols, field_hats):
    return sum(s * f for s, f in zip(symbols, field_hats))


# -- exterior derivative ---------------------------------------------------


def d0(grid, phi):
    """0-form -> 1-form (gradient). (N,N,N) -> (N,N,N,3)."""
    phi_hat = _fft(phi)
    return np.stack([_ifft(s * phi_hat) for s in grid.d_symbol], axis=-1)


def d1(grid, eta):
    """1-form -> 2-form (curl). (N,N,N,3) -> (N,N,N,3)."""
    s = grid.d_symbol
    f = [_fft(eta[..., i]) for i in range(3)]
    return np.stack([_ifft(s[j] * f[k] - s[k] * f[j]) for i, j, k in _CYCLIC], axis=-1)


def d2(grid, omega):
    """2-form -> 3-form (divergence). (N,N,N,3) -> (N,N,N)."""
    return _ifft(_apply(grid.d_symbol, [_fft(omega[..., i]) for i in range(3)]))


# -- codifferential (exact adjoints of the above) --------------------------


def delta1(grid, eta):
    """1-form -> 0-form. Equals d0^T."""
    return _ifft(_apply([np.conj(s) for s in grid.d_symbol], [_fft(eta[..., i]) for i in range(3)]))


def delta2(grid, omega):
    """2-form -> 1-form. Equals d1^T."""
    s = [np.conj(x) for x in grid.d_symbol]
    f = [_fft(omega[..., i]) for i in range(3)]
    return np.stack([_ifft(s[k] * f[j] - s[j] * f[k]) for i, j, k in _CYCLIC], axis=-1)


def delta3(grid, f):
    """3-form -> 2-form. Equals d2^T."""
    f_hat = _fft(f)
    return np.stack([_ifft(np.conj(s) * f_hat) for s in grid.d_symbol], axis=-1)


# -- Laplacian -------------------------------------------------------------


def laplace_psd(grid, field):
    """Positive semi-definite Laplacian, applied componentwise.

    On 0-forms this is d0^T d0; on 1- and 2-forms the Hodge Laplacian
    (d delta + delta d) acts componentwise with the same symbol.
    """
    if field.ndim == 3:
        return _ifft(grid.laplace_symbol * _fft(field))
    return np.stack(
        [_ifft(grid.laplace_symbol * _fft(field[..., i])) for i in range(field.shape[-1])], axis=-1
    )


def laplace_psd_inv(grid, field):
    """Pseudo-inverse of `laplace_psd`; annihilates the constant mode."""
    if field.ndim == 3:
        return _ifft(grid.inv_laplace * _fft(field))
    return np.stack(
        [_ifft(grid.inv_laplace * _fft(field[..., i])) for i in range(field.shape[-1])], axis=-1
    )


def solve_phi(grid, rhs_field):
    """Least squares: argmin_phi ||d0(phi) - rhs||_L2.

    Normal equations d0^T d0 phi = d0^T rhs, i.e. laplace_psd(phi) = delta1(rhs).
    Used by the ADMM phi-step and by the level-set reconstruction, which are the
    same problem. Because d0^T d0 is exactly `laplace_symbol`, this is an exact
    projection -- the mismatched-operator version in the original code left a
    71% relative residual.
    """
    return laplace_psd_inv(grid, delta1(grid, rhs_field))


def coclosed_project(grid, omega):
    """Project a 2-form onto ker(d2), the discrete divergence-free fields.

    delta_Gamma for a closed curve satisfies d(delta_Gamma) = 0 exactly in the
    continuum. Enforcing it discretely is what makes the Biot-Savart identity in
    `initial_guess.biot_savart` hold to machine precision instead of only up to
    the spurious exact part that deposition error introduces.
    """
    return omega - delta3(grid, laplace_psd_inv(grid, d2(grid, omega)))


def mollify(grid, field, sigma_cells=0.0):
    """Gaussian smoothing of width `sigma_cells` cells; identity at 0.

    The multiplier is exp(-alpha |k|^2) with alpha = (sigma_cells * h)^2 / 2, so
    the kernel has a fixed *physical* width. A resolution-independent alpha (the
    original code defaulted to 0.01) attenuates by exp(-alpha k_max^2) with
    k_max ~ pi/h: at N=32 that keeps 7 of 32 modes per axis, and it gets worse
    under refinement rather than better.
    """
    if sigma_cells <= 0:
        return field
    kernel = np.exp(-0.5 * (sigma_cells * grid.h) ** 2 * grid.k2)
    if field.ndim == 3:
        return _ifft(kernel * _fft(field))
    return np.stack(
        [_ifft(kernel * _fft(field[..., i])) for i in range(field.shape[-1])], axis=-1
    )


# -- norms and the shrinkage prox -----------------------------------------


def pointwise_norm(field):
    """|X_v| at every vertex; shape (N,N,N)."""
    return np.linalg.norm(field, axis=-1)


def mass_norm(grid, field):
    """||X||_L1 = sum_v h^3 |X_v|  (Wang & Chern eq. 24).

    For the Dirac-delta form of a surface this is its area.
    """
    return float(pointwise_norm(field).sum() * grid.cell_volume)


def l2_norm_sq(grid, field):
    """||X||^2_L2 = h^3 sum_v |X_v|^2  (their eq. 25)."""
    return float((field**2).sum() * grid.cell_volume)


def shrink(field, threshold):
    """Vectorial soft-threshold max(1 - threshold/|z|, 0) * z, per vertex.

    The proximal operator of the mass norm, and the reason the three components
    must be collocated: |z| couples them at a single point.
    """
    norm = pointwise_norm(field)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(norm > threshold, 1.0 - threshold / np.maximum(norm, 1e-300), 0.0)
    return factor[..., None] * field
