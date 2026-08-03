import numpy as np


class Grid:
    """A periodic regular grid on the flat 3-torus T^3 = [0, L)^3.

    Fields are collocated at vertices. Everything carried on the grid is a
    pointwise *density*, never a DEC-integrated value; `initial_guess.py` is the
    single place where integrated intersection numbers get converted.

    The Fourier wavevectors are built once here so every differential operator in
    `spectral.py` derives from the same `k`. That is what makes delta*d equal the
    Laplacian to machine precision, which the phi-step of the ADMM relies on.
    """

    def __init__(self, resolution: int = 32, length: float = 1.0):
        self.resolution = int(resolution)
        self.length = float(length)

        self.res = (self.resolution,) * 3
        self.h = self.length / self.resolution
        self.scale = np.full(3, self.h)
        self.bounds = np.tile([0.0, self.length], (3, 1))

        self._build_positions()
        self._build_fourier_space()

    # -- geometry ---------------------------------------------------------

    def _build_positions(self):
        axis = np.arange(self.resolution)
        i, j, k = np.meshgrid(axis, axis, axis, indexing="ij")
        self.indices_grid = np.stack([i, j, k], axis=-1)
        self.indices_flat = self.indices_grid.reshape(-1, 3)
        self.positions_grid = self.indices_grid * self.h
        self.positions_flat = self.positions_grid.reshape(-1, 3)

    def get_grid_indices(self):
        return self.indices_grid

    def get_flat_indices(self):
        return self.indices_flat

    def get_grid_positions(self):
        return self.positions_grid

    def get_flat_positions(self):
        return self.positions_flat

    def index_to_position(self, idx):
        return np.mod(np.asarray(idx), self.resolution) * self.h

    def position_to_index(self, pos):
        idx = np.floor(np.asarray(pos) / self.h + 1e-12)
        return np.mod(idx, self.resolution).astype(int)

    # -- spectral ---------------------------------------------------------

    def _build_fourier_space(self):
        """Fourier symbols of the discrete exterior derivative.

        The exterior derivative is the *forward* difference,
        (d phi)_i(v) = (phi(v + h e_i) - phi(v)) / h, whose symbol is

            s_i = (exp(i k_i h) - 1) / h,      |s_i|^2 = 4 sin^2(k_i h / 2) / h^2.

        Summing |s_i|^2 over i reproduces the 7-point stencil of Wang & Chern's
        Algorithm 3 exactly, so their Poisson solver really does invert D^T D --
        it is their eq. (23) midpoint rule for D that breaks the pairing, not the
        solver. Forward differences also localize: the gradient of a step
        occupies one cell, where the midpoint rule smears it over two and a
        spectral derivative rings across the whole axis. Since the minimizer here
        is a Dirac-delta form concentrated on a surface, and the mass norm sums
        |X| rather than letting oscillations cancel, that locality is what makes
        the discrete minimum equal the true area instead of exceeding it.

        Diagonalizing by FFT keeps the inverse Laplacian exact while the
        operators themselves stay local.
        """
        freqs = 2 * np.pi * np.fft.fftfreq(self.resolution, d=self.h)
        self.k_freqs = freqs

        kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
        self.k_space = (kx, ky, kz)
        self.k2 = kx**2 + ky**2 + kz**2

        self.d_symbol = tuple((np.exp(1j * k * self.h) - 1.0) / self.h for k in self.k_space)
        self.laplace_symbol = sum(np.abs(s) ** 2 for s in self.d_symbol)

        # Pseudo-inverse; the kernel is the constant mode alone.
        self.inv_laplace = np.zeros_like(self.laplace_symbol)
        nonzero = self.laplace_symbol > 1e-12 * self.laplace_symbol.max()
        self.inv_laplace[nonzero] = 1.0 / self.laplace_symbol[nonzero]

    # -- measures ---------------------------------------------------------

    @property
    def cell_volume(self) -> float:
        return self.h**3

    @property
    def volume(self) -> float:
        return self.length**3

    def __repr__(self):
        return f"Grid(resolution={self.resolution}, length={self.length}, h={self.h:.5g})"
