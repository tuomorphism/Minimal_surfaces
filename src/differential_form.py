import numpy as np

class DifferentialForm:
    def __init__(self, grid, form_degree, field=None):
        assert form_degree in [0, 1, 2], "Only 0-, 1-, and 2-forms are supported"
        self.grid = grid
        self.h = grid.scale[0]
        self.form_degree = form_degree
        if field is not None:
            self.field = field
        else:
            shape = (*grid.res, 1) if form_degree == 0 else (*grid.res, 3)
            self.field = np.zeros(shape)

    def exterior_derivative(self):
        """
        Computes the exterior derivative dw for k-form form w, dw is a (k + 1)-form.
        """
        h = self.h
        d_field = np.zeros((*self.grid.res, 3))
        if self.form_degree == 0:
            phi = self.field[..., 0]
            d_field[..., 0] = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*h)
            d_field[..., 1] = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*h)
            d_field[..., 2] = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2*h)
            return DifferentialForm(self.grid, 1, d_field)
        elif self.form_degree == 1:
            Fx, Fy, Fz = self.field[..., 0], self.field[..., 1], self.field[..., 2]
            # The rules for computing the different elements come from standard
            # wedge product rules for standard basis functionals dx, dy and dz
            # combined with the definition of an exterior derivative.
            # Notably, the first element is the coefficient for dy v dz
            # second is dx v dz and third is dx v dy.
            d_field[..., 0] = (np.roll(Fz, -1, axis=1) - np.roll(Fz, 1, axis=1)) / (2*h) - \
                              (np.roll(Fy, -1, axis=2) - np.roll(Fy, 1, axis=2)) / (2*h)
            d_field[..., 1] = (np.roll(Fx, -1, axis=2) - np.roll(Fx, 1, axis=2)) / (2*h) - \
                              (np.roll(Fz, -1, axis=0) - np.roll(Fz, 1, axis=0)) / (2*h)
            d_field[..., 2] = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2*h) - \
                              (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2*h)
            return DifferentialForm(self.grid, 2, d_field)
        elif self.form_degree == 2:
            Gx, Gy, Gz = self.field[..., 0], self.field[..., 1], self.field[..., 2]
            div = ((np.roll(Gx, -1, axis=0) - np.roll(Gx, 1, axis=0)) +
                   (np.roll(Gy, -1, axis=1) - np.roll(Gy, 1, axis=1)) +
                   (np.roll(Gz, -1, axis=2) - np.roll(Gz, 1, axis=2))) / (2*h)
            return DifferentialForm(self.grid, 0, div[..., np.newaxis])

    def fft_poisson_solve(self, rhs_field):
        """
        Solves poisson equation on toroidal 3-dimensional space.
        """
        Nx, Ny, Nz = self.grid.res
        kx, ky, kz = self.grid.k_space
        h = self.grid.scale[0] # Assumes uniform grid
        
        k2 = kx**2 + ky**2 + kz**2
        k2[0, 0, 0] = 1  # avoid division by zero for DC component
        f_hat = np.fft.fftn(rhs_field[..., 0])

        # Compute the Laplacian in Fourier space
        laplacian_fourier = 4 / (h ** 2) * (
            np.sin(kx / Nx) ** 2 +
            np.sin(ky / Ny) ** 2 +
            np.sin(kz / Nz) ** 2
        )

        # Regularize the zero frequency component
        laplacian_fourier[0, 0, 0] = 1  # Prevent division by zero

        # Solve Poisson's equation in Fourier space
        result_fft = f_hat / (-laplacian_fourier + 1E-9)

        # Transform back into real space
        result = np.fft.ifftn(result_fft).real

        return DifferentialForm(self.grid, 0, result[..., np.newaxis])
    
    def hodge_star(self):
        """
        The hodge star operation, defined using the standard euclidean metric.
        Produces the dual differential forms for any 0, 1, or 2 -form. 
        """
        if self.form_degree == 0:
            return DifferentialForm(self.grid, 0, self.field.copy()) # 0-form equal to 3-form
        
        # For 1-forms and 2-forms, since the forms are stored as values on vertices, 
        # we just return the same field with degree of dual form.
        # This corresponds to the order of elements we defined, 
        # the first component being associated with x-component for 1-forms
        # and for 2-forms the first component being the flux through dy v dz area form.
        elif self.form_degree == 1:
            return DifferentialForm(self.grid, 2, self.field.copy())
        else:
            return DifferentialForm(self.grid, 1, self.field.copy())

    def codifferential(self):
        return self.hodge_star().exterior_derivative().hodge_star()

    def exterior_derivative_fourier(self, alpha=0.0):
        kx, ky, kz = self.grid.k_space
        k2 = kx**2 + ky**2 + kz**2
        # Apply smoothing to get rid of Gibb's phenomena for rough impulses.
        smoothing = np.exp(-alpha * k2)
        if self.form_degree == 0:
            F = np.fft.fftn(self.field[..., 0]) * smoothing
            grad = np.stack([
                np.fft.ifftn(1j * kx * F).real,
                np.fft.ifftn(1j * ky * F).real,
                np.fft.ifftn(1j * kz * F).real
            ], axis=-1)
            return DifferentialForm(self.grid, 1, grad)
        elif self.form_degree == 1:
            Fx = np.fft.fftn(self.field[..., 0]) * smoothing
            Fy = np.fft.fftn(self.field[..., 1]) * smoothing
            Fz = np.fft.fftn(self.field[..., 2]) * smoothing

            curl = np.stack([
                np.fft.ifftn(1j * (ky * Fz - kz * Fy)).real,
                np.fft.ifftn(1j * (kz * Fx - kx * Fz)).real,
                np.fft.ifftn(1j * (kx * Fy - ky * Fx)).real
            ], axis=-1)
            return DifferentialForm(self.grid, 2, curl)
        else:
            Fx = np.fft.fftn(self.field[..., 0]) * smoothing
            Fy = np.fft.fftn(self.field[..., 1]) * smoothing
            Fz = np.fft.fftn(self.field[..., 2]) * smoothing
            div_hat = 1j * (kx * Fx + ky * Fy + kz * Fz)
            div = np.fft.ifftn(div_hat).real
            return DifferentialForm(self.grid, 0, div[..., np.newaxis]) # Note that 3-forms are equal to 0-forms since they are both scalar fields for manifold of dimension 3.

    def codifferential_fourier(self, alpha=0.0):
        if self.form_degree not in [1, 2]:
            raise NotImplementedError("Fourier codifferential only implemented for 1- and 2-forms")

        return self.hodge_star().exterior_derivative_fourier(alpha=alpha).hodge_star()

    def l2_norm(self):
        assert self.form_degree == 1, 'L_2 norm being only defined for 1-forms.'
        vol = self.h ** 3
        return np.sqrt(np.sum(self.field ** 2) * vol)

    def __repr__(self):
        return f"DifferentialForm(degree={self.form_degree}, shape={self.field.shape})"
    
    def evaluate_at_point(self, point):
        """
        Evaluate the form by nearest grid point, no interpolation
        """
        idx = self.grid.position_to_index(point)
        if np.any(idx < 0) or np.any(idx >= self.grid.res):
            return np.zeros(3 if self.form_degree > 0 else 1)
        return self.field[tuple(idx)]