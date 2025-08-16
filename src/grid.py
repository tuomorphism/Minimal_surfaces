import numpy as np

class Grid:
    def __init__(self, resolution=32):
        """
        Create a 3D grid of integer indices and associated spatial coordinates.

        Parameters:
        - resolution: int
        """
        self.resolution = resolution
        self.scale: float = 1.0 / self.resolution
        self.indices_grid, self.indices_flat = self._generate_indices()
        self.positions_grid, self.positions_flat = self._generate_positions(self.indices_flat)

    def _generate_indices(self) -> tuple:
        i, j, k = np.meshgrid(
            np.arange(self.resolution), np.arange(self.resolution), np.arange(self.resolution),
            indexing="ij"
        )
        indices_grid = np.stack([i, j, k], axis=-1)      # Shape (Nx, Ny, Nz, 3)
        indices_flat = indices_grid.reshape(-1, 3)  # Shape (Nx*Ny*Nz, 3)
        return (indices_grid, indices_flat)

    def _generate_positions(self, indices_flat) -> tuple:
        # From flat indices, generate grid and flat positions.
        positions_flat = indices_flat * self.scale
        positions_grid = positions_flat.reshape(self.resolution, self.resolution, self.resolution, 3)
        return (positions_grid, positions_flat)

    def index_to_position(self, idx):
        """
        Convert a single integer index [i, j, k] to spatial coordinates.
        Wraps indices overflowing from unit cube to create a 3-torus like structure.
        """
        idx = np.mod(idx, self.resolution)
        return np.array(idx) * self.scale

    def position_to_index(self, pos):
        """
        Map a point in space to its corresponding grid index.
        Takes into account floating point inaccuracies and wraps positions correctly to a 3-torus.
        """
        idx = (np.array(pos)) / self.scale
        idx = np.mod(idx, self.resolution)
        return np.floor(idx).astype(int)

    def evaluate(self, func, return_flat = False) -> np.array:
        """
        Get values for function evaluated on each point of the Grid
        """

        values = np.array([])

    def __repr__(self):
        return f"Grid3D(resolution={self.resolution})"

if __name__ == '__main__':
    grid = Grid(resolution=32)