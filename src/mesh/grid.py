import numpy as np

class Grid:
    """
    Generalized dimension-independent Grid.
    """
    def __init__(self, shape, spacing, domain=None):
        """
        shape: tuple of ints (N,) for 1D, (Ny, Nx) for 2D - Number of grid points in each dimension
        spacing: tuple of floats (dx,) for 1D, (dy, dx) for 2D - Spacing of grid
        domain: tuple of tuples ((0, N*dx),) for 1D, ((0, N*dx), (0, N*dy)) for 2D - Physical domain of grid
        """
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self.spacing = tuple(spacing) if isinstance(spacing, (list, tuple)) else (spacing,)
        self.ndim = len(self.shape)
        
        if len(self.spacing) != self.ndim:
            raise ValueError(f"Spacing tuple must match ndim. Got {self.spacing} for ndim {self.ndim}")
            
        if domain is None:
            self.domain = tuple((0, self.shape[i] * self.spacing[i]) for i in range(self.ndim))
        else:
            self.domain = tuple(domain)
            if len(self.domain) != self.ndim:
                raise ValueError(f"Domain tuple must match ndim. Got {self.domain} for ndim {self.ndim}")
            
        # Compute coordinates
        axes_coords = [np.arange(self.shape[i]) * self.spacing[i] + self.domain[i][0] for i in range(self.ndim)]
        
        # Generate the coordinates
        if self.ndim == 1:
            self.coordinates = axes_coords
        else:
            self.coordinates = np.meshgrid(*axes_coords, indexing='ij')

    # Utility functions
    def get_spacing(self, axis):
        return self.spacing[axis]

    def characteristic_spacing(self):
        """Used for CFL calculations, stability analysis, and timestep estimation."""
        return min(self.spacing)

    def mesh_size(self):
        """Used for convergence studies, refinement metrics, and log-log plots. Alias for h()."""
        return min(self.spacing)

    def h(self):
        """Alias for mesh_size()."""
        return self.mesh_size()
