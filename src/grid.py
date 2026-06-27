import numpy as np

class Grid:
    """
    Generalized dimension-independent Grid.
    """
    def __init__(self, shape, spacing, domain=None):
        """
        shape: tuple of ints (e.g., (100,) for 1D, (Ny, Nx) for 2D)
        spacing: tuple of floats (e.g., (dx,) for 1D, (dy, dx) for 2D)
        domain: tuple of tuples (e.g., ((0, Lx),) for 1D)
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
        
        if self.ndim == 1:
            self.coordinates = axes_coords
        else:
            # indexing='ij' ensures matrix indexing matches shape (Ny, Nx)
            self.coordinates = np.meshgrid(*axes_coords, indexing='ij')

    def get_spacing(self, axis):
        return self.spacing[axis]
