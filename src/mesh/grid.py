"""
Tempest spatial discretization grid.
"""

import numpy as np

class Grid:
    """
    Generalized dimension-independent Grid.
    
    Think of the Grid as the coordinate ruler of the simulation domain. 
    It defines the discretised spatial coordinates where physical field quantities live.
    It does not store physical state data (that is the role of Field); instead, 
    it defines the geometry, axes, and mesh spacings needed by spatial operators 
    to compute numerical derivatives.
    
    Attributes:
        shape (tuple[int, ...]): Number of grid points along each spatial axis (e.g., (Nx,) in 1D, (Ny, Nx) in 2D).
        spacing (tuple[float, ...]): Grid step size (spacing) along each axis (e.g., (dx,) in 1D, (dy, dx) in 2D).
        ndim (int): Dimensionality of the grid (inferred from shape).
        domain (tuple[tuple[float, float], ...]): Physical range of the grid boundary along each axis.
        coordinates (list[np.ndarray] or list of grids): Explicit mesh coordinate values.
    """
    
    def __init__(self, shape, spacing, domain=None):
        """
        Initializes the spatial grid geometry.
        
        Args:
            shape (tuple or list or int): Number of grid points.
            spacing (tuple or list or float): Spatial grid spacing.
            domain (tuple of tuples, optional): physical bounds ((xmin, xmax), ...). 
                                               If None, defaults to starts at 0 to shape * spacing.
        """
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self.spacing = tuple(spacing) if isinstance(spacing, (list, tuple)) else (spacing,)
        self.ndim = len(self.shape)
        
        if len(self.spacing) != self.ndim:
            raise ValueError(f"Spacing tuple must match ndim. Got {self.spacing} for ndim {self.ndim}")
            
        if domain is None:
            self.domain = tuple((0.0, self.shape[i] * self.spacing[i]) for i in range(self.ndim))
        else:
            self.domain = tuple(domain)
            if len(self.domain) != self.ndim:
                raise ValueError(f"Domain tuple must match ndim. Got {self.domain} for ndim {self.ndim}")
            
        # Compute coordinate arrays for each axis
        axes_coords = [np.arange(self.shape[i]) * self.spacing[i] + self.domain[i][0] for i in range(self.ndim)]
        
        # Meshgrid indexing 'ij' ensures matrix-style alignment (first axis rows/y, second axis cols/x)
        if self.ndim == 1:
            self.coordinates = axes_coords
        else:
            self.coordinates = np.meshgrid(*axes_coords, indexing='ij')

    def get_spacing(self, axis):
        """Returns the grid spacing (e.g. dx, dy) along the requested axis index."""
        return self.spacing[axis]

    def characteristic_spacing(self):
        """
        Used for CFL stability conditions, stability analysis, and timestep bounds.
        Returns the minimum spatial spacing across all dimensions.
        """
        return min(self.spacing)

    def mesh_size(self):
        """
        Used for grid convergence studies and refinement analysis.
        Returns the characteristic spatial resolution step.
        """
        return min(self.spacing)

    def h(self):
        """Mathematical alias for mesh_size()."""
        return self.mesh_size()
