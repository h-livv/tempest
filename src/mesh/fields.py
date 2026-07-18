"""
Tempest physical simulation field wrappers.
"""

import numpy as np

class Field:
    """
    Base class for physical state fields spanning a spatial Grid.
    
    Think of a Field as the physical quantities (like density, velocity, or temperature)
    distributed across our coordinate ruler (the Grid).
    It wraps a NumPy array and enforces dimensional consistency checks, ensuring 
    that numerical arrays have shapes matching the grid coordinate specifications.
    
    This class supports overloaded basic arithmetic operators (+, -, *, /) to write 
    numerical PDE equations in clean, mathematical syntax.
    
    Attributes:
        grid (Grid): The Grid object representing the domain coordinates.
        data (np.ndarray): The raw array storing state values.
    """
    
    def __init__(self, grid, data=None):
        """
        Creates a Field over a specified grid.
        
        Args:
            grid (Grid): Spatial grid description.
            data (np.ndarray, optional): Spatial values. If None, initialized to zeroes.
        """
        self.grid = grid
        self._flow_velocity_fn = None
        self._surface_scalar_fn = None
        if data is None:
            self.data = np.zeros(self._get_expected_shape())
        else:
            expected_shape = self._get_expected_shape()
            self.data = np.asarray(data)
            if self.data.shape != expected_shape:
                raise ValueError(f"Expected data shape {expected_shape}, got {self.data.shape}")

    @property
    def has_flow(self):
        """Whether this field carries a horizontal velocity for flow visualization."""
        return self._flow_velocity_fn is not None or self._native_has_flow()

    @property
    def flow_style(self):
        """Plot style for the flow panel: 'streamplot' for derived, 'quiver' for native."""
        return 'streamplot' if self._flow_velocity_fn is not None else 'quiver'

    def bind_velocity(self, fn):
        """Attach a callable ``(data) -> (u, v)`` for derived velocity fields."""
        self._flow_velocity_fn = fn
        return self

    def bind_surface_scalar(self, fn):
        """Attach a callable ``(data) -> array`` for the 3D surface plot scalar."""
        self._surface_scalar_fn = fn
        return self

    def get_scalar(self, data=None):
        """Primary scalar component for heatmap plots."""
        data = self.data if data is None else data
        if data.ndim > self.ndim:
            return data[0]
        return data

    def get_surface_scalar(self, data=None):
        """Scalar field for the 3D surface plot (defaults to ``get_scalar``)."""
        data = self.data if data is None else data
        if self._surface_scalar_fn is not None:
            return self._surface_scalar_fn(data)
        return self.get_scalar(data)

    def get_velocity(self, data=None):
        """Return ``(u, v)`` velocity components, or ``None`` if unavailable."""
        data = self.data if data is None else data
        if self._flow_velocity_fn is not None:
            return self._flow_velocity_fn(data)
        return self._native_velocity(data)

    def _native_has_flow(self):
        return False

    def _native_velocity(self, data):
        return None
                
    def _get_expected_shape(self):
        """Must be implemented by subclasses to define shape rules based on Grid dimensions."""
        raise NotImplementedError
        
    @property
    def ndim(self):
        """Returns the spatial dimensionality of the underlying grid."""
        return self.grid.ndim
        
    @property
    def shape(self):
        """Returns the complete shape of the field's data array."""
        return self.data.shape

    def copy(self):
        """Returns a deep copy of this Field with duplicate state data."""
        return self.__class__(self.grid, self.data.copy())

    def __array__(self, dtype=None, copy=None):
        """Supports implicit conversions to standard NumPy arrays for mathematical operations."""
        if copy:
            return np.array(self.data, dtype=dtype, copy=True)
        return np.asarray(self.data, dtype=dtype)
        
    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        if isinstance(other, Field):
            return self.__class__(self.grid, self.data + other.data)
        return self.__class__(self.grid, self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Field):
            return self.__class__(self.grid, self.data - other.data)
        return self.__class__(self.grid, self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Field):
            return self.__class__(self.grid, other.data - self.data)
        return self.__class__(self.grid, other - self.data)

    def __mul__(self, other):
        if isinstance(other, Field):
            return self.__class__(self.grid, self.data * other.data)
        return self.__class__(self.grid, self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Field):
            return self.__class__(self.grid, self.data / other.data)
        return self.__class__(self.grid, self.data / other)


class ScalarField(Field):
    """
    A field representing a single scalar value per grid point (e.g., temperature, density).
    
    The raw data shape exactly matches grid.shape.
    """
    
    @property
    def rank(self):
        return "scalar"
        
    @property
    def components(self):
        """A scalar field has only 1 component per spatial point."""
        return 1

    def _get_expected_shape(self):
        return self.grid.shape

    def _native_has_flow(self):
        return False


class VectorField(Field):
    """
    A field representing multiple directional components per grid point (e.g., fluid velocity).
    
    The raw data shape is (components, spatial_axes...).
    """
    
    def __init__(self, grid, data=None):
        self.grid = grid
        self._flow_velocity_fn = None
        self._surface_scalar_fn = None
        if data is None:
            self.data = np.zeros((self.grid.ndim,) + self.grid.shape)
        else:
            self.data = np.asarray(data)
            if self.data.shape[1:] != self.grid.shape:
                raise ValueError(f"Expected spatial shape {self.grid.shape}, got {self.data.shape}")

    @property
    def rank(self):
        return "vector"
        
    @property
    def components(self):
        """Returns the number of vector components (e.g., Vx, Vy)."""
        return self.data.shape[0] if self.data is not None else self.grid.ndim

    def _get_expected_shape(self):
        return (self.grid.ndim,) + self.grid.shape

    def _native_has_flow(self):
        # 3+ components: e.g. shallow water [h, vy, vx]
        return self.components >= 3

    def _native_velocity(self, data):
        if self.components >= 3:
            # [h, vy, vx] — index 1 = v, index 2 = u
            return data[2], data[1]
        return None
