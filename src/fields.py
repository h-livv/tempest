import numpy as np

class Field:
    """
    Base class for generic physical fields spanning a Grid.
    """
    def __init__(self, grid, data=None):
        self.grid = grid
        if data is None:
            self.data = np.zeros(self._get_expected_shape())
        else:
            expected_shape = self._get_expected_shape()
            self.data = np.asarray(data)
            if self.data.shape != expected_shape:
                raise ValueError(f"Expected data shape {expected_shape}, got {self.data.shape}")
                
    def _get_expected_shape(self):
        raise NotImplementedError
        
    @property
    def ndim(self):
        """Returns the spatial dimensionality of the field (from the grid)."""
        return self.grid.ndim
        
    def copy(self):
        return self.__class__(self.grid, self.data.copy())

    def __array__(self, dtype=None, copy=None):
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
    """A scalar field (e.g., temperature, density) across the grid."""
    def _get_expected_shape(self):
        return self.grid.shape

class VectorField(Field):
    """A vector field (e.g., velocity) across the grid. The vector components are along axis 0."""
    def _get_expected_shape(self):
        return (self.grid.ndim,) + self.grid.shape
