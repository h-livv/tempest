"""
Tempest boundary conditions and padding strategies.
"""

import numpy as np

class BoundaryCondition:
    """
    Abstract interface for boundary padding strategies.
    
    Think of a BoundaryCondition as the guardrails at the edge of the simulation.
    Before computing finite-difference spatial derivatives, operators need neighboring 
    values at the grid boundary. To avoid out-of-bounds indexing, we apply boundary conditions 
    to append a thin layer of "ghost cells" outside the active grid.
    """
    def apply(self, state, axis, grid_ndim, parity=None):
        """
        Pads the state array with ghost cells along a given axis.
        
        Args:
            state (np.ndarray or Field): Input numerical state data.
            axis (int): Axis index along which to pad.
            grid_ndim (int): Number of spatial axes.
            parity (list[int], optional): Sign transformations for vector elements (reflecting boundaries).
        """
        raise NotImplementedError

    def __call__(self, state, grid_ndim, parity=None):
        """Simplifies calling syntax, applying the condition to all spatial axes of the state."""
        # state is expected to be an ndarray
        axis_names = ['x', 'y', 'z', 'w']
        kwargs = { axis_names[i]: self for i in range(grid_ndim) }
        return Boundary(**kwargs)(state, grid_ndim, parity)


class Edge(BoundaryCondition):
    """
    Zero-gradient boundary condition (Neumann). 
    Ghost cells copy values directly from the outermost boundary cells.
    """
    def apply(self, state, axis, grid_ndim, parity=None):
        pad_width = [(0, 0)] * state.ndim
        pad_width[axis] = (1, 1)
        return np.pad(state, pad_width=pad_width, mode='edge')


class Constant(BoundaryCondition):
    """
    Fixed value boundary condition (Dirichlet). 
    Ghost cells are filled with a constant value (defaults to zero).
    """
    def apply(self, state, axis, grid_ndim, parity=None):
        pad_width = [(0, 0)] * state.ndim
        pad_width[axis] = (1, 1)
        return np.pad(state, pad_width=pad_width, mode='constant')


class Periodic(BoundaryCondition):
    """
    Periodic boundary condition. 
    Ghost cells wrap around to the opposite side of the grid.
    """
    def apply(self, state, axis, grid_ndim, parity=None):
        pad_width = [(0, 0)] * state.ndim
        pad_width[axis] = (1, 1)
        return np.pad(state, pad_width=pad_width, mode='wrap')


class Reflect(BoundaryCondition):
    """
    Reflective boundary condition.
    Values mirror at boundaries. If parity is specified, values are flipped 
    (e.g., velocity components hitting a wall invert sign).
    """
    def apply(self, state, axis, grid_ndim, parity=None):
        pad_width = [(0, 0)] * state.ndim
        pad_width[axis] = (1, 1)
        padded = np.pad(state, pad_width=pad_width, mode='reflect')
        
        if parity is not None:
            component_axes = state.ndim - grid_ndim
            expected_components = 1 if component_axes == 0 else state.shape[0]
            if len(parity) != expected_components:
                raise ValueError(f"Parity array length {len(parity)} does not match expected component count {expected_components}")

            left_slice = [slice(None)] * state.ndim
            right_slice = [slice(None)] * state.ndim
            left_slice[axis] = 0
            right_slice[axis] = -1
            
            if component_axes == 0:
                padded[tuple(left_slice)] *= parity[0]
                padded[tuple(right_slice)] *= parity[0]
            else:
                mask_shape = [len(parity)] + [1] * grid_ndim
                parity_mask = np.asarray(parity).reshape(mask_shape)
                
                padded[tuple(left_slice)] *= parity_mask[..., 0]
                padded[tuple(right_slice)] *= parity_mask[..., 0]
                
        return padded


class Dirichlet(BoundaryCondition):
    """
    Non-zero fixed Dirichlet boundary condition.
    Sets explicit left and right ghost values.
    """
    def __init__(self, left_val, right_val):
        self.left_val = left_val
        self.right_val = right_val
        self.__name__ = 'dirichlet'

    def apply(self, state, axis, grid_ndim, parity=None):
        left_shape = list(state.shape)
        left_shape[axis] = 1
        
        left_arr = np.broadcast_to(self.left_val, left_shape)
        right_arr = np.broadcast_to(self.right_val, left_shape)
        
        return np.concatenate([left_arr, state, right_arr], axis=axis)


class Boundary:
    """
    Composite boundary orchestrator that maps specific boundary conditions to different spatial axes.
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Axis identifiers ('x', 'y', 'z', or ints) mapped to BoundaryCondition objects.
        """
        self.conditions = {}
        axis_map = {'x': -1, 'y': -2, 'z': -3}
        for k, v in kwargs.items():
            if isinstance(k, str) and k in axis_map:
                self.conditions[axis_map[k]] = v
            elif isinstance(k, int):
                self.conditions[k] = v
            else:
                self.conditions[-1] = v # Default to spatial dimension -1

    def __call__(self, state, grid_ndim, parity=None):
        # Tensor layout invariant:
        #
        # Leading axes -> component axes
        # Last grid_ndim axes -> spatial axes
        padded = state
        
        # Apply padding sequentially for each mapped axis
        for axis in range(-grid_ndim, 0):
            if axis in self.conditions:
                condition = self.conditions[axis]
            elif -1 in self.conditions:
                condition = self.conditions[-1] # fallback to default if present
            else:
                continue # if no condition for this axis and no default, skip padding (should rarely happen unless intended)
                
            padded = condition.apply(padded, axis, grid_ndim, parity)
        return padded

    @property
    def __name__(self):
        if -1 in self.conditions:
            return self.conditions[-1].__class__.__name__.lower()
        return 'mixed'


# =============================================================================
# BACKWARD COMPATIBILITY SHIMS
# =============================================================================

def _apply_all_axes(condition_class, state, grid_ndim, parity=None):
    axis_names = ['x', 'y', 'z', 'w']
    kwargs = { axis_names[i]: condition_class() for i in range(grid_ndim) }
    return Boundary(**kwargs)(state, grid_ndim, parity)

def edge(state, grid_ndim, parity=None):
    return _apply_all_axes(Edge, state, grid_ndim, parity)
edge.__name__ = 'edge'

def constant(state, grid_ndim, parity=None):
    return _apply_all_axes(Constant, state, grid_ndim, parity)
constant.__name__ = 'constant'

def periodic(state, grid_ndim, parity=None):
    return _apply_all_axes(Periodic, state, grid_ndim, parity)
periodic.__name__ = 'periodic'

def reflect(state, grid_ndim, parity=None):
    return _apply_all_axes(Reflect, state, grid_ndim, parity)
reflect.__name__ = 'reflect'
