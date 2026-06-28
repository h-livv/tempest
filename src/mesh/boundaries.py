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
    def apply(self, state, axis, parity=None):
        """
        Pads the state array with ghost cells along a given axis.
        
        Args:
            state (np.ndarray or Field): Input numerical state data.
            axis (int): Axis index along which to pad.
            parity (list[int], optional): Sign transformations for vector elements (reflecting boundaries).
        """
        raise NotImplementedError

    def __call__(self, state, parity=None):
        """Simplifies calling syntax, applying the condition to all spatial axes of the state."""
        if hasattr(state, 'grid'):
            ndim = state.grid.ndim
        else:
            # Mirror _apply_all_axes: if state's first axis matches len(parity), it's a system
            is_system = (parity is not None and state.ndim > 1 and state.shape[0] == len(parity))
            ndim = state.ndim - 1 if is_system else state.ndim
        axis_names = ['x', 'y', 'z', 'w']
        kwargs = { axis_names[i]: self for i in range(ndim) }
        return Boundary(**kwargs)(state, parity)


class Edge(BoundaryCondition):
    """
    Zero-gradient boundary condition (Neumann). 
    Ghost cells copy values directly from the outermost boundary cells.
    """
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='edge')


class Constant(BoundaryCondition):
    """
    Fixed value boundary condition (Dirichlet). 
    Ghost cells are filled with a constant value (defaults to zero).
    """
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='constant')


class Periodic(BoundaryCondition):
    """
    Periodic boundary condition. 
    Ghost cells wrap around to the opposite side of the grid.
    """
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='wrap')


class Reflect(BoundaryCondition):
    """
    Reflective boundary condition.
    Values mirror at boundaries. If parity is specified, values are flipped 
    (e.g., velocity components hitting a wall invert sign).
    """
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        padded = np.pad(data, pad_width=pad_width, mode='edge')
        
        if parity is not None:
            left_slice = [slice(None)] * data.ndim
            right_slice = [slice(None)] * data.ndim
            left_slice[axis] = 0
            right_slice[axis] = -1
            
            if data.ndim == 1:
                padded[tuple(left_slice)] *= parity[0]
                padded[tuple(right_slice)] *= parity[0]
            else:
                mask_shape = [len(parity)] + [1] * (data.ndim - 1)
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

    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        left_shape = list(data.shape)
        left_shape[axis] = 1
        
        left_arr = np.broadcast_to(self.left_val, left_shape)
        right_arr = np.broadcast_to(self.right_val, left_shape)
        
        return np.concatenate([left_arr, data, right_arr], axis=axis)


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

    def __call__(self, state, parity=None):
        padded = state
        # Apply padding sequentially for each mapped axis
        for axis, condition in self.conditions.items():
            padded = condition.apply(padded, axis, parity)
        return padded

    @property
    def __name__(self):
        if -1 in self.conditions:
            return self.conditions[-1].__class__.__name__.lower()
        return 'mixed'


# =============================================================================
# BACKWARD COMPATIBILITY SHIMS
# =============================================================================

def _apply_all_axes(condition_class, state, parity=None):
    if hasattr(state, 'grid'):
        ndim = state.grid.ndim
    else:
        is_system = (parity is not None and state.shape[0] == len(parity))
        ndim = state.ndim - 1 if is_system else state.ndim
    axis_names = ['x', 'y', 'z', 'w']
    kwargs = { axis_names[i]: condition_class() for i in range(ndim) }
    return Boundary(**kwargs)(state, parity)

def edge(state, parity=None):
    return _apply_all_axes(Edge, state, parity)
edge.__name__ = 'edge'

def constant(state, parity=None):
    return _apply_all_axes(Constant, state, parity)
constant.__name__ = 'constant'

def periodic(state, parity=None):
    return _apply_all_axes(Periodic, state, parity)
periodic.__name__ = 'periodic'

def reflect(state, parity=None):
    return _apply_all_axes(Reflect, state, parity)
reflect.__name__ = 'reflect'
