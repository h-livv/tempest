import numpy as np

class BoundaryCondition:
    def apply(self, state, axis, parity=None):
        raise NotImplementedError

class Edge(BoundaryCondition):
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='edge')

class Constant(BoundaryCondition):
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='constant')

class Periodic(BoundaryCondition):
    def apply(self, state, axis, parity=None):
        data = state.data if hasattr(state, 'data') else state
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 1)
        return np.pad(data, pad_width=pad_width, mode='wrap')

class Reflect(BoundaryCondition):
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
    def __init__(self, **kwargs):
        """
        kwargs map axis indices (or names like 'x', 'y') to BoundaryConditions.
        For multidimensional consistency, we'll map 'x' to -1, 'y' to -2, etc.
        """
        self.conditions = {}
        axis_map = {'x': -1, 'y': -2, 'z': -3}
        for k, v in kwargs.items():
            if isinstance(k, str) and k in axis_map:
                self.conditions[axis_map[k]] = v
            elif isinstance(k, int):
                self.conditions[k] = v
            else:
                self.conditions[-1] = v # Default to -1

    def __call__(self, state, parity=None):
        padded = state
        # Apply padding sequentially for each mapped axis
        for axis, condition in self.conditions.items():
            padded = condition.apply(padded, axis, parity)
        return padded

    @property
    def __name__(self):
        # Heuristic for old pipeline logs that expect __name__ on the boundary
        if -1 in self.conditions:
            return self.conditions[-1].__class__.__name__.lower()
        return 'mixed'


# =============================================================================
# BACKWARD COMPATIBILITY SHIMS
# =============================================================================

def edge(state, parity=None):
    return Boundary(x=Edge())(state, parity)
edge.__name__ = 'edge'

def constant(state, parity=None):
    return Boundary(x=Constant())(state, parity)
constant.__name__ = 'constant'

def periodic(state, parity=None):
    return Boundary(x=Periodic())(state, parity)
periodic.__name__ = 'periodic'

def reflect(state, parity=None):
    return Boundary(x=Reflect())(state, parity)
reflect.__name__ = 'reflect'