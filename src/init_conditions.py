import numpy as np

class InitialCondition:
    """Base class for initial conditions."""
    def __call__(self, grid):
        raise NotImplementedError

class GaussianIC(InitialCondition):
    """A reusable Gaussian initial condition."""
    __name__ = "gaussian"
    
    def __init__(self, center_ratio=0.5, sigma=10.0, amplitude=1.0, num_fields=1, active_field=0, use_L_for_center=False):
        self.center_ratio = center_ratio
        self.sigma = sigma
        self.amplitude = amplitude
        self.num_fields = num_fields
        self.active_field = active_field
        self.use_L_for_center = use_L_for_center

    def __call__(self, grid):
        if grid.ndim == 1:
            x = grid.coordinates[0]
            if self.use_L_for_center:
                L = x.max() + grid.spacing[0]
                center = self.center_ratio * L
            else:
                center = self.center_ratio * x.max()
            pos = self.amplitude * np.exp(-((x - center)**2) / (2 * self.sigma**2))
            state = np.zeros((self.num_fields, *x.shape))
            state[self.active_field] = pos
            return state
        elif grid.ndim == 2:
            Y, X = grid.coordinates
            dy, dx = grid.spacing
            if self.use_L_for_center:
                Ly = Y.max() + dy
                Lx = X.max() + dx
                center_y = self.center_ratio * Ly
                center_x = self.center_ratio * Lx
            else:
                center_y = self.center_ratio * Y.max()
                center_x = self.center_ratio * X.max()
            pos = self.amplitude * np.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * self.sigma**2))
            state = np.zeros((self.num_fields, *Y.shape))
            state[self.active_field] = pos
            return state
        else:
            raise NotImplementedError("GaussianIC currently only supports 1D and 2D grids.")

class SquareIC(InitialCondition):
    """A reusable square pulse initial condition."""
    __name__ = "square"

    def __init__(self, bounds=(0.4, 0.6), high_val=1.0, low_val=0.0, num_fields=1, active_field=0):
        self.bounds = bounds
        self.high_val = high_val
        self.low_val = low_val
        self.num_fields = num_fields
        self.active_field = active_field

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("SquareIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        
        pos = np.where((x > self.bounds[0] * x.max()) & (x < self.bounds[1] * x.max()), self.high_val, self.low_val)
        state = np.zeros((self.num_fields, *x.shape))
        state[self.active_field] = pos
        return state

class SpikeIC(InitialCondition):
    """A point-source spike (Dirac delta-like) initial condition."""
    __name__ = "spike"

    def __init__(self, index_ratio=0.5, amplitude=1.0, ambient_val=0.0, num_fields=1, active_field=0):
        self.index_ratio = index_ratio
        self.amplitude = amplitude
        self.ambient_val = ambient_val
        self.num_fields = num_fields
        self.active_field = active_field

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("SpikeIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        N = x.shape[0]
        center_idx = int(N * self.index_ratio)
        
        pos = np.ones(N) * self.ambient_val
        pos[center_idx] = self.amplitude
        
        state = np.zeros((self.num_fields, N))
        state[self.active_field] = pos
        return state

class ConstantIC(InitialCondition):
    """A constant initial condition."""
    __name__ = "constant"

    def __init__(self, val=1.0, num_fields=1, active_field=0):
        self.val = val
        self.num_fields = num_fields
        self.active_field = active_field

    def __call__(self, grid):
        state = np.zeros((self.num_fields, *grid.shape))
        state[self.active_field] = self.val
        return state

class ShallowGaussianIC(InitialCondition):
    """A Gaussian wave on top of an ambient fluid depth."""
    __name__ = "shallow_gaussian"

    def __init__(self, center_ratio=0.2, sigma=70.710678, amplitude=1.0, ambient_depth=1.0, use_L_for_center=False):
        self.center_ratio = center_ratio
        self.sigma = sigma
        self.amplitude = amplitude
        self.ambient_depth = ambient_depth
        self.use_L_for_center = use_L_for_center

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("ShallowGaussianIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        if self.use_L_for_center:
            L = x.max() + grid.spacing[0]
            center = self.center_ratio * L
        else:
            center = self.center_ratio * x.max()

        h = self.ambient_depth + self.amplitude * np.exp(-((x - center)**2) / (2 * self.sigma**2))
        v = np.zeros_like(h)
        return np.stack([h, v], axis=0)

class ShallowDamIC(InitialCondition):
    """A dam break initial condition for shallow water."""
    __name__ = "shallow_dam"
    
    # Keeping the original convergence metrics
    convergence_order = {"avg_l1": 1.0, "avg_l2": 0.5, "final_l1": 1.0, "final_l2": 0.5}

    def __init__(self, break_ratio=0.5, h_left=2.0, h_right=0.2):
        self.break_ratio = break_ratio
        self.h_left = h_left
        self.h_right = h_right

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("ShallowDamIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        h = np.where(x < self.break_ratio * x.max(), self.h_left, self.h_right)
        v = np.zeros_like(h)
        return np.stack([h, v], axis=0)

class ShallowCollisionIC(InitialCondition):
    """Two shallow water flows colliding."""
    __name__ = "shallow_collision"

    def __init__(self, split_ratio=0.5, h_val=1.5, v_left=2.0, v_right=-2.0):
        self.split_ratio = split_ratio
        self.h_val = h_val
        self.v_left = v_left
        self.v_right = v_right

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("ShallowCollisionIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        h = np.ones_like(x) * self.h_val
        v = np.where(x < self.split_ratio * x.max(), self.v_left, self.v_right)
        return np.stack([h, v], axis=0)

class BurgersStationaryShockIC(InitialCondition):
    """Stationary shock for Burgers' equation."""
    __name__ = "burgers_stationary_shock"

    def __init__(self, nu=0.1, U=1.0):
        self.nu = nu
        self.U = U

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("BurgersStationaryShockIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        x_0 = 0.5 * (x[0] + x[-1])
        u = -self.U * np.tanh(self.U * (x - x_0) / (2.0 * self.nu))
        return np.stack([u], axis=0)

class BurgersTravelingShockIC(InitialCondition):
    """Traveling shock for Burgers' equation."""
    __name__ = "burgers_traveling_shock"

    def __init__(self, nu=0.1, u_L=2.0, u_R=1.0, x_0=None):
        self.nu = nu
        self.u_L = u_L
        self.u_R = u_R
        self.x_0 = x_0

    def __call__(self, grid):
        if grid.ndim != 1:
            raise NotImplementedError("BurgersTravelingShockIC currently only supports 1D grids.")
        x = grid.coordinates[0]
        x_0 = self.x_0
        if x_0 is None:
            L = x.max() + grid.spacing[0]
            x_0 = 0.5 * L
        
        c = 0.5 * (self.u_L + self.u_R)
        u = c - 0.5 * (self.u_L - self.u_R) * np.tanh(((self.u_L - self.u_R) / (4.0 * self.nu)) * (x - x_0))
        return np.stack([u], axis=0)
