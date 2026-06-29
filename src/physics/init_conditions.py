import numpy as np

class InitialCondition:
    """Base class for initial conditions."""
    def __call__(self, grid):
        raise NotImplementedError

class GaussianIC(InitialCondition):
    """A reusable Gaussian initial condition."""
    __name__ = "gaussian"
    
    def __init__(self, center_ratio=0.5, sigma=0.05, amplitude=2.0, num_fields=1, active_field=0, use_L_for_center=False):
        self.center_ratio = center_ratio
        self.sigma = sigma
        self.amplitude = amplitude
        self.num_fields = num_fields
        self.active_field = active_field
        self.use_L_for_center = use_L_for_center

    def __call__(self, grid):
        r2 = 0.0
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            if self.use_L_for_center:
                L = coord.max() + grid.spacing[d]
                center = self.center_ratio * L
            else:
                center = self.center_ratio * coord.max()
            r2 += (coord - center)**2
            
        pos = self.amplitude * np.exp(-r2 / (2 * self.sigma**2))
        state = np.zeros((self.num_fields, *grid.shape))
        state[self.active_field] = pos
        return state

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
        mask = np.ones(grid.shape, dtype=bool)
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            mask &= (coord > self.bounds[0] * coord.max()) & (coord < self.bounds[1] * coord.max())
        pos = np.where(mask, self.high_val, self.low_val)
        state = np.zeros((self.num_fields, *grid.shape))
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
        pos = np.ones(grid.shape) * self.ambient_val
        center_idxs = tuple(int(shape_d * self.index_ratio) for shape_d in grid.shape)
        pos[center_idxs] = self.amplitude
        
        state = np.zeros((self.num_fields, *grid.shape))
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

    def __init__(self, center_ratio=0.2, sigma=1.0, amplitude=2.0, ambient_depth=1.0, use_L_for_center=False):
        self.center_ratio = center_ratio
        self.sigma = sigma
        self.amplitude = amplitude
        self.ambient_depth = ambient_depth
        self.use_L_for_center = use_L_for_center

    def __call__(self, grid):
        r2 = 0.0
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            if self.use_L_for_center:
                L = coord.max() + grid.spacing[d]
                center = self.center_ratio * L
            else:
                center = self.center_ratio * coord.max()
            r2 += (coord - center)**2

        h = self.ambient_depth + self.amplitude * np.exp(-r2 / (2 * self.sigma**2))
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)


class ShallowDamIC(InitialCondition):
    """A dam break initial condition for shallow water."""
    __name__ = "shallow_dam"
    
    # Keeping the original convergence metrics
    convergence_order = {"avg_l1": 1.0, "avg_l2": 0.5, "final_l1": 1.0, "final_l2": 0.5}

    def __init__(self, break_ratio=0.5, h_left=1.0, h_right=0.2):
        self.break_ratio = break_ratio
        self.h_left = h_left
        self.h_right = h_right

    def __call__(self, grid):
        coord_0 = grid.coordinates[0]
        L = coord_0.max()
        h = np.where(coord_0 < self.break_ratio * L, self.h_left, self.h_right)
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)

class ShallowCollisionIC(InitialCondition):
    """Two shallow water flows colliding."""
    __name__ = "shallow_collision"

    def __init__(self, split_ratio=0.5, h_val=0.5, v_left=2.0, v_right=-2.0):
        self.split_ratio = split_ratio
        self.h_val = h_val
        self.v_left = v_left
        self.v_right = v_right

    def __call__(self, grid):
        coord_0 = grid.coordinates[0]
        h = np.ones_like(coord_0) * self.h_val
        v0 = np.where(coord_0 < self.split_ratio * coord_0.max(), self.v_left, self.v_right)
        v_rest = [np.zeros_like(h) for _ in range(grid.ndim - 1)]
        return np.stack([h, v0] + v_rest, axis=0)

class BurgersStationaryShockIC(InitialCondition):
    """Stationary shock for Burgers' equation."""
    __name__ = "burgers_stationary_shock"

    def __init__(self, nu=0.1, U=1.0):
        self.nu = nu
        self.U = U

    def __call__(self, grid):
        coord_0 = grid.coordinates[0]
        x_min = np.min(coord_0)
        x_max = np.max(coord_0)
        x_0 = 0.5 * (x_min + x_max)
        u = -self.U * np.tanh(self.U * (coord_0 - x_0) / (2.0 * self.nu))
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
        coord_0 = grid.coordinates[0]
        x_min = np.min(coord_0)
        x_max = np.max(coord_0)
        
        x_0 = self.x_0
        if x_0 is None:
            L = x_max + grid.spacing[0]
            x_0 = 0.5 * L
        
        c = 0.5 * (self.u_L + self.u_R)
        u = c - 0.5 * (self.u_L - self.u_R) * np.tanh(((self.u_L - self.u_R) / (4.0 * self.nu)) * (coord_0 - x_0))
        return np.stack([u], axis=0)

class ShallowCircularDamIC(InitialCondition):
    """Circular dam break centered in the domain."""
    __name__ = "shallow_circular_dam"

    def __init__(self, break_ratio=0.3, h_inner=2.0, h_outer=0.5):
        self.break_ratio = break_ratio
        self.h_inner = h_inner
        self.h_outer = h_outer

    def __call__(self, grid):
        r2 = 0.0
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            center = 0.5 * (coord.min() + coord.max())
            r2 += (coord - center)**2
        r = np.sqrt(r2)
        domain_len = grid.coordinates[0].max() - grid.coordinates[0].min()
        h = np.where(r < self.break_ratio * (0.5 * domain_len), self.h_inner, self.h_outer)
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)

class ShallowTwoGaussianCollisionIC(InitialCondition):
    """Two Gaussian humps with velocities directing them towards each other."""
    __name__ = "shallow_two_gaussian_collision"

    def __init__(self, bg_depth=1.0, amplitude=0.5, sigma=0.1, v_speed=1.0):
        self.bg_depth = bg_depth
        self.amplitude = amplitude
        self.sigma = sigma
        self.v_speed = v_speed

    def __call__(self, grid):
        coord_0 = grid.coordinates[0]
        L = coord_0.max() - coord_0.min()
        c1 = coord_0.min() + 0.35 * L
        c2 = coord_0.min() + 0.65 * L
        
        r1_sq = (coord_0 - c1)**2
        r2_sq = (coord_0 - c2)**2
        for d in range(1, grid.ndim):
            coord = grid.coordinates[d]
            center = 0.5 * (coord.min() + coord.max())
            r1_sq += (coord - center)**2
            r2_sq += (coord - center)**2
            
        h1 = self.amplitude * np.exp(-r1_sq / (2 * self.sigma**2))
        h2 = self.amplitude * np.exp(-r2_sq / (2 * self.sigma**2))
        h = self.bg_depth + h1 + h2
        
        v0 = (h1 * self.v_speed - h2 * self.v_speed) / self.bg_depth
        v_rest = [np.zeros_like(h) for _ in range(grid.ndim - 1)]
        return np.stack([h, v0] + v_rest, axis=0)

class ShallowRingWaveFocusingIC(InitialCondition):
    """A ring wave that propagates inward towards the center of the domain."""
    __name__ = "shallow_ring_wave_focusing"

    def __init__(self, bg_depth=0.5, amplitude=0.5, ring_radius=0.3, sigma=0.05, v_speed=1.0):
        self.bg_depth = bg_depth
        self.amplitude = amplitude
        self.ring_radius = ring_radius
        self.sigma = sigma
        self.v_speed = v_speed

    def __call__(self, grid):
        r2 = 0.0
        centers = []
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            center = 0.5 * (coord.min() + coord.max())
            centers.append(center)
            r2 += (coord - center)**2
        r = np.sqrt(r2)
        r = np.where(r == 0, 1e-10, r)
        
        domain_len = grid.coordinates[0].max() - grid.coordinates[0].min()
        R0 = self.ring_radius * domain_len
        
        h_profile = self.amplitude * np.exp(-(r - R0)**2 / (2 * self.sigma**2))
        h = self.bg_depth + h_profile
        
        v = []
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            center = centers[d]
            vd = -self.v_speed * ((coord - center) / r) * (h_profile / self.amplitude)
            v.append(vd)
        return np.stack([h] + v, axis=0)

class ShallowRaindropsIC(InitialCondition):
    """Multiple Gaussian drops scattered randomly across the domain."""
    __name__ = "shallow_raindrops"

    def __init__(self, bg_depth=0.50, num_drops=10, amplitude=0.3, sigma=0.05, seed=42):
        self.bg_depth = bg_depth
        self.num_drops = num_drops
        self.amplitude = amplitude
        self.sigma = sigma
        self.seed = seed

    def __call__(self, grid):
        np.random.seed(self.seed)
        h = np.ones(grid.shape) * self.bg_depth
        
        mins = [coord.min() for coord in grid.coordinates]
        maxs = [coord.max() for coord in grid.coordinates]
        lens = [maxs[d] - mins[d] for d in range(grid.ndim)]
        
        for _ in range(self.num_drops):
            drop_center = [mins[d] + np.random.rand() * lens[d] for d in range(grid.ndim)]
            r2 = 0.0
            for d in range(grid.ndim):
                coord = grid.coordinates[d]
                r2 += (coord - drop_center[d])**2
            h += self.amplitude * np.exp(-r2 / (2 * self.sigma**2))
            
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)

class ShallowGaussianIslandsIC(InitialCondition):
    """Multiple Gaussian islands/humps representing land or localized water columns."""
    __name__ = "shallow_gaussian_islands"

    def __init__(self, bg_depth=0.2, num_islands=5, island_height=5.0, sigma=0.1, seed=24):
        self.bg_depth = bg_depth
        self.num_islands = num_islands
        self.island_height = island_height
        self.sigma = sigma
        self.seed = seed

    def __call__(self, grid):
        np.random.seed(self.seed)
        h = np.ones(grid.shape) * self.bg_depth
        
        mins = [coord.min() for coord in grid.coordinates]
        maxs = [coord.max() for coord in grid.coordinates]
        lens = [maxs[d] - mins[d] for d in range(grid.ndim)]
        
        for _ in range(self.num_islands):
            island_center = [mins[d] + np.random.rand() * lens[d] for d in range(grid.ndim)]
            r2 = 0.0
            for d in range(grid.ndim):
                coord = grid.coordinates[d]
                r2 += (coord - island_center[d])**2
            h += self.island_height * np.exp(-r2 / (2 * self.sigma**2))
            
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)

class ShallowMassDropIC(InitialCondition):
    """Initial condition simulating a huge mass dropped into the shallow water bed,
    creating a massive localized depth hump and explosive radially outward velocities."""
    __name__ = "shallow_mass_drop"

    def __init__(self, bg_depth=0.5, drop_amplitude=5.0, sigma=0.1, splash_speed=5.0):
        self.bg_depth = bg_depth
        self.drop_amplitude = drop_amplitude
        self.sigma = sigma
        self.splash_speed = splash_speed

    def __call__(self, grid):
        r2 = 0.0
        centers = []
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            center = 0.5 * (coord.min() + coord.max())
            centers.append(center)
            r2 += (coord - center)**2
        r = np.sqrt(r2)
        r = np.where(r == 0, 1e-10, r)

        # Huge localized water displacement
        h_profile = self.drop_amplitude * np.exp(-r2 / (2 * self.sigma**2))
        h = self.bg_depth + h_profile

        # Explosive radially outward splash velocities
        v = []
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            center = centers[d]
            vd = self.splash_speed * ((coord - center) / r) * (h_profile / self.drop_amplitude)
            v.append(vd)
        return np.stack([h] + v, axis=0)


class LocalizedDamBreakIC(InitialCondition):
    """Localized dam break initial condition."""
    __name__ = "localized_dam_break"

    def __init__(self, high_depth=2.0, low_depth=0.5, center=None, transition_width=0.02, gate_width=0.1, orientation="horizontal"):
        self.high_depth = high_depth
        self.low_depth = low_depth
        self.center = center # list/tuple of ratios, e.g., (0.5, 0.5)
        self.transition_width = transition_width
        self.gate_width = gate_width
        self.orientation = orientation

    def __call__(self, grid):
        if grid.ndim != 2:
            raise NotImplementedError("LocalizedDamBreakIC only supports 2D grids.")
            
        y, x = grid.coordinates
        Ly = y.max() - y.min()
        Lx = x.max() - x.min()
        
        c = self.center if self.center is not None else (0.5, 0.5)
        yc = y.min() + c[0] * Ly
        xc = x.min() + c[1] * Lx
        
        if self.orientation == "horizontal":
            # Dam is horizontal (along x-axis, separating y-domains)
            # Step in y, localized in x
            S = 0.5 * (1.0 - np.tanh((y - yc) / self.transition_width))
            G = np.exp(-((x - xc)**2) / (2.0 * self.gate_width**2))
        else:
            # Dam is vertical (along y-axis, separating x-domains)
            # Step in x, localized in y
            S = 0.5 * (1.0 - np.tanh((x - xc) / self.transition_width))
            G = np.exp(-((y - yc)**2) / (2.0 * self.gate_width**2))
            
        h = self.low_depth + (self.high_depth - self.low_depth) * S * G
        
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)


class CircularDamBreakIC(InitialCondition):
    """Circular dam break with a smooth transition."""
    __name__ = "circular_dam_break"

    def __init__(self, center=None, radius=0.2, high_depth=2.0, low_depth=0.5, transition_width=0.02):
        self.center = center # ratio, e.g., (0.5, 0.5)
        self.radius = radius # ratio of domain length or physical radius
        self.high_depth = high_depth
        self.low_depth = low_depth
        self.transition_width = transition_width

    def __call__(self, grid):
        r2 = 0.0
        domain_len = grid.coordinates[0].max() - grid.coordinates[0].min()
        
        c = self.center if self.center is not None else (0.5,) * grid.ndim
        R = self.radius if self.radius > 1e-4 else 0.2 * domain_len
        # If radius is a small ratio (< 1.0), scale by domain size
        if 0.0 < self.radius <= 1.0:
            R = self.radius * domain_len
            
        for d in range(grid.ndim):
            coord = grid.coordinates[d]
            coord_len = coord.max() - coord.min()
            center_val = coord.min() + c[d] * coord_len
            r2 += (coord - center_val)**2
            
        r = np.sqrt(r2)
        h = self.low_depth + 0.5 * (self.high_depth - self.low_depth) * (1.0 - np.tanh((r - R) / self.transition_width))
        
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)


class ReservoirIC(InitialCondition):
    """Smooth rectangular reservoir initial condition."""
    __name__ = "reservoir"

    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, high_depth=2.0, low_depth=0.5, transition_width=0.02):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.high_depth = high_depth
        self.low_depth = low_depth
        self.transition_width = transition_width

    def __call__(self, grid):
        if grid.ndim != 2:
            raise NotImplementedError("ReservoirIC only supports 2D grids.")
            
        y, x = grid.coordinates
        Ly = y.max() - y.min()
        Lx = x.max() - x.min()
        
        # Parse bounds (defaulting to 0.35 to 0.65 ratios if not set)
        x0 = x.min() + (self.xmin if self.xmin is not None else 0.35) * Lx if (self.xmin is None or self.xmin <= 1.0) else self.xmin
        x1 = x.min() + (self.xmax if self.xmax is not None else 0.65) * Lx if (self.xmax is None or self.xmax <= 1.0) else self.xmax
        y0 = y.min() + (self.ymin if self.ymin is not None else 0.35) * Ly if (self.ymin is None or self.ymin <= 1.0) else self.ymin
        y1 = y.min() + (self.ymax if self.ymax is not None else 0.65) * Ly if (self.ymax is None or self.ymax <= 1.0) else self.ymax
        
        Bx = 0.5 * (np.tanh((x - x0) / self.transition_width) - np.tanh((x - x1) / self.transition_width))
        By = 0.5 * (np.tanh((y - y0) / self.transition_width) - np.tanh((y - y1) / self.transition_width))
        
        h = self.low_depth + (self.high_depth - self.low_depth) * Bx * By
        
        v = [np.zeros_like(h) for _ in range(grid.ndim)]
        return np.stack([h] + v, axis=0)
