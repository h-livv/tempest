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
        if grid.ndim != 1:
            raise NotImplementedError("GaussianIC currently only supports 1D grids.")
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


def make_ic(fn):
    """Adapts legacy (N, x) initial-condition functions to the new initial_condition(grid) interface.

    Legacy functions have the signature::

        fn(N, x)          ->  np.ndarray   (1-D)
        fn(Ny, Nx, Y, X)  ->  np.ndarray   (2-D)

    The returned callable has the new signature::

        wrapped(grid)  ->  np.ndarray

    and can be passed directly to :attr:`SimulationConfig.initial_condition`.

    The original function is stored on ``wrapped._legacy_fn`` so that
    ``Simulation`` can retrieve it for ``validation.validation()``, which
    still expects the legacy ``(N, x)`` signature.  Attributes copied from
    *fn* (e.g. ``__name__``, ``convergence_order``) are preserved so that
    the pipeline logging and convergence metadata continue to work.

    Parameters
    ----------
    fn:
        A legacy initial-condition callable.

    Returns
    -------
    callable
        A new callable ``wrapped(grid) -> np.ndarray``.
    """
    if isinstance(fn, InitialCondition):
        return fn

    def wrapped(grid):
        if grid.ndim == 1:
            return fn(grid.shape[0], grid.coordinates[0])
        elif grid.ndim == 2:
            return fn(*grid.shape, *grid.coordinates)
        else:
            raise NotImplementedError(
                f"make_ic does not yet support {grid.ndim}-D grids. "
                "Wrap the function manually."
            )

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    wrapped._legacy_fn = fn   # used by Simulation._append_snapshot for validation

    # Copy any metadata attributes (e.g. convergence_order) the original carries
    for attr in ("convergence_order", "spatial_order", "is_direct_solver"):
        if hasattr(fn, attr):
            setattr(wrapped, attr, getattr(fn, attr))

    return wrapped

#Gaussian waves

'''def wave_gauss(N, x):
    center = 0.5 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    init_vel = np.zeros(N)
    
    init_state = np.stack([init_pos, init_vel], axis=0)
    return init_state'''

def wave_gauss(N, x):
    # 1. Extract dx and find the absolute non-drifting center
    dx_extracted = x[1] - x[0]
    L = x.max() + dx_extracted
    center = 0.5 * L

    # 2. Tighten sigma so it has room to propagate without immediate boundary chaos
    sigma = 2.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    init_vel = np.zeros(N)
    
    init_state = np.stack([init_pos, init_vel], axis=0)
    return init_state

def shallow_gauss(N, x):
    center = 0.2 * x.max()

    init_height = 1.0 + np.exp(-0.0001 * (x - center)**2)
    init_vel = np.zeros(N)
    
    init_state = np.stack([init_height, init_vel], axis=0)
    return init_state

def shallow_linear_gauss(N, x):
    # 1. Extract the step size directly from the data array
    dx_extracted = x[1] - x[0]
    
    # 2. Compute the true domain length L
    L = x.max() + dx_extracted
    
    # 3. Set center to exactly 50%
    center = 0.5 * L
    
    # 4. Use a massive sigma and extreme microscopic amplitude for a pristine linear regime
    sigma = 20.0
    init_height = 1.0 + 1e-6 * np.exp(-((x - center)**2) / (2 * sigma**2))
    init_vel = np.zeros(N)
    
    init_state = np.stack([init_height, init_vel], axis=0)
    return init_state

def advec_gauss(N, x):
    center = 0.5 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    
    init_state = np.stack([init_pos], axis=0)
    return init_state

def advec_shifted_gauss(N, x):
    center = 0.25 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    
    return np.stack([init_pos], axis=0)

'''def diff_gauss(N, x):
    center = 0.5 * x.max()
    
    init_temp = np.exp(-0.01 * (x - center)**2)
    
    init_state = np.stack([init_temp], axis=0)
    return init_state'''

def diff_gauss(N, x):
    # 1. Extract the step size directly from the data array
    dx_extracted = x[1] - x[0]
    
    # 2. Compute the true domain length L
    L = x.max() + dx_extracted
    
    # 3. Set a rock-solid center at exactly 50% of the true domain
    center = 0.5 * L
    
    # 4. Use a tighter sigma so the Gaussian decays to 0 before the boundaries
    sigma = 2.0
    init_temp = np.exp(-((x - center)**2) / (2 * sigma**2))
    
    init_state = np.stack([init_temp], axis=0)
    return init_state

#Sharp peaks (Diraq delta function + square shapes)

def shallow_peak(N, x):
    ambient_depth = 1.0
    center_idx = N // 2
    
    init_h = np.ones(N) * ambient_depth

    init_h[center_idx] = 100.0
    
    init_v = np.zeros(N)
    init_state = np.stack([init_h, init_v], axis=0)
    return init_state

def wave_peak(N, x):
    init_pos = np.zeros(N)
    center_idx = N // 2
    init_pos[center_idx] = 2.0
    
    init_vel = np.zeros(N)
    return np.stack([init_pos, init_vel], axis=0)

def wave_square(N, x):
    init_pos = np.where((x > 0.4 * x.max()) & (x < 0.6 * x.max()), 1.0, 0.0)
    init_vel = np.zeros(N)
    
    return np.stack([init_pos, init_vel], axis=0)

def advec_peak(N, x):
    init_pos = np.zeros(N)
    center_idx = N // 2
    init_pos[center_idx] = 1.0
    
    return np.stack([init_pos], axis=0)

def advec_square(N, x):
    init_pos = np.where((x > 0.4 * x.max()) & (x < 0.6 * x.max()), 1.0, 0.0)
    
    return np.stack([init_pos], axis=0)

def diff_peak(N, x):
    init_temp = np.zeros(N)
    center_idx = N // 2
    init_temp[center_idx] = 10.0
    
    return np.stack([init_temp], axis=0)

#A rod heated on one end
def diff_rod(N, x):
    init_temp = np.where(x < 0.2 * x.max(), 0.5, 0.0)
    
    init_state = np.stack([init_temp], axis=0)
    return init_state

#Dam breaking
def shallow_dam(N, x):
    init_h = np.where(x < 0.5 * x.max(), 2.0, 0.2)
    init_v = np.zeros(N)
    
    init_state = np.stack([init_h, init_v], axis=0)
    return init_state

shallow_dam.convergence_order = {"avg_l1": 1.0, "avg_l2": 0.5, "final_l1": 1.0, "final_l2": 0.5}

#Two waves colliding at the center
def shallow_collision(N, x):
    init_h = np.ones(N) * 1.5
    init_v = np.where(x < 0.5 * x.max(), 2.0, -2.0)
    
    init_state = np.stack([init_h, init_v], axis=0)
    return init_state

#Constant for diagnostic purposes
def constant(N, x, num_fields=1, default_val=1.0):

    init_state = np.zeros((num_fields, N))
    init_state[0, :] = default_val 
    return init_state

def burgers_stationary_shock(N, x, nu=0.1, U=1.0):
    """
    Initial condition for the stationary shock of Burgers' equation:
    u(x, 0) = -U * tanh(U * (x - x_0) / (2 * nu))
    where x_0 is the domain center, so the shock is visible in the field.
    """
    x_0 = 0.5 * (x[0] + x[-1])
    u = -U * np.tanh(U * (x - x_0) / (2.0 * nu))
    return np.stack([u], axis=0)

def burgers_traveling_shock(N, x, nu=0.1, u_L=2.0, u_R=1.0, x_0=None):
    """
    Initial condition for the traveling shock of Burgers' equation:
    u(x, 0) = c - ((u_L - u_R) / 2) * tanh(((u_L - u_R) / (4 * nu)) * (x - x_0))
    where c = (u_L + u_R) / 2.
    
    WARNING: Because we are using periodic boundary conditions, it is strongly
    recommended to center x_0 in the middle of the domain and keep the validation
    time t small enough so that the shock does not wrap around and interact with the
    boundary discontinuities.
    """
    if x_0 is None:
        dx = x[1] - x[0]
        L = x.max() + dx
        x_0 = 0.5 * L
    c = 0.5 * (u_L + u_R)
    u = c - 0.5 * (u_L - u_R) * np.tanh(((u_L - u_R) / (4.0 * nu)) * (x - x_0))
    return np.stack([u], axis=0)

def burgers_traveling_smooth(N, x, nu=2.0, u_L=2.0, u_R=1.0, x_0=None):
    """
    A significantly smoothed version of the traveling shock, acting as a gentle curve.
    Achieved by defaulting to a large viscosity (nu).
    """
    return burgers_traveling_shock(N, x, nu=nu, u_L=u_L, u_R=u_R, x_0=x_0)

def diff_gauss_2d(Ny, Nx, Y, X):
    """
    2D Gaussian pulse for diffusion.
    """
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    
    Ly = np.max(Y) + dy
    Lx = np.max(X) + dx
    
    center_y = 0.5 * Ly
    center_x = 0.5 * Lx
    
    sigma = 2.0
    
    init_temp = np.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * sigma**2))
    
    return np.stack([init_temp], axis=0)

def wave_gauss_2d(Ny, Nx, Y, X):
    """
    2D Gaussian pulse for Wave Equation.
    """
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    
    center_y = 0.5 * (np.max(Y) + dy)
    center_x = 0.5 * (np.max(X) + dx)
    
    sigma = 2.0
    init_pos = np.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * sigma**2))
    init_vel = np.zeros_like(init_pos)
    
    return np.stack([init_pos, init_vel], axis=0)

def advec_gauss_2d(Ny, Nx, Y, X):
    """
    2D Gaussian pulse for Advection.
    """
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    
    center_y = 0.5 * (np.max(Y) + dy)
    center_x = 0.5 * (np.max(X) + dx)
    
    sigma = 2.0
    init_pos = np.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * sigma**2))
    
    return np.stack([init_pos], axis=0)
