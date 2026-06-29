from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions
# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {
        "N": (100, 100),
        "dx": (0.01, 0.01),
        "dt": 2e-5,  # 2D CFL: α*dt*(1/dx²+1/dy²) = 1.0*2e-5*2e4 = 0.4 < 0.5 ✓
    }
]

initial_conditions = [init_conditions.GaussianIC(sigma=0.05)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.rk4]

FINAL_TIME = 10.0         # Watch diffusion spread to flat state
STEPS_PER_FRAME = 100     # Steps per frame to stay fluid at small dt
RECORD_INTERVAL = 500

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
