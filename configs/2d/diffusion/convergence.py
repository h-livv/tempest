# Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions

# Define grid configurations for convergence (dt must scale as dx^2 for diffusion stability)
grid_configs = [
    {"N": (50, 50),   "dx": (0.02, 0.02), "dt": 4.00e-5},
    {"N": (100,100),  "dx": (0.01, 0.01), "dt": 1.00e-5},
    {"N": (200,200),  "dx": (0.005, 0.005), "dt": 2.50e-6 },
    {"N": (400,400),  "dx": (0.0025, 0.0025), "dt": 6.25e-7},
]

# Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.rk4, integrators.euler]

FINAL_TIME = 5.0  # Reduce final time for fast convergence run
STEPS_PER_FRAME = 300   # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
