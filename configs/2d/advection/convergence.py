# Module imports
import numpy as np
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions

# Define grid configurations for convergence
grid_configs = [
    {"N": (50, 50),   "dx": (0.02, 0.02), "dt": 5.66e-3},
    {"N": (100,100),  "dx": (0.01, 0.01), "dt": 2.83e-3},
    {"N": (200,200),  "dx": (0.005, 0.005), "dt": 1.41e-3},
    {"N": (400,400),  "dx": (0.0025, 0.0025), "dt": 0.71e-3},
]

# Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind]
equations_list = [equations.AdvectionEquation(velocity=np.array([1.0, 1.0]).reshape(2, 1, 1))]
integrators_list = [flux_methods.lax_w]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 10   # animation only: steps between plot refreshes
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
