import numpy as np
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (60, 60), "dx": (0.3, 0.3), "dt": 0.02}
]

initial_conditions = [init_conditions.advec_gauss_2d]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind]
equations_list = [equations.advection]
integrators_list = [direct_solvers.lax_w]

# Velocity vector (y-velocity, x-velocity)
coefficients = [np.array([1.0, 1.0]).reshape(2, 1, 1)]

FINAL_TIME = 20.0        # Enough time for the wave to wrap around periodic boundaries
STEPS_PER_FRAME = 2      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
