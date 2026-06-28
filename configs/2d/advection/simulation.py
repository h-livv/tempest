import numpy as np
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (60, 60), "dx": (0.3, 0.3), "dt": 0.02}
]

initial_conditions = [init_conditions.GaussianIC(sigma=2.0)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind]
equations_list = [equations.AdvectionEquation(velocity=np.array([1.0, 1.0)]
integrators_list = [direct_solvers.lax_w]

# Velocity vector (y-velocity, x-velocity)
).reshape(2, 1, 1)]

FINAL_TIME = 10.0        # Enough time for the wave to wrap around periodic boundaries
STEPS_PER_FRAME = 2      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
