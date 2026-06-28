import numpy as np
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions
# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (100,100),
     "dx": (0.01,0.01),
     "dt": 2.83e-3}
]

initial_conditions = [init_conditions.GaussianIC(sigma=0.05)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.gradient]
equations_list = [equations.AdvectionEquation(velocity=np.array([1.0, 1.0]).reshape(2, 1, 1))]
integrators_list = [integrators.rk4]

# Velocity vector (y-velocity, x-velocity)


FINAL_TIME = 10.0        # Enough time for the wave to wrap around periodic boundaries
STEPS_PER_FRAME = 5      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
