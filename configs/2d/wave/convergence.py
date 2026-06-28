# Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions

# Define grid configurations for convergence (CFL constrained: dt <= dx / (c * sqrt(2)))
grid_configs = [
    {"N": (50, 50),   "dx": (0.02, 0.02), "dt": 7.07e-3},
    {"N": (100,100),  "dx": (0.01, 0.01), "dt": 3.54e-3},
    {"N": (200,200),  "dx": (0.005, 0.005), "dt": 1.77e-3},
    {"N": (400,400),  "dx": (0.0025, 0.0025), "dt": 8.84e-4},
]

# Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05, num_fields=2)]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.WaveEquation(wave_speed=1.0)]
integrators_list = [integrators.leapfrog]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
