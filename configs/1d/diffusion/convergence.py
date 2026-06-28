#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 50,  "dx": 1.0,    "dt": 0.25},
    {"N": 50,  "dx": 0.5,   "dt": 0.0625},
    {"N": 50, "dx": 0.25,  "dt": 0.015625},
    {"N": 50, "dx": 0.125, "dt": 0.00390625},
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=2.0, use_L_for_center=True)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.euler, integrators.rk4]

FINAL_TIME = 10
STEPS_PER_FRAME = 300   # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
