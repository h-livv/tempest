#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 2500,  "dx": 1.0,    "dt": 0.25},
    {"N": 5000,  "dx": 0.5,   "dt": 0.0625},
    {"N": 10000, "dx": 0.25,  "dt": 0.015625},
    {"N": 20000, "dx": 0.125, "dt": 0.00390625},
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.diff_gauss]
boundary_functions = [boundaries.periodic]
operators_list = [operators.laplacian]
equations_list = [equations.diffusion]
integrators_list = [integrators.euler, integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 30
STEPS_PER_FRAME = 100   # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
