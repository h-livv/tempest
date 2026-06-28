#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 50, "dx": 0.1,    "dt": 0.01},
    {"N": 50, "dx": 0.05,   "dt": 0.005},
    {"N": 50, "dx": 0.025,  "dt": 0.0025},
    {"N": 50, "dx": 0.0125, "dt": 0.00125},
    {"N": 50, "dx": 0.00625, "dt": 0.000625},
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=2.0, num_fields=2, use_L_for_center=True)]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.wave]
integrators_list = [integrators.leapfrog]
coefficients = [1.0]

FINAL_TIME = 10
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
