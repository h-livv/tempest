#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

grid_configs = [
    {"N": 50,  "dx": 0.1,    "dt": 0.01},
    {"N": 50,  "dx": 0.05,   "dt": 0.005},
    {"N": 50, "dx": 0.025,  "dt": 0.0025},
    {"N": 50, "dx": 0.0125, "dt": 0.00125},

]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=10.0, center_ratio=0.5)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind, operators.gradient]
equations_list = [equations.AdvectionEquation]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 10
STEPS_PER_FRAME = 100   # animation only: steps between plot refreshes
RECORD_INTERVAL = 20    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
