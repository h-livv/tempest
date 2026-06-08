#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 1000, "dx": 0.1,    "dt": 0.01},
    {"N": 2000, "dx": 0.05,   "dt": 0.005},
    {"N": 4000, "dx": 0.025,  "dt": 0.0025},
    {"N": 8000, "dx": 0.0125, "dt": 0.00125},
    {"N": 16000, "dx": 0.00625, "dt": 0.000625},
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.wave_gauss]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.wave]
integrators_list = [integrators.leapfrog]
coefficients = [1.0]

FINAL_TIME = 10
STEPS_PER_FRAME = 100    # animation only
RECORD_INTERVAL = 50    # metrics: snapshot every N timesteps