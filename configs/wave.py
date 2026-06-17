#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 4000, "dx": 0.025, "dt": 0.025},   # CFL = 1.0
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.wave_gauss]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.wave]
integrators_list = [integrators.rk4, integrators.euler, integrators.leapfrog]
coefficients = [1.0]

FINAL_TIME = 5000
STEPS_PER_FRAME = 50    # animation only
RECORD_INTERVAL = 25    # metrics: snapshot every N timesteps