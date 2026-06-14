#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 5000, "dx": 0.05, "dt": 0.005}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.advec_gauss, init_conditions.advec_square]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind]
equations_list = [equations.advection]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 50
STEPS_PER_FRAME = 100   # animation only: steps between plot refreshes
RECORD_INTERVAL = 20    # metrics: snapshot every N timesteps (fixed for all runs)