#Module imports
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 50, "dx": 0.05, "dt": 0.005}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.advec_gauss]
boundary_functions = [boundaries.periodic]
operators_list = [operators.upwind]
equations_list = [equations.AdvectionEquation]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 10
STEPS_PER_FRAME = 100   # animation only: steps between plot refreshes
RECORD_INTERVAL = 20    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
