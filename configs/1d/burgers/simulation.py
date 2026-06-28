#Module imports
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

grid_configs = [
    {"N": 50,  "dx": 1.0,   "dt": 0.02}
]


#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.burgers_stationary_shock]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]
operators_list = [operators.upwind]
equations_list = [equations.burgers]
integrators_list = [integrators.rk4]
coefficients = [0.02]

FINAL_TIME = 10.0
STEPS_PER_FRAME = 10   # animation only: steps between plot refreshes
RECORD_INTERVAL = 2    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
