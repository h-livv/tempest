#Module imports
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

grid_configs = [
    {"N": 250,  "dx": 1.0,   "dt": 0.02},
    {"N": 500,  "dx": 0.5,   "dt": 0.01},
    {"N": 1000, "dx": 0.25,  "dt": 0.005},
    {"N": 2000, "dx": 0.125, "dt": 0.0025}
]


#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.burgers_traveling_shock, init_conditions.burgers_stationary_shock]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]
operators_list = [operators.upwind]
equations_list = [equations.burgers]
integrators_list = [integrators.rk4]
coefficients = [0.02]

FINAL_TIME = 10.0
STEPS_PER_FRAME = 1   # animation only: steps between plot refreshes
RECORD_INTERVAL = 2    # metrics: snapshot every N timesteps (fixed for all runs)