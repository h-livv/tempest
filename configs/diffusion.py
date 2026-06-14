#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 5000, "dx": 0.05, "dt": 0.001}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.diff_gauss]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.diffusion]
integrators_list = [integrators.rk4, integrators.euler]
coefficients = [1.0]

FINAL_TIME = 30
STEPS_PER_FRAME = 300   # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps