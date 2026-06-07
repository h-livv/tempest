#Module imports
from Core import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 500, "dx": 1.0, "dt": 0.05}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.wave_gauss]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.wave]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 2500
STEPS_PER_FRAME = 100