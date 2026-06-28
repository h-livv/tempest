#Module imports
from src import boundaries, operators, equations, integrators, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 50, "dx": 0.05, "dt": 0.001}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=2.0, use_L_for_center=True)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.diffusion]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 10
STEPS_PER_FRAME = 300   # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
