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
integrators_list = [integrators.leapfrog]
coefficients = [1.0]

FINAL_TIME = 100
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
