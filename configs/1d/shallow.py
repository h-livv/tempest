#Module imports
from src import boundaries, operators, equations, integrators, direct_solvers, init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 1000, "dx": 0.5, "dt": 0.05}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.shallow_dam]
boundary_functions = [boundaries.reflect]
operators_list = [operators.central_flux_divergence]
equations_list = [equations.shallow_water]
integrators_list = [direct_solvers.lax_f]
coefficients = [1.0]

FINAL_TIME = 2500
STEPS_PER_FRAME = 50    # animation only
RECORD_INTERVAL = 50    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
