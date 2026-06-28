#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 0.002
    }
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.ShallowDamIC()]
boundary_functions = [boundaries.edge]
operators_list = [operators.central_flux_divergence]
equations_list = [equations.ShallowWaterEquation()]
integrators_list = [flux_methods.lax_w]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 50    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
