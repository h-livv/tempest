#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions 
#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": 2000,
        "dx": 0.005,
        "dt": 0.002
    }
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05, center_ratio=0.5)]
boundary_functions = [boundaries.periodic]
operators_list = [operators.gradient]
equations_list = [equations.AdvectionEquation(velocity=1.0)]
integrators_list = [integrators.rk4]

FINAL_TIME = 10.0
STEPS_PER_FRAME = 10   # animation only: steps between plot refreshes
RECORD_INTERVAL = 20    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
