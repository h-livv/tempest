#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 1.0e-5
    }
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=2.0, use_L_for_center=True)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.rk4, integrators.euler]

FINAL_TIME = 0.5
STEPS_PER_FRAME = 5000   # animation only
RECORD_INTERVAL = 100    # metrics: snapshot every N timesteps

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
