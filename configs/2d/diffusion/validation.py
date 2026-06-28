# Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions

# Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": (100, 100),
        "dx": (0.01, 0.01),
        "dt": 1e-5,
    }
]

# Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.rk4, integrators.euler]

FINAL_TIME = 5.0         # Fast diffusion to flat state
STEPS_PER_FRAME = 3      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
