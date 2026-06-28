# Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions

# Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": (100,100),
        "dx": (0.01,0.01),
        "dt": 3.54e-3
    }
]

# Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=0.05, num_fields=2)]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.WaveEquation(wave_speed=1.0)]
integrators_list = [integrators.leapfrog]

FINAL_TIME = 5.0        # Wave will propagate, hit reflect boundary, and bounce back
STEPS_PER_FRAME = 5      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
