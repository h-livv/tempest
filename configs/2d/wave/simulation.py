from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (60, 60), "dx": (0.3, 0.3), "dt": 0.02}
]

initial_conditions = [init_conditions.GaussianIC(sigma=2.0, num_fields=2)]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.WaveEquation(wave_speed=1.0)]
integrators_list = [integrators.leapfrog]

FINAL_TIME = 10.0        # Wave will propagate, hit reflect boundary, and bounce back
STEPS_PER_FRAME = 5      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
