#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 50, "dx": 0.025, "dt": 0.025}
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.GaussianIC(sigma=2.0, num_fields=2, use_L_for_center=True)]
boundary_functions = [boundaries.reflect]
operators_list = [operators.laplacian]
equations_list = [equations.WaveEquation(wave_speed=1.0)]
integrators_list = [integrators.leapfrog]

FINAL_TIME = 10
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
