#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 0.002
    }
]

#Define custom conditions for automated pipeline, as many parameters as required
# Wave equation requires [displacement, velocity]; keep velocity zero at t=0.
WAVE_NUM_FIELDS = 2

initial_conditions = [
    init_conditions.GaussianIC(num_fields=WAVE_NUM_FIELDS),
    init_conditions.SquareIC(num_fields=WAVE_NUM_FIELDS),
    init_conditions.ShiftedGaussianIC(num_fields=WAVE_NUM_FIELDS),
    init_conditions.SineWaveIC(num_fields=WAVE_NUM_FIELDS),
    init_conditions.DoubleGaussianIC(num_fields=WAVE_NUM_FIELDS, speed=0.0),
]
boundary_functions = [boundaries.periodic]
operators_list = [operators.laplacian]
equations_list = [equations.WaveEquation(wave_speed=1.0)]
integrators_list = [integrators.leapfrog]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 10    # animation only
RECORD_INTERVAL = 10    # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
