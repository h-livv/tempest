# Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions

grid_configs = [
    {
        "N": 2000,
        "dx": 0.005,
        "dt": 0.002,
    }
]

initial_conditions = [
    init_conditions.GaussianIC(),
    init_conditions.SquareIC(),
    init_conditions.ShiftedGaussianIC(),
    init_conditions.SineWaveIC(),
    init_conditions.SpikeIC(),
    init_conditions.DoubleGaussianIC(),
]
boundary_functions = [boundaries.periodic]
operators_list = [operators.gradient]
equations_list = [equations.AdvectionEquation(velocity=1.0)]
integrators_list = [flux_methods.lax_f]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 10
RECORD_INTERVAL = 10
VISUAL_MODE = False
