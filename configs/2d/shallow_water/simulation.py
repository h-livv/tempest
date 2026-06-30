import numpy as np
from src.mesh import boundaries
from src.numerics import operators, flux_methods
from src.physics import equations, init_conditions

grid_configs = [
    {
        "N": (100, 100),
        "dx": (0.1, 0.1),
        "dt": 0.01
    }
]

initial_conditions = [init_conditions.ShallowGaussianIC()]
boundary_functions = [boundaries.reflect]
operators_list = [operators.central_flux_divergence]
equations_list = [equations.ShallowWaterEquation()]
integrators_list = [flux_methods.lax_f]

FINAL_TIME = 3
STEPS_PER_FRAME = 3
RECORD_INTERVAL = 1
VISUAL_MODE = True
