import numpy as np
from src.mesh import boundaries
from src.numerics import operators, flux_methods
from src.physics import equations, init_conditions

grid_configs = [
    {
        "N": (30, 30),
        "dx": (0.1, 0.1),
        "dt": 0.01
    }
]

# Using a standard 2D Gaussian IC for a scalar field
initial_conditions = [init_conditions.GaussianIC()]
boundary_functions = [boundaries.edge]
operators_list = [operators.gradient] # For Lax-Wendroff, the operator argument is passed, even though the solver mainly uses flux
equations_list = [equations.BurgersEquation(viscosity=0.01)]
integrators_list = [flux_methods.lax_f]

FINAL_TIME = 10
STEPS_PER_FRAME = 1 
RECORD_INTERVAL = 1
VISUAL_MODE = True
