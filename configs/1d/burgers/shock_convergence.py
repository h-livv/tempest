#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions
from src.numerics import direct_solvers

grid_configs= [
    {"N": 50,  "dx": 1.0,   "dt": 0.02},
    {"N": 50,  "dx": 0.5,   "dt": 0.01},
    {"N": 50, "dx": 0.25,  "dt": 0.005},
    {"N": 50, "dx": 0.125, "dt": 0.0025}
]

initial_conditions = [init_conditions.BurgersTravelingShockIC(nu=0.1)]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]

# We supply an operator even though direct solvers ignore it. RK4 requires it.
operators_list = [operators.upwind]
equations_list = [equations.BurgersEquation(viscosity=0.02)]

# Comparing Lax-Friedrichs, Lax-Wendroff, and RK4
integrators_list = [direct_solvers.lax_f, direct_solvers.lax_w, integrators.rk4]


FINAL_TIME = 10.0
STEPS_PER_FRAME = 25  
RECORD_INTERVAL = 50   

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
