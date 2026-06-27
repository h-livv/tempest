#Module imports
from src import boundaries, operators, equations, integrators, init_conditions, direct_solvers

grid_configs = [
    {"N": 250,  "dx": 1.0,   "dt": 0.02},
    {"N": 500,  "dx": 0.5,   "dt": 0.01},
    {"N": 1000, "dx": 0.25,  "dt": 0.005},
    {"N": 2000, "dx": 0.125, "dt": 0.0025}
]

initial_conditions = [init_conditions.burgers_traveling_smooth]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]

operators_list = [operators.upwind]
equations_list = [equations.burgers]

# Comparing Lax-Friedrichs, Lax-Wendroff, and RK4
integrators_list = [direct_solvers.lax_f, direct_solvers.lax_w, integrators.rk4]

coefficients = [2.0] # Large viscosity to ensure a very smooth curve

FINAL_TIME = 10.0
STEPS_PER_FRAME = 25  
RECORD_INTERVAL = 50   

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
