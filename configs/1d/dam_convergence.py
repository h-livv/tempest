from src import equations, boundaries, operators, integrators, direct_solvers, init_conditions

# Convergence Configuration
grid_configs = [
    {"N": 500, "dx": 1.0, "dt": 0.02},
    {"N": 1000, "dx": 0.5, "dt": 0.01},
    {"N": 2000, "dx": 0.25, "dt": 0.005},
    {"N": 4000, "dx": 0.125, "dt": 0.0025}
]

equations_list = [equations.shallow_water]
boundary_functions = [boundaries.edge]
operators_list = [operators.gradient] 
integrators_list = [direct_solvers.lax_w, direct_solvers.lax_f]
initial_conditions = [init_conditions.shallow_dam]
coefficients = [1.0]

# Global Simulation Parameters
FINAL_TIME = 15
STEPS_PER_FRAME = 50   
RECORD_INTERVAL = 100

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
