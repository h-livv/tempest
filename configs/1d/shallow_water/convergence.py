from src import equations, boundaries, operators, integrators, direct_solvers, init_conditions

# Convergence Configuration (Halving dx and dt to maintain CFL)
grid_configs = [
    {"N": 50, "dx": 1.0, "dt": 0.02},
    {"N": 50, "dx": 0.5, "dt": 0.01},
    {"N": 50, "dx": 0.25, "dt": 0.005},
    {"N": 50, "dx": 0.125, "dt": 0.0025}
]

equations_list = [equations.ShallowWaterEquation()]
boundary_functions = [boundaries.periodic]
operators_list = [operators.gradient] 
integrators_list = [direct_solvers.lax_w]
initial_conditions = [init_conditions.ShallowGaussianIC(sigma=20.0, amplitude=1e-6, ambient_depth=1.0, center_ratio=0.5, use_L_for_center=True)]

# Global Simulation Parameters
FINAL_TIME = 10
STEPS_PER_FRAME = 50   
RECORD_INTERVAL = 100

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
