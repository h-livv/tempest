from src.physics import equations, init_conditions
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods# Convergence Configuration (Halving dx and dt to maintain CFL)
grid_configs = [
    {
        "N": 50,
        "dx": 0.02,
        "dt": 0.003
    },
    {
        "N": 100,
        "dx": 0.01,
        "dt": 0.0015
    },
    {
        "N": 200,
        "dx": 0.005,
        "dt": 0.00075
    },
    {
        "N": 400,
        "dx": 0.0025,
        "dt": 0.000375
    }
]

equations_list = [equations.ShallowWaterEquation()]
boundary_functions = [boundaries.periodic]
operators_list = [operators.gradient] 
integrators_list = [flux_methods.lax_w]
initial_conditions = [init_conditions.ShallowGaussianIC(sigma=20.0, amplitude=1e-6, ambient_depth=1.0, center_ratio=0.5, use_L_for_center=True)]

# Global Simulation Parameters
FINAL_TIME = 1.0
STEPS_PER_FRAME = 20   
RECORD_INTERVAL = 40

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
