#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators, flux_methods
from src.physics import equations, init_conditions#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 1.0e-3   # CFL: sqrt(g*h_max)*dt/dx = 3.13*1e-3/0.005 = 0.63 ≤ 1 ✓
    }
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.ShallowDamIC()]
boundary_functions = [boundaries.edge]
operators_list = [operators.central_flux_divergence]
equations_list = [equations.ShallowWaterEquation()]
integrators_list = [flux_methods.lax_f]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 5    # animation only
RECORD_INTERVAL = 100  # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
