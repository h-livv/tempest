#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions
from src.numerics import flux_methods

grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 5.0e-4
    }
]


#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.BurgersTravelingShockIC(nu=0.1), init_conditions.BurgersStationaryShockIC(nu=0.1, U=1.0)]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]
operators_list = [operators.upwind]
equations_list = [equations.BurgersEquation(viscosity=0.02)]
integrators_list = [integrators.rk4]

FINAL_TIME = 1.0
STEPS_PER_FRAME = 20   # animation only: steps between plot refreshes
RECORD_INTERVAL = 40    # metrics: snapshot every N timesteps (fixed for all runs)
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = False
