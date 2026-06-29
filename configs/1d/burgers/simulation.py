#Module imports
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions
from src.numerics import flux_methods

grid_configs = [
    {
        "N": 200,
        "dx": 0.005,
        "dt": 2.5e-4  # Viscous CFL: ν*dt/dx² = 0.02*2.5e-4/2.5e-5 = 0.2 < 0.5 ✓
    }
]


#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.BurgersStationaryShockIC(nu=0.1, U=1.0)]
boundary_functions = [boundaries.Dirichlet(2.0, 1.0)]
operators_list = [operators.upwind]
equations_list = [equations.BurgersEquation(viscosity=0.02)]
integrators_list = [integrators.rk4]

FINAL_TIME = 5.0
STEPS_PER_FRAME = 20   # steps per frame so animation stays fluid
RECORD_INTERVAL = 100  # metrics: snapshot every N timesteps
# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
