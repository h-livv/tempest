#Module imports
from src import solver, boundaries, operators, equations, integrators, init_conditions

#Run the engine
solver.solver(
    N=2500,
    init_state=init_conditions.shallow_gauss,
    boundary=boundaries.reflect,    
    operator=operators.upwind,      
    equation=equations.shallow_water,  
    integrator=integrators.lax_f,
    coefficient=1.0,
    dt=0.001,                  
    dx=0.1,
    FINAL_TIME=300,
    STEPS_PER_FRAME=600,
    RECORD_INTERVAL=10,
)

