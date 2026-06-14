#Module imports
from src import solver, boundaries, operators, equations, integrators, init_conditions

#Run the engine
solver.solver(
    N=5000,
    init_state=init_conditions.shallow_dam,
    boundary=boundaries.reflect,    
    operator=operators.gradient,      
    equation=equations.shallow_water,  
    integrator=integrators.lax_f,
    coefficient=1.0,
    dt=0.001,
    dx=0.1,
    FINAL_TIME=500,
    STEPS_PER_FRAME=100,
    RECORD_INTERVAL=5,
)

