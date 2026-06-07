#Module imports
from Core import solver, boundaries, operators, equations, integrators, init_conditions


#Run the engine
solver.solver(
    N=500,
    init_state=init_conditions.shallow_dam,
    boundary=boundaries.reflect,    
    operator=operators.laplacian,      
    equation=equations.shallow_water,  
    integrator=integrators.lax,
    coefficient=1.0,
    dt=0.05,                  
    dx=1.0,
    FINAL_TIME=500,
    STEPS_PER_FRAME=30         
)

