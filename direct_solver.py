#Module imports
from src import solver, boundaries, operators, equations, integrators, direct_solvers, init_conditions

#Run the engine
solver.solver(
    N=1000,
    init_state=init_conditions.advec_gauss,
    boundary=boundaries.periodic,    
    operator=operators.upwind,      
    equation=equations.advection,  
    integrator=direct_solvers.lax_w,
    coefficient=1.0,
    dt=0.01,
    dx=0.5,
    FINAL_TIME=1000,
    STEPS_PER_FRAME=100,
    RECORD_INTERVAL=5,
)
