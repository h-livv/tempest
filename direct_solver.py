#Module imports
from src import solver, boundaries, operators, equations, integrators, direct_solvers, init_conditions

#Run the engine
solver.solver(
    N=5000,
    init_state=init_conditions.burgers_traveling_shock,
    boundary=boundaries.Dirichlet(2, 1),    
    operator=operators.upwind,      
    equation=equations.burgers,  
    integrator=integrators.rk4,
    coefficient=1.0,
    dt=0.01,
    dx=0.5,
    FINAL_TIME=1000,
    STEPS_PER_FRAME=250,
    RECORD_INTERVAL=5,
)
