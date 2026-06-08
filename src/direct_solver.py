#Module imports
import solver, boundaries, operators, equations, integrators, init_conditions


#Run the engine
solver.solver(
    N=2500,
    init_state=init_conditions.diff_rod,
    boundary=boundaries.edge,    
    operator=operators.laplacian,      
    equation=equations.diffusion,  
    integrator=integrators.rk4,
    coefficient=3.0,
    dt=0.01,                  
    dx=0.1,
    FINAL_TIME=100,
    STEPS_PER_FRAME=50,
    RECORD_INTERVAL=10,
)

