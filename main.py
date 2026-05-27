#Module imports
import numpy as np
from Core import grid
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators

#Run the engine
grid.grid1d(
    boundary=boundaries.edge,    
    operator=operators.laplacian,      
    equation=equations.diffusion,  
    integrator=integrators.rk4,
    coefficient=1.0,
    dt=0.1,                  
    dx=1.0                  
)