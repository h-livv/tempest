#Module imports
import numpy as np
from Core import grid
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators

#Run the engine
grid.grid1d(
    boundary=boundaries.reflect,    
    operator=operators.upwind,      
    equation=equations.advection,  
    integrator=integrators.rk4,
    coefficient=5.0,
    dt=0.01,                  
    dx=1.0                  
)