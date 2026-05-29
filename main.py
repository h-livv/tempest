#Module imports
import numpy as np
from Core import grid
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators
from Experiments import init_conditions


#Run the engine
grid.grid1d(
    init_state=init_conditions.shallow_dam,
    boundary=boundaries.reflect,    
    operator=operators.laplacian,      
    equation=equations.shallow_water,  
    integrator=integrators.lax,
    coefficient=1.0,
    dt=0.01,                  
    dx=1.0                  
)