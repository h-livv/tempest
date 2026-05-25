#Module imports
import numpy as np
from Core import grid
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators

#Initial state of grid, single state systems for now
N = 250
x = np.linspace(0, N, N)
init_pos = np.exp(-0.01 * (x - N/5)**2)
init_vel = np.zeros(N)
init_state = np.vstack([init_pos, init_vel])

#Run the engine
grid.grid1d(
    initial_state=init_state,
    boundary=boundaries.reflect,    
    operator=operators.laplacian,      
    equation=equations.wave,  
    integrator=integrators.rk4,   
    dt=0.5,                  
    dx=1.0                  
)