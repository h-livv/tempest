import numpy as np

def advection(t, u_pres, dx, boundary, operator):
    der = operator(u_pres, dx, boundary)
    
    c = 1.0
    dudt = -c*der
    
    return dudt

#Todo: Add wave and diffusion equations