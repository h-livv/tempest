import numpy as np

def advection(t, u_pres, dx, boundary, operator, equation):
    der = operator(u_pres, dx, boundary)
    
    c = 1.0
    dudt = -c*der
    
    return dudt