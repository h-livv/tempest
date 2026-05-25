import numpy as np
from Core import operators

def advection(t, state, dx, boundary, operator):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    dudx = operator(state, dx, boundary) #First derivative of state
    
    c = 1.0
    dudt = -c*dudx #PDE equation for advection
    
    return dudt #Returns velocity


def wave(t, state, dx, boundary, operator):
    
    #Break the state down into position and velocity
    u = state[0]
    v = state[1]
    
    d2udx2 = operators.laplacian(u, dx, boundary) #Second derivative of state
    
    c = 1.0
    d2udt2 = (c**2)*d2udx2 #PDE equation for wave propagation
    
    return np.array([v, d2udt2]) #Input = [u, v] Output = [v, a]

#Todo: Add diffusion equation