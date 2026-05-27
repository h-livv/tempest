import numpy as np
from Core import operators

def advection(t, state, dx, boundary, operator, coefficient, c = 1.0):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    dudx = operator(state, dx, boundary) #First derivative of state

    if c > 0:
        # Wave moves right: right-most active cell looks purely upstream (left)
        dudx[..., -1] = (state[..., -1] - state[..., -2]) / dx
    elif c < 0:
        # Wave moves left: left-most active cell looks purely upstream (right)
        dudx[..., 0] = (state[..., 1] - state[..., 0]) / dx
    
    c = coefficient
    dudt = -c*dudx #PDE equation for advection
    
    return dudt #Returns velocity


def wave(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Wave propagation is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
    
    #Break the state down into position and velocity
    u = state[0]
    v = state[1]
    
    d2udx2 = operators.laplacian(u, dx, boundary) #Second derivative of state
    
    c = coefficient
    d2udt2 = (c**2)*d2udx2 #PDE equation for wave propagation
    
    return np.array([v, d2udt2]) #Input = [u, v] Output = [v, a]

#Todo: Add diffusion equation

def diffusion(t, state, dx, boundary, operator, coefficient):

    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Diffusion is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
        
    d2udx2 = operators.laplacian(state, dx, boundary) #Second derivative of state
    
    d = coefficient
    du_dt = d*(d2udx2)
    
    return du_dt
    
    