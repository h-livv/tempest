import numpy as np
from Core import operators

def advection(t, state, dx, boundary, operator, coefficient, c = 1.0):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    dudx = operator(state, dx, boundary) #First derivative of state
    
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
    u, v = state
    
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
    
def shallow_water(t, state, dt, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Shallow water equation is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    h, v = state
    q = h*v
    g = 9.8
        
    dh_dt = -operator(q, dt, dx, boundary)
    dq_dt = -operator(((q**2/h) + 0.5*g*(h**2)), dt, dx, boundary)
        
    return np.vstack([dh_dt, dq_dt])

def _sw_flux(padded_cons):
    h, q = padded_cons[0], padded_cons[1]
    g = 9.81
    f1 = q
    
    # Safe-division mask: Only compute q^2 / h where water depth is active
    eps = 1e-5
    q_sq_over_h = np.where(h > eps, (q**2) / h, 0.0)
    
    f2 = q_sq_over_h + 0.5 * g * (h**2)
    return np.stack([f1, f2])

def _sw_to_conservative(primitive_state):
    h, v = primitive_state
    return np.stack([h, h * v])

def _sw_to_primitive(conservative_state):
    h, q = conservative_state
    eps = 1e-5
    v = np.where(h > eps, q / h, 0.0)
    return np.stack([h, v])

shallow_water.flux = _sw_flux
shallow_water.to_conservative = _sw_to_conservative
shallow_water.to_primitive = _sw_to_primitive   
    