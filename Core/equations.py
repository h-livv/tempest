import numpy as np

def advection(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    dudx = operator(state, dx, boundary, velocity=coefficient) #First derivative of state
    
    dudt = -coefficient*dudx #PDE equation for advection
    
    return dudt #Returns velocity


def wave(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Wave propagation is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
    
    #Break the state down into position and velocity
    u, v = state
    
    d2udx2 = operator(u, dx, boundary) #Second derivative of state
    
    d2udt2 = (coefficient**2)*d2udx2 #PDE equation for wave propagation
    
    return np.vstack([v, d2udt2]) #Input = [u, v] Output = [v, a]

#Todo: Add diffusion equation

def diffusion(t, state, dx, boundary, operator, coefficient):

    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Diffusion is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
        
    d2udx2 = operator(state, dx, boundary) #Second derivative of state
    du_dt = coefficient*(d2udx2)
    
    return du_dt
    
def shallow_water(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Shallow water equation is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
    
    g = 9.81 
    eps = 1e-5
    
    h, v = state
    q = h*v
    q_sq_by_h = np.where(h > eps, (q**2) / h, 0.0)
    
    dh_dt = -operator(q, dx, boundary)
    dq_dt = -operator((q_sq_by_h + 0.5*g*(h**2)), dx, boundary)
        
    return np.vstack([dh_dt, dq_dt])

#Flux calculation for lax-friedrichs
def _sw_flux(padded_cons):
    h, q = padded_cons[0], padded_cons[1]
    g = 9.81
    f1 = q
    
    eps = 1e-5
    q_sq_over_h = np.where(h > eps, (q**2) / h, 0.0)
    
    f2 = q_sq_over_h + 0.5 * g * (h**2)
    return np.vstack([f1, f2])

#Primitive [h,v] to Conservative [h,q]
def _sw_to_conservative(primitive_state):
    h, v = primitive_state
    return np.vstack([h, h * v])

#Conservative [h,q] to Primitive [h,v]
def _sw_to_primitive(conservative_state):
    h, q = conservative_state
    eps = 1e-5
    v = np.where(h > eps, q / h, 0.0)
    return np.vstack([h, v])

#Connecting back to integrators.py
shallow_water.flux = _sw_flux
shallow_water.to_conservative = _sw_to_conservative
shallow_water.to_primitive = _sw_to_primitive   
    