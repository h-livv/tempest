import numpy as np
from src import operators

def advection(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    parity = [1]
    
    padded_state = boundary(state, parity)
    
    dudx = operator(padded_state, dx, velocity=coefficient) #First derivative of state
    
    if isinstance(dx, tuple) or (hasattr(dx, 'ndim') and dx.ndim > 1):
        dudt = -np.sum(coefficient * dudx, axis=0)
    else:
        dudt = -coefficient*dudx #PDE equation for advection
    
    return dudt #Returns velocity

#Linear Advection
def _advection_flux(padded_state, coefficient, dx):
    return coefficient * padded_state


def wave(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Wave propagation is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
    
    parity = [1, -1] #Position is symmetric, velocity is asymmetric
    
    #Break the state down into position and velocity
    
    padded_state = boundary(state, parity)
    
    u, v = padded_state
    
    d2udx2 = operator(u, dx) #Second derivative of state
    
    d2udt2 = (coefficient**2)*d2udx2 #PDE equation for wave propagation
    
    return np.stack([state[1], d2udt2], axis=0) #Input = [u, v] Output = [v, a]

#Wave Equation
def _wave_flux(padded_state, coefficient, dx):
    u, v = padded_state
    
    dudx = operators.gradient(u, dx)
    ndim = dx.ndim if hasattr(dx, 'ndim') else 1
    pad_width = [(0, 0)] * (dudx.ndim - ndim) + [(1, 1)] * ndim
    padded_dudx = np.pad(dudx, pad_width=pad_width, mode='edge')
    
    f1 = np.zeros_like(padded_dudx)
    f2 = -(coefficient**2) * padded_dudx
    return np.stack([f1, f2], axis=0)

def _wave_source(padded_state, coefficient, dx):
    u, v = padded_state
    return np.stack([v, np.zeros_like(v)], axis=0)


def diffusion(t, state, dx, boundary, operator, coefficient):

    if operator.__name__ == 'gradient' or operator.__name__ == 'upwind':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Diffusion is a 2nd-order spatial PDE. "
            "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
        )
        
    parity = [1]
    
    padded_state = boundary(state, parity)
    
    d2udx2 = operator(padded_state, dx) #Second derivative of state
    du_dt = coefficient*(d2udx2)
    
    return du_dt

# Diffusion
def _diffusion_flux(padded_state, coefficient, dx):
    # Gradient drops shape from N+2 to N
    dudx = operators.gradient(padded_state, dx)
    # Pad the flux back to N+2 so central_flux_divergence can slice it down to N later
    ndim = dx.ndim if hasattr(dx, 'ndim') else 1
    pad_width = [(0, 0)] * (dudx.ndim - ndim) + [(1, 1)] * ndim
    padded_flux = np.pad(dudx, pad_width=pad_width, mode='edge')
    return -coefficient * padded_flux
    
def shallow_water(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Shallow water equation is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
    
    parity = [1,-1]
    
    padded_state = boundary(state, parity)
    
    g = 9.81 
    eps = 1e-5
    
    v_unpadded = state[1]
    h, v = padded_state
    q = h*v
    q_sq_by_h = np.where(h > eps, (q**2) / h, 0.0)
    
    dh_dt = -operator(q, dx)
    dq_dt = -operator((q_sq_by_h + 0.5*g*(h**2)), dx)
    dv_dt = -v_unpadded * operator(v, dx) - g * operator(h, dx)
        
    return np.stack([dh_dt, dv_dt], axis=0)

#Shallow water lux calculation for lax-friedrichs
def _sw_flux(padded_cons, coefficient, dx):
    h, q = padded_cons[0], padded_cons[1]
    g = 9.81
    f1 = q
    
    eps = 1e-5
    q_sq_over_h = np.where(h > eps, (q**2) / h, 0.0)
    
    f2 = q_sq_over_h + 0.5 * g * (h**2)
    return np.stack([f1, f2], axis=0)

def burgers(t, state, dx, boundary, operator, coefficient):

    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: You are controlling the operator of the advection term in Burgers' equation "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )

    v = coefficient

    parity = [1]

    padded_state = boundary(state, parity)

    wave_term = v * operators.laplacian(padded_state, dx)

    adv_term = -operator(0.5*(padded_state)**2, dx, velocity=padded_state)
    if isinstance(dx, tuple) or (hasattr(dx, 'ndim') and dx.ndim > 1):
        adv_term = np.sum(adv_term, axis=0)

    du_dt = wave_term + adv_term

    return du_dt

def _burgers_flux(padded_state, coefficient, dx):
    
    adv_flux = 0.5*(padded_state)**2

    wave_flux = -coefficient*(operators.gradient(padded_state, dx))
    ndim = dx.ndim if hasattr(dx, 'ndim') else 1
    pad_width = [(0, 0)] * (wave_flux.ndim - ndim) + [(1, 1)] * ndim
    padded_flux = np.pad(wave_flux, pad_width=pad_width, mode='edge')

    return adv_flux + padded_flux

#Primitive [h,v] to Conservative [h,q]
def _sw_to_conservative(primitive_state):
    h, v = primitive_state
    return np.stack([h, h * v], axis=0)

#Conservative [h,q] to Primitive [h,v]
def _sw_to_primitive(conservative_state):
    h, q = conservative_state
    eps = 1e-5
    v = np.where(h > eps, q / h, 0.0)
    return np.stack([h, v], axis=0)

# Wave Speed methods for Direct Upwind
def _advection_wave_speed(padded_state, coefficient):
    return coefficient

def _diffusion_wave_speed(padded_state, coefficient):
    return 0.0

def _wave_wave_speed(padded_state, coefficient):
    return coefficient

def _sw_wave_speed(padded_state, coefficient):
    h, q = padded_state
    g = 9.81
    eps = 1e-5
    v = np.where(h > eps, q / h, 0.0)
    return np.abs(v) + np.sqrt(g * h)

def _burgers_wave_speed(padded_state, coefficient):
    return padded_state

advection.parity = [1]          # 1-field: scalar is symmetric
diffusion.parity = [1]          # 1-field: temperature is symmetric
wave.parity = [1, -1]           # 2-fields: position (u) is symmetric, velocity (v) is anti-symmetric
shallow_water.parity = [1, -1]  # 2-fields: height (h) is symmetric, velocity (v) is anti-symmetric
burgers.parity = [1]

advection.flux = _advection_flux
advection.wave_speed = _advection_wave_speed

diffusion.flux = _diffusion_flux
diffusion.wave_speed = _diffusion_wave_speed

wave.flux = _wave_flux
wave.source = _wave_source
wave.wave_speed = _wave_wave_speed

burgers.flux = _burgers_flux
burgers.wave_speed = _burgers_wave_speed

#Connecting back to integrators.py / direct_solvers.py
shallow_water.flux = _sw_flux
shallow_water.to_conservative = _sw_to_conservative
shallow_water.to_primitive = _sw_to_primitive   
shallow_water.wave_speed = _sw_wave_speed
    