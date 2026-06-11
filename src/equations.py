import numpy as np

def advection(t, state, dx, boundary, operator, coefficient):
    
    if operator.__name__ == 'laplacian':
        raise ValueError(
            "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
            "You cannot pass 'laplacian' (2nd-order) as its operator."
        )
        
    parity = [1]
    
    padded_state = boundary(state, parity)
    
    dudx = operator(padded_state, dx, velocity=coefficient) #First derivative of state
    
    dudt = -coefficient*dudx #PDE equation for advection
    
    return dudt #Returns velocity

#Linear Advection: F(u) = c * u
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
    
    return np.vstack([state[1], d2udt2]) #Input = [u, v] Output = [v, a]

#Wave Equation: F(U) = [0, -c² * du/dx]ᵀ  |  S(U) = [v, 0]ᵀ
def _wave_flux(padded_state, coefficient, dx):
    u, v = padded_state
    from src import operators
    
    dudx = operators.gradient(u, dx)
    padded_dudx = np.pad(dudx, pad_width=(1, 1), mode='edge')
    
    f1 = np.zeros_like(padded_dudx)
    f2 = -(coefficient**2) * padded_dudx
    return np.vstack([f1, f2])

def _wave_source(padded_state, coefficient, dx):
    u, v = padded_state
    # Note: Because avg_term and div_term output unpadded shapes [2, N],
    # our source term matrix must also return sliced, unpadded values matching domain size N.
    return np.vstack([v[1:-1], np.zeros_like(v[1:-1])])


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

# Diffusion: F(u) = -D * du/dx
def _diffusion_flux(padded_state, coefficient, dx):
    from src import operators
    # Gradient drops shape from N+2 to N
    dudx = operators.gradient(padded_state, dx)
    # Pad the flux back to N+2 so central_flux_divergence can slice it down to N later
    padded_flux = np.pad(dudx, pad_width=[(0, 0)] * (dudx.ndim - 1) + [(1, 1)], mode='edge')
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
        
    return np.vstack([dh_dt, dv_dt])

#Shallow water lux calculation for lax-friedrichs
def _sw_flux(padded_cons):
    h, q = padded_cons[0], padded_cons[1]
    g = 9.81
    f1 = q
    
    eps = 1e-5
    q_sq_over_h = np.where(h > eps, (q**2) / h, 0.0)
    
    f2 = q_sq_over_h + 0.5 * g * (h**2)
    return np.vstack([f1, f2])

'''def _LW_flux(padded_cons):
    h,q = '''

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

advection.parity = [1]          # 1-field: scalar is symmetric
diffusion.parity = [1]          # 1-field: temperature is symmetric
wave.parity = [1, -1]           # 2-fields: position (u) is symmetric, velocity (v) is anti-symmetric
shallow_water.parity = [1, -1]  # 2-fields: height (h) is symmetric, velocity (v) is anti-symmetric

advection.flux = _advection_flux

diffusion.flux = _diffusion_flux

wave.flux = _wave_flux
wave.source = _wave_source

#Connecting back to integrators.py
shallow_water.flux = _sw_flux
shallow_water.to_conservative = _sw_to_conservative
shallow_water.to_primitive = _sw_to_primitive   
    