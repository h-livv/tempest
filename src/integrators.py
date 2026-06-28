import numpy as np
from src import operators

#Euler integration
def euler(state, t, dt, grid, boundary, operator, equation, coefficient):
    dudt = equation(t, state, grid, boundary, operator, coefficient) #Returns first time derivative
    state_futr = state + (dudt*dt) #Future state
    return state_futr
    
#RK4 integration
def rk4(state, t, dt, grid, boundary, operator, equation, coefficient):
    
    k1 = equation(t, state, grid, boundary, operator, coefficient)
    k2 = equation(t + dt/2, state + (k1*dt/2), grid, boundary, operator, coefficient)
    k3 = equation(t + dt/2, state + (k2*dt/2), grid, boundary, operator, coefficient)
    k4 = equation(t + dt, state + (k3*dt), grid, boundary, operator, coefficient)
    
    state_futr = state + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0)
    return state_futr

#Leapfrog
def leapfrog(state, t, dt, grid, boundary, operator, equation, coefficient):

    if state.shape[0] < 2:
        raise ValueError(
            "Leapfrog integration requires a multi-state coupled system matrix "
            "(like the Wave Equation) to separate positions and velocities."
        )
    
    x1, v1 = state
    a1 = equation(t, state, grid, boundary, operator, coefficient)[1]
    
    dstatedt_mid = v1 + a1*(dt/2)
    state_futr = x1 + dstatedt_mid*dt
    state_2_data = np.stack([state_futr, dstatedt_mid], axis=0)
    if hasattr(state, 'grid'):
        state_2 = state.__class__(state.grid, state_2_data)
    else:
        state_2 = state_2_data
        
    dstate_2dt = equation(t + dt, state_2, grid, boundary, operator, coefficient)
    a2 = dstate_2dt[1]
    dstatedt = dstatedt_mid + a2*(dt/2)
    
    res = np.stack([state_futr, dstatedt], axis=0)
    if hasattr(state, 'grid'):
        return state.__class__(state.grid, res)
    return res
    
