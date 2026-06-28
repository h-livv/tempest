"""
Tempest ODE time integrators.
"""

import numpy as np

def euler(state, t, dt, grid, boundary, operator, equation):
    """
    First-order Explicit Euler time integration.
    
    Math:
        u(t + dt) = u(t) + dt * f(t, u)
    """
    dudt = equation(t, state, grid, boundary, operator)
    return state + (dudt * dt)
    
def rk4(state, t, dt, grid, boundary, operator, equation):
    """
    Fourth-order Runge-Kutta (RK4) time integration.
    
    Math:
        k1 = f(t, u)
        k2 = f(t + dt/2, u + dt/2 * k1)
        k3 = f(t + dt/2, u + dt/2 * k2)
        k4 = f(t + dt, u + dt * k3)
        u(t + dt) = u(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = equation(t, state, grid, boundary, operator)
    k2 = equation(t + dt/2, state + (k1*dt/2), grid, boundary, operator)
    k3 = equation(t + dt/2, state + (k2*dt/2), grid, boundary, operator)
    k4 = equation(t + dt, state + (k3*dt), grid, boundary, operator)
    
    return state + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0)

def leapfrog(state, t, dt, grid, boundary, operator, equation):
    """
    Symplectic Leapfrog time integration.
    
    Required specifically for second-order wave propagation systems to preserve 
    conservation properties (shadow Hamiltonian) over long simulations.
    
    State is expected to contain [position, velocity].
    """
    if state.shape[0] < 2:
        raise ValueError(
            "Leapfrog integration requires a multi-state coupled system matrix "
            "(like the Wave Equation) to separate positions and velocities."
        )
    
    x1, v1 = state
    a1 = equation(t, state, grid, boundary, operator)[1]
    
    dstatedt_mid = v1 + a1*(dt/2)
    state_futr = x1 + dstatedt_mid*dt
    state_2_data = np.stack([state_futr, dstatedt_mid], axis=0)
    if hasattr(state, 'grid'):
        state_2 = state.__class__(state.grid, state_2_data)
    else:
        state_2 = state_2_data
        
    dstate_2dt = equation(t + dt, state_2, grid, boundary, operator)
    a2 = dstate_2dt[1]
    dstatedt = dstatedt_mid + a2*(dt/2)
    
    res = np.stack([state_futr, dstatedt], axis=0)
    if hasattr(state, 'grid'):
        return state.__class__(state.grid, res)
    return res
