import numpy as np
from Core import operators

#Euler integration
def euler(state, t, dt, dx, boundary, operator, equation, coefficient):
    dudt = equation(t, state, dx, boundary, operator, coefficient) #Returns first time derivative
    state_futr = state + (dudt*dt) #Future state
    return state_futr
    

#RK4 integration
def rk4(state, t, dt, dx, boundary, operator, equation, coefficient):
    
    k1 = equation(t, state, dx, boundary, operator, coefficient)
    k2 = equation(t + dt/2, state + (k1*dt/2), dx, boundary, operator, coefficient)
    k3 = equation(t + dt/2, state + (k2*dt/2), dx, boundary, operator, coefficient)
    k4 = equation(t + dt, state + (k3*dt), dx, boundary, operator, coefficient)
    
    state_futr = state + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0)
    return state_futr

    
#Leapfrog
def leapfrog(state, t, dt, dx, boundary, operator, equation, coefficient):

    if state.ndim < 2:
        raise ValueError(
            "Leapfrog integration requires a multi-state coupled system matrix "
            "(like the Wave Equation) to separate positions and velocities."
        )
    
    x1, v1 = state
    a1 = equation(t, state, dx, boundary, operator, coefficient)[1]
    
    dstatedt_mid = v1 + a1*(dt/2)
    state_futr = x1 + dstatedt_mid*dt
    state_2 = np.vstack([state_futr, dstatedt_mid])
    dstate_2dt = equation(t + dt, state_2, dx, boundary, operator, coefficient)
    a2 = dstate_2dt[1]
    dstatedt = dstatedt_mid + a2*(dt/2)
    
    return np.vstack([state_futr, dstatedt])
    
def lax(state, t, dt, dx, boundary, operator, equation, coefficient):
    # 1. Transform variables if the function has transformation mappings
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state

    # 2. Apply boundary conditions
    padded_cons = boundary(cons_state)

    # 3. Compute physical fluxes via the attached attribute
    flux = equation.flux(padded_cons)

    # 4. Spacetime update step via pure math stencils
    cons_next = operators.spatial_average(padded_cons) - dt * operators.central_flux_divergence(flux, dx)

    # 5. Transform back to primitive variables for visualization
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next