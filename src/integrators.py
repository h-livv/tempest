import numpy as np
from src import operators

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

    if state.shape[0] < 2:
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
    # 1. Transform variables if the function has transformation mappings (primitive to conservative)
    #Checks if the input equation has a "to_conservative" attribute, only then applies this transformation
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state

    # 2. Compute physical fluxes via the attached attribute
    flux = equation.flux(cons_state) if hasattr(equation, "flux") else equation(t, cons_state, dx, boundary, operator, coefficient)

    # 3. Spacetime update step 
    avg_term = operators.spatial_average(cons_state, boundary)
    div_term = operators.central_flux_divergence(flux, dx, boundary)
    
    cons_next = avg_term - dt * div_term

    # 4. Transform back to primitive variables for pipeline visualization
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next