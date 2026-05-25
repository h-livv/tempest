import numpy as np

#Euler integration
def euler(state, t, dt, dx, boundary, operator, equation):
    dudt = equation(t, state, dx, boundary, operator) #Returns first time derivative
    state_futr = state + (dudt*dt) #Future state
    return state_futr
    

#RK4 integration
def rk4(state, t, dt, dx, boundary, operator, equation):
    
    k1 = equation(t, state, dx, boundary, operator)
    k2 = equation(t + dt/2, state + (k1*dt/2), dx, boundary, operator)
    k3 = equation(t + dt/2, state + (k2*dt/2), dx, boundary, operator)
    k4 = equation(t + dt, state + (k3*dt), dx, boundary, operator)
    
    state_futr = state + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0)
    return state_futr
    
#Todo: Update Leapfrog for PDEs

#Leapfrog
def leapfrog(pva_matrix, dt, get_acc):
    pos, vel, acc = pva_matrix
    
    x1 = pos
    v1 = vel
    a1 = acc
    
    v_mid = v1 + a1*(dt/2)
    x2 = x1 + v_mid*dt
    a2 = get_acc(x2, v_mid)
    v2 = v_mid + a2*(dt/2)
    
    pva_matrix[0] = x2
    pva_matrix[1] = v2
    pva_matrix[2] = a2
    
    