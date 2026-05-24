import numpy as np

#Euler integration
def euler(u_pres, t, dt, dx, boundary, operator, equation):
    acc = equation(t, u_pres, dx, boundary, operator, equation)
    u_futr = u_pres + (acc*dt)
    return u_futr
    
#Todo: Update RK4 and Leapfrog for PDEs
#RK4 integration
def rk4(pva_matrix, dt, get_acc):
    
    pos, vel, acc = pva_matrix
    
    v1 = vel
    a1 = acc
    
    x_mid1 = pos + v1*(dt/2)
    v_mid1 = vel + a1*(dt/2)
    
    v2 = v_mid1
    a2 = get_acc(x_mid1, v_mid1)
    
    x_mid2 = pos + v2*(dt/2)
    v_mid2 = vel + a2*(dt/2)
    
    v3 = v_mid2
    a3 = get_acc(x_mid2, v_mid2)
    
    x_end = pos + v3*dt
    v_end = vel + a3*dt
    
    v4 = v_end
    a4 = get_acc(x_end, v_end)
    
    pos_next = pos + (v1 + 2*v2 + 2*v3 + v4)*(dt/6)
    vel_next = vel + (a1 + 2*a2 + 2*a3 + a4)*(dt/6)
    
    pva_matrix[0] = pos_next
    pva_matrix[1] = vel_next
    pva_matrix[2] = get_acc(pva_matrix[0], pva_matrix[1])

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
    
    