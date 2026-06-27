import numpy as np
from src import operators

def tracking(state, dx, boundary, equation, coefficient):
    #equation is expected as a string name passed from the solver
    
    c = coefficient
    
    if equation == 'wave':
        u,v = state
    
        u_padded = boundary(u, parity=[1])
    
        dudx = operators.gradient(u_padded, dx)
        
        pe = ((c**2)/2)*(np.sum(dudx**2))*dx
        
        ke = (1/2)*(np.sum(v**2))*dx
        
        total_e = pe + ke
        
        return pe, ke, total_e
    
    elif equation == 'advection':
        
        total_e = np.sum(state**2)*dx
        
        return 0.0, 0.0, total_e
    
    elif equation == 'diffusion':
        
        total_e = np.sum(state**2)*dx
        
        return 0.0, 0.0, total_e
    
    elif equation == 'shallow_water':
        h, v = state  # The state is passed in primitive form [depth, velocity]
        g = 9.81
        
        # Kinetic Energy: Integral of 0.5 * depth * velocity^2
        ke = 0.5 * np.sum(h * (v**2)) * dx
        
        # Potential Energy: Integral of 0.5 * g * depth^2
        pe = 0.5 * g * np.sum(h**2) * dx
        
        total_e = pe + ke
        return pe, ke, total_e

    elif equation == 'burgers':

        total_e = np.sum(0.5 * state**2) * dx
        return 0.0, 0.0, total_e
        