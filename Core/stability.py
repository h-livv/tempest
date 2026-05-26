import numpy as np
from Core import operators

def tracking(state, dx, boundary, equation, coefficient):
    
    c = coefficient
    
    if equation == 'wave':
        u,v = state
    
        dudx = operators.gradient(u, dx, boundary)
        
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