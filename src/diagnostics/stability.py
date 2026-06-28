import numpy as np
from src import operators

def tracking(state, grid, boundary, equation):
    state = np.asarray(state)
    
    dV = np.prod(grid.spacing)
    
    if equation.__name__ == 'wave':
        c = equation.wave_speed_val if hasattr(equation, 'wave_speed_val') else 0.0
        u,v = state
    
        u_padded = boundary(u, parity=[1])
    
        dudx = operators.gradient(u_padded, grid)
        
        pe = ((c**2)/2)*(np.sum(dudx**2))*dV
        
        ke = (1/2)*(np.sum(v**2))*dV
        
        total_e = pe + ke
        
        return pe, ke, total_e
    
    elif equation.__name__ == 'advection':
        
        total_e = np.sum(state**2)*dV
        
        return 0.0, 0.0, total_e
    
    elif equation.__name__ == 'diffusion':
        
        total_e = np.sum(state**2)*dV
        
        return 0.0, 0.0, total_e
    
    elif equation.__name__ == 'shallow_water':
        h, v = state  # The state is passed in primitive form [depth, velocity]
        g = 9.81
        
        # Kinetic Energy: Integral of 0.5 * depth * velocity^2
        ke = 0.5 * np.sum(h * (v**2)) * dV
        
        # Potential Energy: Integral of 0.5 * g * depth^2
        pe = 0.5 * g * np.sum(h**2) * dV
        
        total_e = pe + ke
        return pe, ke, total_e

    elif equation.__name__ == 'burgers':

        total_e = np.sum(0.5 * state**2) * dV
        return 0.0, 0.0, total_e
        