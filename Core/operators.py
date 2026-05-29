import numpy as np

def gradient(state, dx, boundary, c = 1.0):
    u_x = boundary(state) #Apply boundary conditions
    
    #Ellipsis maintains dimensionality
    left = u_x[..., 0:-2] #Left neighbors
    center = u_x[..., 1:-1] #The grid
    right = u_x[..., 2:] #Right neighbors
    
    grad = (right - left)/(2*dx) #Finite difference solution for the gradient
    return grad

def laplacian(state, dx, boundary):
    u_x = boundary(state)
    left = u_x[..., 0:-2]
    center = u_x[..., 1:-1]
    right = u_x[..., 2:]
    
    lap = (right - 2*center + left)/(dx*dx) #Finite difference solution for the laplacian
    return lap

def upwind(state, dx, boundary):
    u_x = boundary(state)
    left = u_x[..., 0:-2]
    center = u_x[..., 1:-1]
    
    return (center - left) / dx #Looks only backward in space, instead of both forward and backward like the gradient
                                #Gradient is more accurate, but violates physical causality
           
#Components for the lax friedrichs solver                                 
def spatial_average(padded_state):
    """Pure geometric averaging: (U_{i+1} + U_{i-1}) / 2"""
    return 0.5 * (padded_state[..., 2:] + padded_state[..., 0:-2])

def central_flux_divergence(padded_flux, dx):
    """Conservative central flux derivative: (F_{i+1} - F_{i-1}) / (2*dx)"""
    return (padded_flux[..., 2:] - padded_flux[..., 0:-2]) / (2 * dx)