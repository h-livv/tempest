def gradient(state, dx, boundary, **kwargs):
    u_x = boundary(state) #Apply boundary conditions
    
    #Ellipsis maintains dimensionality. Basically groups all dimensions together and only slices along columns.
    left = u_x[..., 0:-2] #Left neighbors
    center = u_x[..., 1:-1] #The grid
    right = u_x[..., 2:] #Right neighbors
    
    grad = (right - left)/(2*dx) #Finite difference solution for the gradient
    return grad

def laplacian(state, dx, boundary, **kwargs):
    u_x = boundary(state)
    left = u_x[..., 0:-2]
    center = u_x[..., 1:-1]
    right = u_x[..., 2:]
    
    lap = (right - 2*center + left)/(dx*dx) #Finite difference solution for the laplacian
    return lap

def upwind(state, dx, boundary, velocity=1.0, **kwargs):
    u_x = boundary(state)

    center = u_x[..., 1:-1]
    
    if velocity >= 0.0:
        # Information flowing right: use backward difference
        left = u_x[..., 0:-2]
        return (center - left) / dx
    else:
        # Information flowing left: use forward difference
        right = u_x[..., 2:]
        return (right - center) / dx
    
    #Looks only one direction in space, instead of both like central gradient
    #Gradient is more accurate, but violates physical causality
           
#Components for the lax friedrichs solver      

def spatial_average(state, boundary):
    """
    Pure geometric averaging for Lax-Friedrichs style dissipation: 
    (U_{i+1} + U_{i-1}) / 2
    """
    u_x = boundary(state)
    return 0.5 * (u_x[..., 2:] + u_x[..., 0:-2])

def central_flux_divergence(flux, dx, boundary):
    """
    Conservative central flux derivative: 
    (F_{i+1} - F_{i-1}) / (2*dx)
    """
    f_x = boundary(flux)
    return (f_x[..., 2:] - f_x[..., 0:-2]) / (2 * dx)