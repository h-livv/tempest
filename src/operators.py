def gradient(state, dx, **kwargs):
    
    #Ellipsis maintains dimensionality. Basically groups all dimensions together and only slices along columns.
    left = state[..., 0:-2] #Left neighbors
    center = state[..., 1:-1] #The grid
    right = state[..., 2:] #Right neighbors
    
    grad = (right - left)/(2*dx) #Finite difference solution for the gradient
    return grad

def laplacian(state, dx, **kwargs):
    
    left = state[..., 0:-2]
    center = state[..., 1:-1]
    right = state[..., 2:]
    
    lap = (right - 2*center + left)/(dx*dx) #Finite difference solution for the laplacian
    return lap

def upwind(state, dx,   velocity=1.0, **kwargs):

    center = state[..., 1:-1]
    
    if velocity >= 0.0:
        # Information flowing right: use backward difference
        left = state[..., 0:-2]
        return (center - left) / dx
    else:
        # Information flowing left: use forward difference
        right = state[..., 2:]
        return (right - center) / dx
    
    #Looks only one direction in space, instead of both like central gradient
    #Gradient is more accurate, but violates physical causality
           
#Components for the lax friedrichs solver      
#Expect pre-padded state matrix, return unpadded outputs
def spatial_average(state):
    """
    Pure geometric averaging for Lax-Friedrichs style dissipation: 
    (U_{i+1} + U_{i-1}) / 2
    """
    return 0.5 * (state[..., 2:] + state[..., 0:-2])

def central_flux_divergence(flux, dx):
    """
    Conservative central flux derivative: 
    (F_{i+1} - F_{i-1}) / (2*dx)
    """
    return (flux[..., 2:] - flux[..., 0:-2]) / (2 * dx)