import numpy as np

def _slice_along_axis(array, shift, active_axis, spatial_axes):
    """
    Constructs a multidimensional slice.
    For the `active_axis`, it shifts the slice (-1 for left, 0 for center, 1 for right).
    For all other `spatial_axes`, it takes the center slice (1:-1).
    For non-spatial axes, it takes everything (:).
    """
    slices = [slice(None)] * array.ndim
    for ax in spatial_axes:
        if ax == active_axis:
            if shift == -1:
                slices[ax] = slice(0, -2) # Left
            elif shift == 0:
                slices[ax] = slice(1, -1) # Center
            elif shift == 1:
                slices[ax] = slice(2, None) # Right
        else:
            slices[ax] = slice(1, -1) # Center slice for non-active spatial axes
    return tuple(slices)


# =============================================================================
# LOW-LEVEL AXIS-AWARE OPERATORS
# =============================================================================

def gradient_axis(padded_array, spacing, active_axis, spatial_axes):
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - left) / (2 * spacing)

def laplacian_axis(padded_array, spacing, active_axis, spatial_axes):
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    center = padded_array[_slice_along_axis(padded_array, 0, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - 2*center + left) / (spacing**2)

def upwind_axis(padded_array, spacing, velocity, active_axis, spatial_axes):
    center = padded_array[_slice_along_axis(padded_array, 0, active_axis, spatial_axes)]
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]

    backward_diff = (center - left) / spacing
    forward_diff = (right - center) / spacing

    if isinstance(velocity, (float, int)):
        v_center = velocity
    else:
        v_center = velocity[_slice_along_axis(velocity, 0, active_axis, spatial_axes)]
    
    return np.where(v_center >= 0.0, backward_diff, forward_diff)

def spatial_average_axis(padded_array, active_axis, spatial_axes):
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return 0.5 * (right + left)

def central_flux_divergence_axis(padded_array, spacing, active_axis, spatial_axes):
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - left) / (2 * spacing)

# =============================================================================
# HIGH-LEVEL DIMENSION-AGNOSTIC OPERATORS
# =============================================================================

def _get_spatial_axes(ndim):
    """
    Returns the negative indices of spatial axes.
    E.g., for 2D, returns (-2, -1). For 1D, returns (-1,).
    """
    return tuple(range(-ndim, 0))

def gradient(padded_state, grid_or_dx):
    """
    Computes the gradient across all spatial dimensions of the grid.
    Returns a stacked array of gradients (e.g. [grad_y, grad_x] for 2D, [grad_x] for 1D).
    """
    if isinstance(grid_or_dx, (float, int)):
        # Compatibility shim for 1D
        spacing = grid_or_dx
        left = padded_state[..., 0:-2]
        right = padded_state[..., 2:]
        return (right - left) / (2 * spacing)
        
    grid = grid_or_dx
    spatial_axes = _get_spatial_axes(grid.ndim)
    grads = []
    # Compute gradient for each spatial axis
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        grad = gradient_axis(padded_state, spacing, ax, spatial_axes)
        grads.append(grad)
    
    if len(grads) == 1:
        return grads[0]
    return np.stack(grads, axis=0)

def laplacian(padded_state, grid_or_dx):
    """
    Computes the Laplacian by summing the second derivatives across all spatial axes.
    """
    if isinstance(grid_or_dx, (float, int)):
        # Compatibility shim for 1D
        spacing = grid_or_dx
        left = padded_state[..., 0:-2]
        center = padded_state[..., 1:-1]
        right = padded_state[..., 2:]
        return (right - 2*center + left) / (spacing**2)

    grid = grid_or_dx
    spatial_axes = _get_spatial_axes(grid.ndim)
    lap = 0
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        lap += laplacian_axis(padded_state, spacing, ax, spatial_axes)
    return lap

def upwind(padded_state, grid_or_dx, velocity=None):
    """
    Computes the upwind advection derivative. 
    If velocity is a vector field (stack of velocities for each axis), it applies the 
    corresponding velocity component to each axis gradient and sums them (dot product).
    """
    if velocity is None:
        velocity = padded_state

    if isinstance(grid_or_dx, (float, int)):
        # Compatibility shim for 1D
        spacing = grid_or_dx
        center = padded_state[..., 1:-1]
        left = padded_state[..., 0:-2]
        right = padded_state[..., 2:]

        backward_diff = (center - left) / spacing
        forward_diff = (right - center) / spacing

        v_center = velocity if isinstance(velocity, (float, int)) else velocity[..., 1:-1]
        return np.where(v_center >= 0.0, backward_diff, forward_diff)

    grid = grid_or_dx
    spatial_axes = _get_spatial_axes(grid.ndim)
    
    total_derivative = 0
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        # If velocity is a vector field array (shape: [ndim, ...]), extract the i-th component.
        # Otherwise assume it's a scalar or already aligned.
        if isinstance(velocity, np.ndarray) and velocity.shape[0] == grid.ndim and velocity.ndim == grid.ndim + 1:
            v_comp = velocity[i]
        elif isinstance(velocity, np.ndarray) and velocity.shape[0] == grid.ndim and padded_state.ndim == grid.ndim + 2:
            # Multi-field state, vector velocity
            v_comp = velocity[i]
        else:
            v_comp = velocity
            
        total_derivative += upwind_axis(padded_state, spacing, v_comp, ax, spatial_axes)
        
    return total_derivative

def spatial_average(padded_state, grid_or_dx=None):
    """
    Applies spatial averaging across all spatial axes for Lax-Friedrichs schemes.
    Averages over all spatial dimensions sequentially.
    """
    if isinstance(grid_or_dx, (float, int)) or grid_or_dx is None:
        # Compatibility shim for 1D
        left = padded_state[..., 0:-2]
        right = padded_state[..., 2:]
        return 0.5 * (right + left)

    grid = grid_or_dx
    spatial_axes = _get_spatial_axes(grid.ndim)
    
    # We apply averaging recursively over each axis
    averaged = padded_state
    for ax in spatial_axes:
        # For each axis, we average the current array.
        # Note: spatial_average_axis drops the shape by 2 along the active axis,
        # but requires the other axes to already be correct OR we must do it carefully.
        # Actually, Lax-Friedrichs in multi-D averages all 2*ndim neighbors.
        pass
        
    # Standard multi-dimensional Lax-Friedrichs average: 
    # Average of 2*d neighbors (cross stencil).
    # We can compute this by taking spatial_average_axis for each axis on the padded_state,
    # and summing them up, then dividing by ndim.
    total_avg = 0
    for ax in spatial_axes:
        total_avg += spatial_average_axis(padded_state, ax, spatial_axes)
    return total_avg / grid.ndim

def central_flux_divergence(padded_flux, grid_or_dx):
    """
    Computes the divergence of a flux vector field or scalar field.
    If padded_flux is a stacked vector field, computes sum(dF_i / dx_i).
    """
    if isinstance(grid_or_dx, (float, int)):
        # Compatibility shim for 1D
        spacing = grid_or_dx
        left = padded_flux[..., 0:-2]
        right = padded_flux[..., 2:]
        return (right - left) / (2 * spacing)

    grid = grid_or_dx
    spatial_axes = _get_spatial_axes(grid.ndim)
    
    divergence = 0
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        
        # If padded_flux is a vector field (stack of fluxes for each direction)
        if isinstance(padded_flux, np.ndarray) and padded_flux.shape[0] == grid.ndim:
            flux_comp = padded_flux[i]
        else:
            flux_comp = padded_flux
            
        divergence += central_flux_divergence_axis(flux_comp, spacing, ax, spatial_axes)
        
    return divergence