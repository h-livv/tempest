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

def _extract_vector_component(field, index, grid_ndim, state_ndim=None):
    """
    Extracts the i-th component if field is a stacked vector field array.
    """
    if isinstance(field, np.ndarray) and field.shape[0] == grid_ndim:
        # If no state_ndim is provided (e.g. flux), just extract the component
        if state_ndim is None:
            return field[index]
        # For velocity, ensure the dimensions match either a scalar field or a multi-field state
        if field.ndim == grid_ndim + 1 or state_ndim == grid_ndim + 2:
            return field[index]
    return field


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

    v_center = velocity[_slice_along_axis(velocity, 0, active_axis, spatial_axes)] if isinstance(velocity, np.ndarray) else velocity
    
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

def gradient(padded_state, grid, **kwargs):
    """
    Computes the gradient across all spatial dimensions of the grid.
    Returns a stacked array of gradients (e.g. [grad_y, grad_x] for 2D, [grad_x] for 1D).
    """
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

def laplacian(padded_state, grid, **kwargs):
    """
    Computes the Laplacian by summing the second derivatives across all spatial axes.
    """
    spatial_axes = _get_spatial_axes(grid.ndim)
    lap = 0
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        lap += laplacian_axis(padded_state, spacing, ax, spatial_axes)
    return lap

def upwind(padded_state, grid, velocity=None):
    """
    Computes the upwind advection derivative. 
    If velocity is a vector field (stack of velocities for each axis), it applies the 
    corresponding velocity component to each axis gradient and sums them (dot product).
    """
    if velocity is None:
        velocity = padded_state

    spatial_axes = _get_spatial_axes(grid.ndim)
    
    grads = []
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        # Extract the i-th component if velocity is a vector field array
        v_comp = _extract_vector_component(velocity, i, grid.ndim, padded_state.ndim)
            
        grad = upwind_axis(padded_state, spacing, v_comp, ax, spatial_axes)
        grads.append(grad)
        
    if len(grads) == 1:
        return grads[0]
    return np.stack(grads, axis=0)

def spatial_average(padded_state, grid):
    """
    Applies spatial averaging across all spatial axes for Lax-Friedrichs schemes.
    Averages over all spatial dimensions sequentially.
    """
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

def central_flux_divergence(padded_flux, grid):
    """
    Computes the divergence of a flux vector field or scalar field.
    If padded_flux is a stacked vector field, computes sum(dF_i / dx_i).
    """
    # System PDE recursion: If padded_flux has an extra dimension for component states
    if padded_flux.ndim == grid.ndim + 2:
        return np.stack([central_flux_divergence(padded_flux[c], grid) for c in range(padded_flux.shape[0])], axis=0)

    spatial_axes = _get_spatial_axes(grid.ndim)
    
    divergence = 0
    for i, ax in enumerate(spatial_axes):
        spacing = grid.get_spacing(i)
        
        # Extract the i-th component if padded_flux is a vector field
        flux_comp = _extract_vector_component(padded_flux, i, grid.ndim)
            
        divergence += central_flux_divergence_axis(flux_comp, spacing, ax, spatial_axes)
        
    return divergence

gradient.convergence_order = 2
laplacian.convergence_order = 2
upwind.convergence_order = 1