"""
Tempest spatial finite-difference operators.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _slice_along_axis(array, shift, active_axis, spatial_axes):
    """
    Constructs a multidimensional slice for stencil computation.
    
    For the `active_axis`, it shifts the slice based on direction:
      - shift = -1 (left neighbor) -> slice(0, -2)
      - shift =  0 (center node)   -> slice(1, -1)
      - shift =  1 (right neighbor) -> slice(2, None)
    For non-active spatial axes, it slices the interior cells: slice(1, -1).
    For non-spatial component axes, it takes everything: slice(None).
    """
    slices = [slice(None)] * array.ndim
    for ax in spatial_axes:
        if ax == active_axis:
            if shift == -1:
                slices[ax] = slice(0, -2)
            elif shift == 0:
                slices[ax] = slice(1, -1)
            elif shift == 1:
                slices[ax] = slice(2, None)
        else:
            slices[ax] = slice(1, -1)
    return tuple(slices)

def _extract_vector_component(field, index, grid_ndim):
    """
    Extract the requested vector component.

    Expected layouts:
        Scalar coefficient:
            (...)
        Vector coefficient:
            (grid_ndim, ...)
    """

    if not isinstance(field, np.ndarray):
        return field

    # Vector coefficient field
    if field.ndim == grid_ndim + 1:
        if field.shape[0] == grid_ndim:
            return field[index]
        elif field.shape[0] == 1:
            return field[0]
        else:
            raise ValueError(
                f"Vector field components {field.shape[0]} does not match grid_ndim {grid_ndim}."
            )

    # Scalar / broadcast coefficient
    if field.ndim <= grid_ndim:
        return field

    raise ValueError(
        f"Unsupported coefficient shape {field.shape} for {grid_ndim}D simulation."
    )


# =============================================================================
# LOW-LEVEL AXIS-AWARE OPERATORS
# =============================================================================

def gradient_axis(padded_array, spacing, active_axis, spatial_axes):
    """
    Central Difference gradient stencil on a single axis.
    
    Math:
        d_u / d_x ≈ (u_{i+1} - u_{i-1}) / (2 * dx)
    """
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - left) / (2 * spacing)

def laplacian_axis(padded_array, spacing, active_axis, spatial_axes):
    """
    Second-derivative central difference stencil on a single axis.
    
    Math:
        d^2_u / d_x^2 ≈ (u_{i+1} - 2*u_i + u_{i-1}) / dx^2
    """
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    center = padded_array[_slice_along_axis(padded_array, 0, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - 2*center + left) / (spacing**2)

def upwind_axis(padded_array, spacing, velocity, active_axis, spatial_axes):
    """
    Upwind transport stencil on a single axis.
    
    Math:
        d_u / d_x ≈ (u_i - u_{i-1}) / dx   (if velocity >= 0)
                  ≈ (u_{i+1} - u_i) / dx   (if velocity < 0)
    """
    center = padded_array[_slice_along_axis(padded_array, 0, active_axis, spatial_axes)]
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]

    backward_diff = (center - left) / spacing
    forward_diff = (right - center) / spacing

    v_center = velocity[_slice_along_axis(velocity, 0, active_axis, spatial_axes)] if isinstance(velocity, np.ndarray) else velocity
    
    return np.where(v_center >= 0.0, backward_diff, forward_diff)

def spatial_average_axis(padded_array, active_axis, spatial_axes):
    """Spatial average over nearest neighbors for Lax schemes."""
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return 0.5 * (right + left)

def central_flux_divergence_axis(padded_array, spacing, active_axis, spatial_axes):
    """Central flux derivative on a single axis."""
    left = padded_array[_slice_along_axis(padded_array, -1, active_axis, spatial_axes)]
    right = padded_array[_slice_along_axis(padded_array, 1, active_axis, spatial_axes)]
    return (right - left) / (2 * spacing)


# =============================================================================
# HIGH-LEVEL DIMENSION-AGNOSTIC OPERATORS
# =============================================================================

def _get_spatial_axes(ndim):
    return tuple(range(-ndim, 0))

def gradient(padded_state, dx, **kwargs):
    """
    Computes spatial gradients across all active dimensions.
    
    Returns:
        np.ndarray: Stack of gradient components.
    """
    spatial_axes = _get_spatial_axes(len(dx))
    grads = []
    for i, ax in enumerate(spatial_axes):
        spacing = dx[i]
        grad = gradient_axis(padded_state, spacing, ax, spatial_axes)
        grads.append(grad)
    
    if len(grads) == 1:
        return grads[0]
    return np.stack(grads, axis=0)

def laplacian(padded_state, dx, **kwargs):
    """Computes the Laplacian (sum of second derivatives) across all spatial axes."""
    spatial_axes = _get_spatial_axes(len(dx))
    lap = 0.0
    for i, ax in enumerate(spatial_axes):
        spacing = dx[i]
        lap += laplacian_axis(padded_state, spacing, ax, spatial_axes)
    return lap

def upwind(padded_state, dx, velocity=None):
    """Computes upwind advection terms based on directional wind speed velocity."""
    if velocity is None:
        velocity = padded_state

    spatial_axes = _get_spatial_axes(len(dx))
    grads = []
    for i, ax in enumerate(spatial_axes):
        spacing = dx[i]
        v_comp = _extract_vector_component(velocity, i, len(dx))
        grad = upwind_axis(padded_state, spacing, v_comp, ax, spatial_axes)
        grads.append(grad)
        
    if len(grads) == 1:
        return grads[0]
    return np.stack(grads, axis=0)

def spatial_average(padded_state, dx):
    """Computes multi-dimensional spatial averages of neighbors for Lax schemes."""
    spatial_axes = _get_spatial_axes(len(dx))
    total_avg = 0.0
    for ax in spatial_axes:
        total_avg += spatial_average_axis(padded_state, ax, spatial_axes)
    return total_avg / len(dx)

def central_flux_divergence(padded_flux, dx):
    """Computes flux divergence (sum of dF_i / dx_i) over all spatial dimensions."""
    if padded_flux.ndim == len(dx) + 2:
        return np.stack([central_flux_divergence(padded_flux[c], dx) for c in range(padded_flux.shape[0])], axis=0)

    spatial_axes = _get_spatial_axes(len(dx))
    divergence = 0.0
    for i, ax in enumerate(spatial_axes):
        spacing = dx[i]
        flux_comp = _extract_vector_component(padded_flux, i, len(dx))
        divergence += central_flux_divergence_axis(flux_comp, spacing, ax, spatial_axes)
    return divergence

def build_1d_laplacian(N, dx):
    L = np.zeros((N, N))

    for i in range(N):
        L[i, i] = -2

        if i > 0:
            L[i, i-1] = 1

        if i < N-1:
            L[i, i+1] = 1

    return L / dx**2

class PoissonSolver:
    """
    Sparse 2-D Poisson solver: ∇²ψ = q  →  ψ = L⁻¹ q.

    The discrete Laplacian L is assembled once from sparse 1-D blocks using
    scipy.sparse.diags / eye / kron, then pre-factorized via SuperLU
    (scipy.sparse.linalg.factorized).  Every subsequent call to solve()
    reuses the factorization — no matrix operations at time-step cost.

    The stencil, boundary assumptions (implicit Dirichlet by truncation),
    and flattened row-major ordering are identical to the previous dense
    implementation, so all numerical results are bit-for-bit compatible.
    """

    def __init__(self, shape, dx):
        self.Ny, self.Nx = shape
        self.dx_x, self.dx_y = dx

        # --- build sparse 1-D Laplacians (same tridiagonal stencil as before) ---
        def _sparse_1d_laplacian(N, h):
            diags_data = [
                -2.0 * np.ones(N),   # main diagonal
                 1.0 * np.ones(N - 1),  # super-diagonal
                 1.0 * np.ones(N - 1),  # sub-diagonal
            ]
            return sp.diags(diags_data, [0, 1, -1], shape=(N, N), format='csc') / h**2

        Lx = _sparse_1d_laplacian(self.Nx, self.dx_x)
        Ly = _sparse_1d_laplacian(self.Ny, self.dx_y)
        Ix = sp.eye(self.Nx, format='csc')
        Iy = sp.eye(self.Ny, format='csc')

        # --- assemble 2-D operator (identical to dense kron construction) ---
        self.L = sp.kron(Iy, Lx, format='csc') + sp.kron(Ly, Ix, format='csc')

        # --- pre-factorize once; _solve is a callable that applies L⁻¹ ---
        self._solve = spla.factorized(self.L)

    def solve(self, state):
        """Apply the pre-factorized L⁻¹ to the flattened RHS; reshape back."""
        psi = self._solve(state.ravel().astype(np.float64))
        return psi.reshape(state.shape)

gradient.convergence_order = 2
laplacian.convergence_order = 2
upwind.convergence_order = 1
