import numpy as np
from src.numerics import operators

def _get_inner_slice(array, grid_or_dx):
    ndim = grid_or_dx.ndim if hasattr(grid_or_dx, "ndim") else 1
    slices = [slice(None)] * array.ndim
    for ax in range(-ndim, 0):
        slices[ax] = slice(1, -1)
    return tuple(slices)

def _get_flux_comp(flux_array, d, ndim):
    if ndim == 1:
        return flux_array
    if flux_array.ndim == ndim + 2:  # system: (components, ndim, spatial...)
        return flux_array[:, d]
    else:  # scalar: (ndim, spatial...)
        return flux_array[d]

def lax_f(state, t, dt, dx, boundary, operator, equation):
    """
    Lax-Friedrichs Direct Spatiotemporal Solver.
    """
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state
    parity = equation.parity if hasattr(equation, "parity") else [1] * state.shape[0]
    padded_cons = boundary(cons_state, parity)

    if hasattr(equation, "flux"):
        flux = equation.flux(padded_cons, dx)
    else: 
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .flux method to run under Lax-Friedrichs.")
    
    inner_slice = _get_inner_slice(padded_cons, dx)
    if hasattr(equation, "source") and equation.source(padded_cons, dx) is not None:
        source_term = equation.source(padded_cons, dx)[inner_slice]
    else:
        source_term = 0.0

    avg_term = operators.spatial_average(padded_cons, dx)
    div_term = operators.central_flux_divergence(flux, dx)
    
    cons_next = avg_term - dt * div_term + dt * source_term
    
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next

def lax_w(state, t, dt, dx, boundary, operator, equation):
    """
    Lax-Wendroff Direct Spatiotemporal Solver (MacCormack Predictor-Corrector).
    Naturally handles non-linear PDEs and systems.
    """
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state
    parity = equation.parity if hasattr(equation, "parity") else [1] * state.shape[0]
    padded_cons = boundary(cons_state, parity)

    if not hasattr(equation, "flux"):
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .flux method to run under Lax-Wendroff.")

    F_n = equation.flux(padded_cons, dx)
    inner_slice = _get_inner_slice(padded_cons, dx)
    if hasattr(equation, "source") and equation.source(padded_cons, dx) is not None:
        S_n = equation.source(padded_cons, dx)[inner_slice]
    else:
        S_n = 0.0
        
    cons_inner = padded_cons[inner_slice]

    # Use time step to alternate predictor/corrector direction to avoid bias and oscillations
    step = int(round(t / dt))
    ndim = dx.ndim if hasattr(dx, "ndim") else 1
    spatial_axes = tuple(range(-ndim, 0))
    
    # Predictor
    pred_flux_terms = 0.0
    for d in range(ndim):
        spacing = dx.get_spacing(d) if hasattr(dx, "get_spacing") else dx
        active_axis = spatial_axes[d]
        flux_comp = _get_flux_comp(F_n, d, ndim)
        
        shift = 1 if (step + d) % 2 == 0 else -1
        
        if shift == 1:
            right = flux_comp[operators._slice_along_axis(flux_comp, 1, active_axis, spatial_axes)]
            center = flux_comp[operators._slice_along_axis(flux_comp, 0, active_axis, spatial_axes)]
            F_diff = (right - center)
        else:
            center = flux_comp[operators._slice_along_axis(flux_comp, 0, active_axis, spatial_axes)]
            left = flux_comp[operators._slice_along_axis(flux_comp, -1, active_axis, spatial_axes)]
            F_diff = (center - left)
            
        pred_flux_terms += (dt / spacing) * F_diff
        
    U_star = cons_inner - pred_flux_terms + dt * S_n
    
    # Corrector
    padded_U_star = boundary(U_star, parity)
    F_star = equation.flux(padded_U_star, dx)
    source_res = equation.source(padded_U_star, dx) if hasattr(equation, "source") else None
    S_star = source_res[inner_slice] if source_res is not None else 0.0
    
    corr_flux_terms = 0.0
    for d in range(ndim):
        spacing = dx.get_spacing(d) if hasattr(dx, "get_spacing") else dx
        active_axis = spatial_axes[d]
        flux_comp = _get_flux_comp(F_star, d, ndim)
        
        shift = -1 if (step + d) % 2 == 0 else 1
        
        if shift == 1:
            right = flux_comp[operators._slice_along_axis(flux_comp, 1, active_axis, spatial_axes)]
            center = flux_comp[operators._slice_along_axis(flux_comp, 0, active_axis, spatial_axes)]
            F_diff = (right - center)
        else:
            center = flux_comp[operators._slice_along_axis(flux_comp, 0, active_axis, spatial_axes)]
            left = flux_comp[operators._slice_along_axis(flux_comp, -1, active_axis, spatial_axes)]
            F_diff = (center - left)
            
        corr_flux_terms += (dt / spacing) * F_diff
        
    cons_next = 0.5 * (cons_inner + U_star) - 0.5 * corr_flux_terms + 0.5 * dt * S_star
    
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next

def upwind(state, t, dt, dx, boundary, operator, equation):
    """
    Direct Upwind Spatiotemporal Solver.
    Restricted to scalar PDEs.
    """
    if state.shape[0] > 1 or (state.ndim == 2 and state.shape[0] > 1):
        raise ValueError(f"CRITICAL PHYSICS ERROR: Upwind direct solver is only supported for scalar PDEs. Equation '{equation.__name__}' is a system.")
        
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state
    parity = equation.parity if hasattr(equation, "parity") else [1] * state.shape[0]
    padded_cons = boundary(cons_state, parity)

    if not hasattr(equation, "wave_speed"):
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .wave_speed method to run under Direct Upwind.")
        
    if not hasattr(equation, "flux"):
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .flux method to run under Direct Upwind.")

    speed = equation.wave_speed(padded_cons)
    flux = equation.flux(padded_cons, dx)
    
    if hasattr(equation, "source") and equation.source(padded_cons, dx) is not None:
        S_n = equation.source(padded_cons, dx)[..., 1:-1]
    else:
        S_n = 0.0
    
    cons_inner = padded_cons[..., 1:-1]
    
    if np.all(speed >= 0):
        F_diff = flux[..., 1:-1] - flux[..., :-2]
    elif np.all(speed < 0):
        F_diff = flux[..., 2:] - flux[..., 1:-1]
    else:
        speed_inner = speed[..., 1:-1] if np.ndim(speed) > 0 else speed
        F_diff_bwd = flux[..., 1:-1] - flux[..., :-2]
        F_diff_fwd = flux[..., 2:] - flux[..., 1:-1]
        F_diff = np.where(speed_inner >= 0, F_diff_bwd, F_diff_fwd)
        
    cons_next = cons_inner - (dt / dx) * F_diff + dt * S_n
    
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next

lax_f.is_direct_solver = True
lax_w.is_direct_solver = True
upwind.is_direct_solver = True
