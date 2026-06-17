import numpy as np
from src import operators

def lax_f(state, t, dt, dx, boundary, operator, equation, coefficient):
    """
    Lax-Friedrichs Direct Spatiotemporal Solver.
    """
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state
    parity = equation.parity if hasattr(equation, "parity") else [1] * state.shape[0]
    padded_cons = boundary(cons_state, parity)

    if hasattr(equation, "flux"):
        flux = equation.flux(padded_cons, coefficient, dx)
    else: 
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .flux method to run under Lax-Friedrichs.")
    
    if hasattr(equation, "source"):
        source_term = equation.source(padded_cons, coefficient, dx)[..., 1:-1]
    else:
        source_term = 0.0

    avg_term = operators.spatial_average(padded_cons)
    div_term = operators.central_flux_divergence(flux, dx)
    
    cons_next = avg_term - dt * div_term + dt * source_term
    
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next

def lax_w(state, t, dt, dx, boundary, operator, equation, coefficient):
    """
    Lax-Wendroff Direct Spatiotemporal Solver (MacCormack Predictor-Corrector).
    Naturally handles non-linear PDEs and systems.
    """
    cons_state = equation.to_conservative(state) if hasattr(equation, "to_conservative") else state
    parity = equation.parity if hasattr(equation, "parity") else [1] * state.shape[0]
    padded_cons = boundary(cons_state, parity)

    if not hasattr(equation, "flux"):
        raise AttributeError(f"CRITICAL: Equation '{equation.__name__}' must register a .flux method to run under Lax-Wendroff.")

    F_n = equation.flux(padded_cons, coefficient, dx)
    if hasattr(equation, "source"):
        S_n = equation.source(padded_cons, coefficient, dx)[..., 1:-1]
    else:
        S_n = 0.0
        
    cons_inner = padded_cons[..., 1:-1]

    # Use time step to alternate predictor/corrector direction to avoid bias and oscillations
    step = int(round(t / dt))
    
    if step % 2 == 0:
        # --- EVEN STEPS: Predictor (Forward), Corrector (Backward) ---
        F_diff_pred = F_n[..., 2:] - F_n[..., 1:-1]
        U_star = cons_inner - (dt / dx) * F_diff_pred + dt * S_n
        
        padded_U_star = boundary(U_star, parity)
        F_star = equation.flux(padded_U_star, coefficient, dx)
        S_star = equation.source(padded_U_star, coefficient, dx)[..., 1:-1] if hasattr(equation, "source") else 0.0
        
        F_diff_corr = F_star[..., 1:-1] - F_star[..., :-2]
    else:
        # --- ODD STEPS: Predictor (Backward), Corrector (Forward) ---
        F_diff_pred = F_n[..., 1:-1] - F_n[..., :-2]
        U_star = cons_inner - (dt / dx) * F_diff_pred + dt * S_n
        
        padded_U_star = boundary(U_star, parity)
        F_star = equation.flux(padded_U_star, coefficient, dx)
        S_star = equation.source(padded_U_star, coefficient, dx)[..., 1:-1] if hasattr(equation, "source") else 0.0
        
        F_diff_corr = F_star[..., 2:] - F_star[..., 1:-1]
        
    cons_next = 0.5 * (cons_inner + U_star) - 0.5 * (dt / dx) * F_diff_corr + 0.5 * dt * S_star
    
    return equation.to_primitive(cons_next) if hasattr(equation, "to_primitive") else cons_next

def upwind(state, t, dt, dx, boundary, operator, equation, coefficient):
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

    speed = equation.wave_speed(padded_cons, coefficient)
    flux = equation.flux(padded_cons, coefficient, dx)
    
    if hasattr(equation, "source"):
        S_n = equation.source(padded_cons, coefficient, dx)[..., 1:-1]
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
