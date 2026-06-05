import numpy as np

def l2_error(numerical, analytical):
    return np.sqrt(np.mean((numerical - analytical)**2))

def l1_error(numerical, analytical):
    return (np.mean(np.abs(numerical - analytical)))
    

def validation(equation, state, init_condition, N, x, t, c, bound_func, dx):
    
    if equation.__name__ == "advection":
        #Calculates length of the domain
        L = N * dx

        # x - c*t is the shape of the wave at time t
        # -x[0] to set the coordinates to 0
        # %L to shift the out of bounds points back into [0,L]
        # +x[0] to reset the coordinates-
        x_shifted = (x - c*t - x[0]) % L + x[0]
        analytic_state = init_condition(N, x_shifted)[0]
    
        l2 = l2_error(state, analytic_state)
        l1 = l1_error(state, analytic_state)
        max_error = np.max(np.abs(state - analytic_state))

    if equation.__name__ == "wave":
        L = N * dx
        boundary = bound_func.__name__
        
        x_minus = x - c * t #moving to the right
        x_plus  = x + c * t #moving to the left
        
        if boundary == "periodic":
            x_R = (x_minus - x[0]) % L + x[0]
            x_L = (x_plus - x[0]) % L + x[0]
            
            analytic_state = 0.5 * (init_condition(N, x_R)[0] + init_condition(N, x_L)[0])
            
        elif boundary == 'reflect':
            def map_reflective(x_val):
                rel_x = (x_val - x[0]) % (2 * L)
                mask = rel_x > L
                mapped = rel_x.copy()
                mapped[mask] = 2 * L - rel_x[mask]
                return mapped + x[0]
            
            analytic_state = 0.5 * (init_condition(N, map_reflective(x_minus))[0] + init_condition(N, map_reflective(x_plus))[0])
            
        elif boundary == "constant":
            def map_constant(x_val):
                rel_x = (x_val - x[0]) % (2 * L)
                mask = rel_x > L
                mapped = rel_x.copy()
                mapped[mask] = 2 * L - rel_x[mask]
                
                sign = np.ones_like(x_val)
                sign[mask] = -1.0
                return mapped + x[0], sign
            
            x_R, sign_R = map_constant(x_minus)
            x_L, sign_L = map_constant(x_plus)
            
            analytic_state = 0.5 * (sign_R * init_condition(N, x_R)[0] + sign_L * init_condition(N, x_L)[0])
                
        
        elif boundary == "edge":
            x_R = np.clip(x_minus, x[0], x[-1])
            x_L = np.clip(x_plus, x[0], x[-1])
            
            analytic_state = 0.5 * (init_condition(N, x_R)[0] + init_condition(N, x_L)[0])
            
        else:
            raise ValueError(f"Unknown boundary type for wave validation: {boundary}")
        
        actual_u = state[0] if (state.ndim > 1 and state.shape[0] == 2) else state

        l2 = l2_error(actual_u, analytic_state)
        l1 = l1_error(actual_u, analytic_state)
        max_error = np.max(np.abs(actual_u - analytic_state))
        
    elif equation.__name__ == "diffusion":
        L = N * dx
        boundary = bound_func.__name__
        
        u0 = init_condition(N, x)[0]
        x_c = x[np.argmax(u0)]
        
        # Sample non-peak grid points to isolate the coefficient 'a' from: u = exp(-a*(x-xc)^2)
        curve_mask = (u0 > 0.01) & (u0 < 0.99)
        if np.any(curve_mask):
            a_estimates = -np.log(u0[curve_mask]) / (x[curve_mask] - x_c)**2
            a = float(np.nanmean(a_estimates[np.isfinite(a_estimates)]))
        else:
            a = 100.0  # Safe fallback matching your physical coordinate setup
            
        # 2. Infinite-domain analytical Gaussian diffusion equation
        def u_inf(x_arr, t_val, center):
            if t_val == 0:
                return np.exp(-a * (x_arr - center)**2)
            variance_factor = 1.0 + 4.0 * a * c * t_val
            return (1.0 / np.sqrt(variance_factor)) * np.exp(-a * (x_arr - center)**2 / variance_factor)

        # 3. Sum over neighboring mirror images to satisfy boundary conditions
        analytic_state = np.zeros_like(x, dtype=float)
        
        # Summing from -3 to 3 covers 7 virtual worlds (perfect for machine precision accuracy)
        for m in range(-3, 4):
            if boundary == "periodic":
                # Periodic images repeat every full domain length
                analytic_state += u_inf(x, t, x_c + m * L)
                
            elif boundary in ["reflect", "edge"]:
                # Rigid walls: add overlapping right-side-up mirror profiles
                analytic_state += u_inf(x, t, x_c + 2 * m * L)
                analytic_state += u_inf(x, t, 2 * x[0] - x_c + 2 * m * L)
                
            elif boundary == "constant":
                # Fixed walls (u=0): subtract inverted mirror profiles to force cancellation at boundaries
                analytic_state += u_inf(x, t, x_c + 2 * m * L)
                analytic_state -= u_inf(x, t, 2 * x[0] - x_c + 2 * m * L)
                
            else:
                raise ValueError(f"Unknown boundary type for diffusion validation: {boundary}")

        actual_u = state[0] if (state.ndim > 1 and state.shape[0] == 2) else state
        l2 = l2_error(actual_u, analytic_state)
        l1 = l1_error(actual_u, analytic_state)
        max_error = np.max(np.abs(actual_u - analytic_state))
        
        
    return {"l2_error": l2, "l1_error": l1, "max_error": max_error, "relative": state, "analytic_state": analytic_state}


    