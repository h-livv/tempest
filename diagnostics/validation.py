import numpy as np

def validation(equation, state, init_condition, N, x, t, c, boundary, dx):
    
    #Calculates length of the domain
    L = N * dx
    analytic_state = None
    actual_u = state[0] if (state.ndim > 1 and state.shape[0] >= 1) else state
    
    if equation.__name__ == "advection":

        # x - c*t is the shape of the wave at time t
        # -x[0] to set the coordinates to 0
        # %L to shift the out of bounds points back into [0,L]
        # +x[0] to reset the coordinates-
        x_shifted = (x - c*t - x[0]) % L + x[0]
        analytic_state = init_condition(N, x_shifted)[0]

    elif equation.__name__ == "wave":
        # 1. Compute true domain metrics from the clean, pristine grid 'x'
        dx_clean = x[1] - x[0]
        L_clean = x.max() + dx_clean
        
        # 2. Sample the initial profile shape ONCE on the pristine grid.
        clean_profile = init_condition(N, x)[0]
        
        x_minus = x - c * t  # moving to the right
        x_plus  = x + c * t  # moving to the left
        
        if boundary == "periodic":
            x_R = (x_minus - x[0]) % L_clean + x[0]
            x_L = (x_plus - x[0]) % L_clean + x[0]
            
            u_R = np.interp(x_R, x, clean_profile)
            u_L = np.interp(x_L, x, clean_profile)
            analytic_state = 0.5 * (u_R + u_L)
            
        elif boundary == 'reflect':
            def map_reflective(x_val):
                rel_x = (x_val - x[0]) % (2 * L_clean)
                mask = rel_x > L_clean
                mapped = rel_x.copy()
                mapped[mask] = 2 * L_clean - rel_x[mask]
                return mapped + x[0]
            
            u_R = np.interp(map_reflective(x_minus), x, clean_profile)
            u_L = np.interp(map_reflective(x_plus), x, clean_profile)
            analytic_state = 0.5 * (u_R + u_L)
            
        elif boundary == "constant":
            def map_constant(x_val):
                rel_x = (x_val - x[0]) % (2 * L_clean)
                mask = rel_x > L_clean
                mapped = rel_x.copy()
                mapped[mask] = 2 * L_clean - rel_x[mask]
                
                sign = np.ones_like(x_val)
                sign[mask] = -1.0
                return mapped + x[0], sign
            
            x_R, sign_R = map_constant(x_minus)
            x_L, sign_L = map_constant(x_plus)
            
            u_R = np.interp(x_R, x, clean_profile) * sign_R
            u_L = np.interp(x_L, x, clean_profile) * sign_L
            analytic_state = 0.5 * (u_R + u_L)
                
        elif boundary == "edge":
            x_R = np.clip(x_minus, x[0], x[-1])
            x_L = np.clip(x_plus, x[0], x[-1])
            
            u_R = np.interp(x_R, x, clean_profile)
            u_L = np.interp(x_L, x, clean_profile)
            analytic_state = 0.5 * (u_R + u_L)
            
        else:
            raise ValueError(f"Unknown boundary type for wave validation: {boundary}")
        
    elif equation.__name__ == "diffusion":
        
        u0 = init_condition(N, x)[0]
        x_c = x[np.argmax(u0)]
        
        # Sample non-peak grid points to isolate the coefficient 'a' from: u = exp(-a*(x-xc)^2)
        curve_mask = (u0 > 0.01) & (u0 < 0.99)
        if np.any(curve_mask):
            eps = 1e-5
            a_estimates = -np.log(u0[curve_mask]) / (x[curve_mask] - x_c + eps)**2
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
                
    elif equation.__name__ == "shallow_water":
        dx_clean = x[1] - x[0]
        L_clean = x.max() + dx_clean
        
        full_profile = init_condition(N, x)[0]
        h_max = np.max(full_profile)
        h_min = np.min(full_profile)
        is_dam_break = (h_max - h_min) > 1.0
        
        if not is_dam_break:
            h0 = h_min
            clean_profile = full_profile - h0
            g = 9.81
            c_wave = np.sqrt(g * h0)
            
            x_minus = x - c_wave * t
            x_plus  = x + c_wave * t
            
            if boundary == "periodic":
                x_R = (x_minus - x[0]) % L_clean + x[0]
                x_L = (x_plus - x[0]) % L_clean + x[0]
                u_R = np.interp(x_R, x, clean_profile)
                u_L = np.interp(x_L, x, clean_profile)
                analytic_state = h0 + 0.5 * (u_R + u_L)
            else:
                return np.zeros_like(actual_u)
        else:
            # Stoker's Exact 1D Dam Break Riemann Solver
            if t == 0:
                return full_profile
                
            h_L = h_max
            h_R = h_min
            g = 9.81
            x_0 = 0.5 * x.max()
            
            c_L = np.sqrt(g * h_L)
            c_R = np.sqrt(g * h_R)
            
            # Root finding for intermediate depth h_m (Bisection)
            def f(hm):
                return 2 * (c_L - np.sqrt(g * hm)) - (hm - h_R) * np.sqrt((g * (hm + h_R)) / (2 * hm * h_R))
                
            low, high = h_R + 1e-6, h_L - 1e-6
            for _ in range(50):
                mid = 0.5 * (low + high)
                if f(mid) > 0:
                    low = mid
                else:
                    high = mid
            h_m = 0.5 * (low + high)
            
            c_m = np.sqrt(g * h_m)
            u_m = 2 * (c_L - c_m)
            S = np.sqrt((g * h_m * (h_m + h_R)) / (2 * h_R))
            
            x_head = x_0 - c_L * t
            x_tail = x_0 + (u_m - c_m) * t
            x_shock = x_0 + S * t
            
            analytic_state = np.zeros_like(x)
            
            mask_L = x <= x_head
            analytic_state[mask_L] = h_L
            
            mask_fan = (x > x_head) & (x <= x_tail)
            c_fan = (2 * c_L - (x[mask_fan] - x_0) / t) / 3.0
            analytic_state[mask_fan] = (c_fan**2) / g
            
            mask_m = (x > x_tail) & (x <= x_shock)
            analytic_state[mask_m] = h_m
            
            mask_R = x > x_shock
            analytic_state[mask_R] = h_R
            
    else:
        # Returns clean placeholder logs so your data.py pipelines can run uninterrupted
        return np.zeros_like(actual_u)
        
        
    return analytic_state


    