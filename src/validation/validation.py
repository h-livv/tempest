import numpy as np

def validation(equation, state, init_condition, grid, t, boundary):
    # Dynamically extract physical constant if it exists
    c = 0.0
    if equation.__name__ == 'advection' and hasattr(equation, 'velocity'):
        c = equation.velocity
    elif equation.__name__ == 'wave' and hasattr(equation, 'wave_speed_val'):
        c = equation.wave_speed_val
    elif equation.__name__ == 'diffusion' and hasattr(equation, 'diffusivity'):
        c = equation.diffusivity
    elif equation.__name__ == 'burgers' and hasattr(equation, 'viscosity'):
        c = equation.viscosity

    
    # Extract properties from grid to minimize disruption to existing analytical math
    if grid.ndim == 1:
        x = grid.coordinates[0]
        N = grid.shape[0]
        dx = grid.spacing[0]
        L = N * dx
    else:
        x = grid.coordinates
        N = grid.shape
        dx = grid.spacing
        L = None

    analytic_state = None
    actual_u = state[0] if (state.ndim > 1 and state.shape[0] >= 1) else state
    
    if grid.ndim > 1 and equation.__name__ not in ["advection", "diffusion"]:
        return np.zeros_like(actual_u)

    
    if equation.__name__ == "advection":
        class GridProxy:
            ndim = grid.ndim
            shape = grid.shape
            spacing = grid.spacing
            
        shifted_coords = []
        for i in range(grid.ndim):
            # Velocity might be an N-D tensor, so we flatten it to extract components
            if hasattr(c, "__len__") and len(c.flatten()) >= grid.ndim:
                v_i = float(c.flatten()[i])
            else:
                v_i = float(c)
                
            x_i = grid.coordinates[i] if grid.ndim > 1 else grid.coordinates[0]
            L_i = grid.shape[i] * grid.spacing[i]
            x0 = float(x_i.flat[0])
            
            if boundary == "dirichlet":
                shifted_coords.append(x_i - v_i * t)
            else:
                shifted_coords.append((x_i - v_i * t - x0) % L_i + x0)
                
        GridProxy.coordinates = shifted_coords
        analytic_state = init_condition(GridProxy())[0]

    elif equation.__name__ == "wave":
        # 1. Compute true domain metrics from the clean, pristine grid 'x'
        dx_clean = x[1] - x[0]
        L_clean = x.max() + dx_clean
        
        # 2. Sample the initial profile shape ONCE on the pristine grid.
        clean_profile = init_condition(grid)[0]
        
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
            
        elif boundary in ["constant", "dirichlet"]:
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
        u0 = init_condition(grid)[0]
        analytic_state = np.ones_like(actual_u, dtype=float)
        
        coords = grid.coordinates if grid.ndim > 1 else [grid.coordinates[0]]
        shapes = grid.shape if grid.ndim > 1 else [grid.shape[0]]
        spacings = grid.spacing if grid.ndim > 1 else [grid.spacing[0]]
        
        for i in range(grid.ndim):
            x_i = coords[i]
            L_i = shapes[i] * spacings[i]
            
            axes_to_reduce = tuple(j for j in range(grid.ndim) if j != i)
            u0_1d = np.max(u0, axis=axes_to_reduce) if len(axes_to_reduce) > 0 else u0
            
            slices = [0] * grid.ndim
            slices[i] = slice(None)
            x_i_1d = x_i[tuple(slices)] if grid.ndim > 1 else x_i
            
            x_c = x_i_1d[np.argmax(u0_1d)]
            
            curve_mask = (u0_1d > 0.01) & (u0_1d < 0.99)
            if np.any(curve_mask):
                eps = 1e-5
                a_estimates = -np.log(u0_1d[curve_mask]) / (x_i_1d[curve_mask] - x_c + eps)**2
                a = float(np.nanmean(a_estimates[np.isfinite(a_estimates)]))
            else:
                a = 100.0  # Safe fallback
                
            def u_inf(x_arr, t_val, center):
                if t_val == 0:
                    return np.exp(-a * (x_arr - center)**2)
                variance_factor = 1.0 + 4.0 * a * c * t_val
                return (1.0 / np.sqrt(variance_factor)) * np.exp(-a * (x_arr - center)**2 / variance_factor)
                
            analytic_1d = np.zeros_like(x_i, dtype=float)
            x0_val = float(x_i.flat[0])
            
            for m in range(-3, 4):
                if boundary == "periodic":
                    analytic_1d += u_inf(x_i, t, x_c + m * L_i)
                elif boundary in ["reflect", "edge"]:
                    analytic_1d += u_inf(x_i, t, x_c + 2 * m * L_i)
                    analytic_1d += u_inf(x_i, t, 2 * x0_val - x_c + 2 * m * L_i)
                elif boundary in ["constant", "dirichlet"]:
                    analytic_1d += u_inf(x_i, t, x_c + 2 * m * L_i)
                    analytic_1d -= u_inf(x_i, t, 2 * x0_val - x_c + 2 * m * L_i)
                else:
                    raise ValueError(f"Unknown boundary type for diffusion validation: {boundary}")
                    
            analytic_state *= analytic_1d
                
    elif equation.__name__ == "shallow_water":
        dx_clean = x[1] - x[0]
        L_clean = x.max() + dx_clean
        
        full_profile = init_condition(grid)[0]
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
            
    elif equation.__name__ == "burgers":
        u0 = init_condition(grid)[0]
        # Route based on the initial condition function name
        if "stationary" in init_condition.__name__:
            # Estimate U from the maximum amplitude, center at domain midpoint to match IC
            U_est = float(np.max(np.abs(u0)))
            x_0 = 0.5 * (x[0] + x[-1])
            analytic_state = -U_est * np.tanh(U_est * (x - x_0) / (2.0 * c))
        elif "traveling" in init_condition.__name__:
            # u_L is on the left, u_R is on the right
            u_L_est = float(u0[0])
            u_R_est = float(u0[-1])
            c_speed = 0.5 * (u_L_est + u_R_est)
            # Find the point where u0 crosses c_speed (shock center)
            idx_left = np.where(u0 >= c_speed)[0]
            if len(idx_left) > 0 and len(idx_left) < N:
                idx = idx_left[-1]
                # Linear interpolation for x_0
                x_0_est = x[idx] + (c_speed - u0[idx]) * (x[idx+1] - x[idx]) / (u0[idx+1] - u0[idx] + 1e-12)
            else:
                x_0_est = 0.5 * (x.max() + (x[1] - x[0]))
            
            dx_clean = x[1] - x[0]
            L_clean = x.max() + dx_clean
            
            # Map spatial coordinates to [-L/2, L/2) relative to the moving center
            rel_x = x - (x_0_est + c_speed * t)
            
            if boundary == 'dirichlet':
                wrapped_rel_x = rel_x
            else:
                wrapped_rel_x = (rel_x + 0.5 * L_clean) % L_clean - 0.5 * L_clean
            
            analytic_state = c_speed - 0.5 * (u_L_est - u_R_est) * np.tanh(((u_L_est - u_R_est) / (4.0 * c)) * wrapped_rel_x)
        else:
            analytic_state = np.zeros_like(actual_u)
    else:
        # Returns clean placeholder logs so your data.py pipelines can run uninterrupted
        return np.zeros_like(actual_u)
        
    return analytic_state