import numpy as np
import warnings

def _check_compatibility(equation_name, init_condition_name, boundary):
    supported_configs = {
        "advection": {
            "ics": ["Any"],
            "bcs": ["periodic"]
        },
        "wave": {
            "ics": ["Any"],
            "bcs": ["periodic", "reflect"]
        },
        "diffusion": {
            "ics": ["gaussian"],
            "bcs": ["periodic", "reflect", "constant", "dirichlet", "edge"]
        },
        "burgers": {
            "ics": ["burgers_stationary_shock", "burgers_traveling_shock"],
            "bcs": ["periodic", "dirichlet"]
        },
        "shallow_water": {
            "ics": ["shallow_gaussian", "shallow_dam"],
            "bcs": ["Any"]
        }
    }
    
    if equation_name not in supported_configs:
        return
        
    config = supported_configs[equation_name]
    ic_supported = "Any" in config["ics"] or init_condition_name in config["ics"]
    
    if equation_name == "shallow_water":
        if init_condition_name == "shallow_gaussian":
            bc_supported = boundary == "periodic"
            valid_bcs_str = "periodic"
        elif init_condition_name == "shallow_dam":
            bc_supported = True
            valid_bcs_str = "Any"
        else:
            bc_supported = False
            valid_bcs_str = "N/A"
    else:
        bc_supported = "Any" in config["bcs"] or boundary in config["bcs"]
        valid_bcs_str = "\n         ".join(config["bcs"])
        
    if not (ic_supported and bc_supported):
        valid_ics_str = "\n         ".join(config["ics"])
        
        if equation_name == "shallow_water" and ic_supported:
            valid_bcs_str = "periodic" if init_condition_name == "shallow_gaussian" else "Any"
            
        msg = f"""Analytical validation for {equation_name.capitalize()}Equation currently supports:

    Initial Conditions:
         {valid_ics_str}

    Boundary Conditions:
         {valid_bcs_str}

Received:

    Initial Condition:
        {init_condition_name}

    Boundary:
        {boundary}

No closed-form analytical solution is implemented for this configuration."""
        raise ValueError(msg)


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
    
    if grid.ndim > 1 and equation.__name__ not in ["advection", "diffusion", "wave"]:
        return np.zeros_like(actual_u)
        
    _check_compatibility(equation.__name__, init_condition.__name__, boundary)
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
        if grid.ndim == 1:
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
            elif boundary == "reflect":
                u0 = clean_profile
                N = len(u0)
                u0_mirrored = np.concatenate([u0, u0[-2:0:-1]]) if N > 2 else u0
                N_m = len(u0_mirrored)
                U_fft = np.fft.fft(u0_mirrored)
                k = np.fft.fftfreq(N_m, d=dx_clean) * 2 * np.pi
                omega = c * np.abs(k)
                analytic_mirrored = np.real(np.fft.ifft(U_fft * np.cos(omega * t)))
                analytic_state = analytic_mirrored[:N]
            else:
                raise ValueError(f"Unknown boundary type for wave validation: {boundary}")
        elif grid.ndim == 2:
            u0 = init_condition(grid)[0]
            Ny, Nx = grid.shape
            dy, dx = grid.spacing
            
            if boundary == "periodic":
                U_fft = np.fft.fft2(u0)
                ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
                kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
                KY, KX = np.meshgrid(ky, kx, indexing='ij')
                omega = c * np.sqrt(KY**2 + KX**2)
                analytic_state = np.real(np.fft.ifft2(U_fft * np.cos(omega * t)))
            elif boundary == "reflect":
                u0_mirrored = np.concatenate([u0, u0[-2:0:-1]], axis=0) if Ny > 2 else u0
                u0_mirrored = np.concatenate([u0_mirrored, u0_mirrored[:, -2:0:-1]], axis=1) if Nx > 2 else u0_mirrored
                
                Ny_m, Nx_m = u0_mirrored.shape
                U_fft = np.fft.fft2(u0_mirrored)
                ky = np.fft.fftfreq(Ny_m, d=dy) * 2 * np.pi
                kx = np.fft.fftfreq(Nx_m, d=dx) * 2 * np.pi
                KY, KX = np.meshgrid(ky, kx, indexing='ij')
                omega = c * np.sqrt(KY**2 + KX**2)
                analytic_mirrored = np.real(np.fft.ifft2(U_fft * np.cos(omega * t)))
                analytic_state = analytic_mirrored[:Ny, :Nx]
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
            
            # The Gaussian parameter should only be obtained explicitly, but since the implementation relies on extracting it
            # from the profile, we'll keep the extraction but we know it's a Gaussian due to the compatibility check.
            eps = 1e-5
            curve_mask = (u0_1d > 0.01) & (u0_1d < 0.99)
            if np.any(curve_mask):
                a_estimates = -np.log(u0_1d[curve_mask]) / (x_i_1d[curve_mask] - x_c + eps)**2
                a = float(np.nanmean(a_estimates[np.isfinite(a_estimates)]))
            else:
                a = 100.0  # Fallback for very sharp gaussians
                
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
        
        if init_condition.__name__ == "shallow_gaussian":
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
            c_max = max(c_L, c_R)
            
            if c_max * t > L_clean / 2.0:
                warnings.warn(f"Dam break waves may have reached the boundaries (c_max*t = {c_max*t:.3f} > L/2 = {L_clean/2.0:.3f}). Analytical solution assumes infinite domain and may be invalid.", UserWarning)
            
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