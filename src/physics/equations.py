import numpy as np
from src.numerics import operators

class Equation:
    """Base class for all Tempest equations."""
    def __call__(self, t, state, grid, boundary, operator):
        raise NotImplementedError

    def flux(self, padded_state, grid_or_dx):
        raise NotImplementedError

    def wave_speed(self, padded_state):
        raise NotImplementedError

    def source(self, padded_state, grid):
        return None

    def compute_energies(self, field, boundary):
        """Generic energy computation (L2 norm)."""
        dV = np.prod(field.grid.spacing)
        data = field.data if hasattr(field, 'data') else np.asarray(field)
        total_e = np.sum(data**2) * dV
        return 0.0, 0.0, total_e


class AdvectionEquation(Equation):
    """Linear advection equation."""
    def __init__(self, velocity):
        self.__name__ = 'advection'
        self.velocity = velocity
        self.spatial_order = 1
        self.parity = [1]

    def __call__(self, t, state, grid, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )
            
        padded_state = boundary(state, self.parity)
        grad_u = operator(padded_state, grid, velocity=self.velocity)
        
        if grid.ndim > 1:
            dudt = -np.sum(self.velocity * grad_u, axis=0)
        else:
            dudt = -self.velocity * grad_u
        
        return dudt

    def flux(self, padded_state, grid):
        return self.velocity * padded_state

    def wave_speed(self, padded_state):
        return self.velocity


class WaveEquation(Equation):
    """Wave propagation equation."""
    def __init__(self, wave_speed):
        self.__name__ = 'wave'
        self.wave_speed_val = wave_speed
        self.spatial_order = 1
        self.parity = [1, -1] # Position is symmetric, velocity is asymmetric

    def __call__(self, t, state, grid, boundary, operator):
        if operator.__name__ in ('gradient', 'upwind'):
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Wave propagation is a 2nd-order spatial PDE. "
                "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
            )
        
        padded_state = boundary(state, self.parity)
        u, v = padded_state
        
        d2udx2 = operator(u, grid)
        d2udt2 = (self.wave_speed_val**2) * d2udx2
        
        return np.stack([state[1], d2udt2], axis=0)

    def flux(self, padded_state, grid):
        u, v = padded_state
        grad_u = operators.gradient(u, grid)
        ndim = grid.ndim
        pad_width = [(0, 0)] * (grad_u.ndim - ndim) + [(1, 1)] * ndim
        padded_grad_u = np.pad(grad_u, pad_width=pad_width, mode='edge')
        
        f1 = np.zeros_like(padded_grad_u)
        f2 = -(self.wave_speed_val**2) * padded_grad_u
        return np.stack([f1, f2], axis=0)

    def source(self, padded_state, grid):
        u, v = padded_state
        return np.stack([v, np.zeros_like(v)], axis=0)

    def wave_speed(self, padded_state):
        return self.wave_speed_val

    def compute_energies(self, field, boundary):
        dV = np.prod(field.grid.spacing)
        data = field.data if hasattr(field, 'data') else np.asarray(field)
        u, v = data[0], data[1]
        
        u_padded = boundary(u)
        grad_u = operators.gradient(u_padded, field.grid)
        
        pe = ((self.wave_speed_val**2)/2) * np.sum(grad_u**2) * dV
        ke = (1/2) * np.sum(v**2) * dV
        return pe, ke, pe + ke


class DiffusionEquation(Equation):
    """Diffusion equation."""
    def __init__(self, diffusivity):
        self.__name__ = 'diffusion'
        self.diffusivity = diffusivity
        self.spatial_order = 2
        self.parity = [1]

    def __call__(self, t, state, grid, boundary, operator):
        if operator.__name__ in ('gradient', 'upwind'):
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Diffusion is a 2nd-order spatial PDE. "
                "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
            )
        
        padded_state = boundary(state, self.parity)
        d2udx2 = operator(padded_state, grid)
        du_dt = self.diffusivity * d2udx2
        
        return du_dt

    def flux(self, padded_state, grid):
        grad_u = operators.gradient(padded_state, grid)
        ndim = grid.ndim
        pad_width = [(0, 0)] * (grad_u.ndim - ndim) + [(1, 1)] * ndim
        padded_flux = np.pad(grad_u, pad_width=pad_width, mode='edge')
        return -self.diffusivity * padded_flux

    def wave_speed(self, padded_state):
        return 0.0


class ShallowWaterEquation(Equation):
    """Shallow water equation."""
    def __init__(self):
        self.__name__ = 'shallow_water'
        self.spatial_order = 1
        self.parity = [1, -1]

    def __call__(self, t, state, grid, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Shallow water equation is a 1st-order spatial PDE. "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )
        
        padded_state = boundary(state, self.parity)
        g = 9.81 
        eps = 1e-5
        
        v_unpadded = state[1:]
        h, v = padded_state[0], padded_state[1:]
        q = h * v
        q_sq_by_h = np.where(h > eps, (q**2) / h, 0.0)
        
        dh_dt = -operator(q, grid)
        dq_dt = -operator((q_sq_by_h + 0.5 * g * (h**2)), grid)
        dv_dt = -v_unpadded * operator(v, grid) - g * operator(h, grid)
            
        return np.stack([dh_dt, dv_dt], axis=0)

    def flux(self, padded_cons, grid):
        h, q = padded_cons[0], padded_cons[1]
        g = 9.81
        f1 = q
        
        eps = 1e-5
        q_sq_over_h = np.zeros_like(q)
        mask = h > eps
        q_sq_over_h[mask] = (q[mask]**2) / h[mask]
        
        f2 = q_sq_over_h + 0.5 * g * (h**2)
        return np.stack([f1, f2], axis=0)

    def wave_speed(self, padded_state):
        h, q = padded_state
        g = 9.81
        eps = 1e-5
        v = np.zeros_like(q)
        mask = h > eps
        v[mask] = q[mask] / h[mask]
        return np.abs(v) + np.sqrt(g * h)
        
    def to_conservative(self, primitive_state):
        h, v = primitive_state
        return np.stack([h, h * v], axis=0)
        
    def to_primitive(self, conservative_state):
        h = conservative_state[0]                # shape (..., N) or (N,)
        q = conservative_state[1:]               # shape (n_v, N) or (N,)
        eps = 1e-5
        mask = h > eps
        v = np.zeros_like(q)
        # Broadcast-safe division: expand mask to match q's shape if needed
        if v.ndim > mask.ndim:
            expanded_mask = np.broadcast_to(mask, v.shape)
            v[expanded_mask] = q[expanded_mask] / np.broadcast_to(h, v.shape)[expanded_mask]
        else:
            v[mask] = q[mask] / h[mask]
        # Return as primitive [h, v1, v2, ...] stacked on axis 0
        v_squeezed = v[0] if (v.ndim > h.ndim and v.shape[0] == 1) else v
        return np.stack([h, v_squeezed], axis=0)

    def compute_energies(self, field, boundary):
        dV = np.prod(field.grid.spacing)
        data = field.data if hasattr(field, 'data') else np.asarray(field)
        h = data[0]
        v = data[1:]
        g = 9.81
        
        ke = 0.5 * np.sum(h * np.sum(v**2, axis=0)) * dV
        pe = 0.5 * g * np.sum(h**2) * dV
        return pe, ke, pe + ke


class BurgersEquation(Equation):
    """Burgers' equation."""
    def __init__(self, viscosity):
        self.__name__ = 'burgers'
        self.viscosity = viscosity
        self.spatial_order = 1
        self.parity = [1]

    def __call__(self, t, state, grid, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: You are controlling the operator of the advection term in Burgers' equation "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )

        padded_state = boundary(state, self.parity)

        wave_term = self.viscosity * operators.laplacian(padded_state, grid)
        adv_term = -operator(0.5*(padded_state)**2, grid, velocity=padded_state)
        
        if hasattr(grid, 'ndim') and grid.ndim > 1:
            adv_term = np.sum(adv_term, axis=0)

        du_dt = wave_term + adv_term
        return du_dt

    def flux(self, padded_state, grid):
        adv_flux = 0.5 * (padded_state)**2
        # Compute viscous flux using a central difference that preserves the
        # full padded shape. For each spatial axis, use np.roll to compute
        # (u[i+1] - u[i-1]) / (2*dx) across the whole padded domain.
        ndim = grid.ndim
        wave_flux = np.zeros_like(padded_state, dtype=float)
        for i in range(ndim):
            spacing = grid.get_spacing(i)
            ax = -(ndim - i)
            wave_flux += -self.viscosity * (
                np.roll(padded_state, -1, axis=ax) - np.roll(padded_state, 1, axis=ax)
            ) / (2.0 * spacing)
        return adv_flux + wave_flux

    def wave_speed(self, padded_state):
        return padded_state

    def compute_energies(self, field, boundary):
        dV = np.prod(field.grid.spacing)
        data = field.data if hasattr(field, 'data') else np.asarray(field)
        total_e = np.sum(0.5 * data**2) * dV
        return 0.0, 0.0, total_e