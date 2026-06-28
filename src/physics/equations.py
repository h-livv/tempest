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

    def source(self, padded_state, grid_or_dx):
        return None


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
        dudx = operator(padded_state, grid, velocity=self.velocity)
        
        if grid.ndim > 1:
            dudt = -np.sum(self.velocity * dudx, axis=0)
        else:
            dudt = -self.velocity * dudx
        
        return dudt

    def flux(self, padded_state, dx):
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

    def flux(self, padded_state, dx):
        u, v = padded_state
        dudx = operators.gradient(u, dx)
        ndim = dx.ndim if hasattr(dx, 'ndim') else 1
        pad_width = [(0, 0)] * (dudx.ndim - ndim) + [(1, 1)] * ndim
        padded_dudx = np.pad(dudx, pad_width=pad_width, mode='edge')
        
        f1 = np.zeros_like(padded_dudx)
        f2 = -(self.wave_speed_val**2) * padded_dudx
        return np.stack([f1, f2], axis=0)

    def source(self, padded_state, dx):
        u, v = padded_state
        return np.stack([v, np.zeros_like(v)], axis=0)

    def wave_speed(self, padded_state):
        return self.wave_speed_val


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

    def flux(self, padded_state, dx):
        dudx = operators.gradient(padded_state, dx)
        ndim = dx.ndim if hasattr(dx, 'ndim') else 1
        pad_width = [(0, 0)] * (dudx.ndim - ndim) + [(1, 1)] * ndim
        padded_flux = np.pad(dudx, pad_width=pad_width, mode='edge')
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
        
        v_unpadded = state[1]
        h, v = padded_state
        q = h * v
        q_sq_by_h = np.where(h > eps, (q**2) / h, 0.0)
        
        dh_dt = -operator(q, grid)
        dq_dt = -operator((q_sq_by_h + 0.5 * g * (h**2)), grid)
        dv_dt = -v_unpadded * operator(v, grid) - g * operator(h, grid)
            
        return np.stack([dh_dt, dv_dt], axis=0)

    def flux(self, padded_cons, dx):
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
        h, q = conservative_state
        eps = 1e-5
        v = np.zeros_like(q)
        mask = h > eps
        v[mask] = q[mask] / h[mask]
        return np.stack([h, v], axis=0)


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

    def flux(self, padded_state, dx):
        adv_flux = 0.5*(padded_state)**2
        wave_flux = -self.viscosity*(operators.gradient(padded_state, dx))
        ndim = dx.ndim if hasattr(dx, 'ndim') else 1
        pad_width = [(0, 0)] * (wave_flux.ndim - ndim) + [(1, 1)] * ndim
        padded_flux = np.pad(wave_flux, pad_width=pad_width, mode='edge')

        return adv_flux + padded_flux

    def wave_speed(self, padded_state):
        return padded_state