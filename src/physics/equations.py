import numpy as np
from src.numerics import operators
from src.mesh.boundaries import Dirichlet

class Equation:
    """Base class for all Tempest equations."""
    scalar_label = "State"

    def __call__(self, t, state_data, dx, boundary, operator):
        raise NotImplementedError

    def get_velocity(self, state_data, dx, boundary, operator):
        """Return derived (u, v) velocity for visualization, or None."""
        return None

    def get_surface_scalar(self, state_data, dx, boundary, operator):
        """Return an alternate scalar for the 3D surface plot, or None."""
        return None

    def parity(self, grid_ndim):
        """Returns the reflection parity of each component. By default, scalar fields are fully symmetric [1]."""
        return [1]

    def flux(self, padded_state, dx):
        """
        Computes the spatial flux for the equation.
        
        Expected Return Shapes:
        - Scalar equations: (grid_ndim, ...)
        - System equations: (num_components, grid_ndim, ...)
        """
        raise NotImplementedError

    def wave_speed(self, padded_state):
        raise NotImplementedError

    #def source(self, padded_state, dx):
        #return None

    def compute_energies(self, state_data, dx, boundary):
        """Generic energy computation (L2 norm)."""
        dV = np.prod(dx)
        total_e = np.sum(state_data**2) * dV
        return 0.0, 0.0, total_e

    #def source(self, t, state, grid):
        #return np.zeros_like(state)


class AdvectionEquation(Equation):
    """Linear advection equation."""
    scalar_label = "Scalar"

    def __init__(self, velocity):
        self.__name__ = 'advection'
        self.velocity = velocity
        self.spatial_order = 1

    def parity(self, grid_ndim):
        return [1]

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Linear advection is a 1st-order spatial PDE. "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )
            
        vel = self.velocity
        if isinstance(vel, (list, tuple, np.ndarray)):
            vel = np.asarray(vel)
            if vel.ndim == 1 and len(dx) > 1:
                vel = vel.reshape((len(dx),) + (1,) * len(dx))

        # Strip leading component axis if present (e.g. VectorField with 1 component)
        # so operators work on the pure spatial array
        ndim = len(dx)
        has_component_axis = state_data.ndim > ndim
        if has_component_axis:
            u = state_data[0]  # shape: (*grid_shape)
        else:
            u = state_data

        padded_u = boundary(u, ndim, self.parity(ndim))
        grad_u = operator(padded_u, dx, velocity=vel)
        
        if ndim > 1:
            # vel broadcast shape: (ndim, 1, ...) against grad_u shape: (ndim, *grid)
            dudt = -np.sum(vel * grad_u, axis=0)
        else:
            dudt = -vel * grad_u
        
        # Restore component axis if it was present
        if has_component_axis:
            return dudt[np.newaxis, ...]
        return dudt

    def flux(self, padded_state, dx):
        """Returns scalar flux vector: shape (grid_ndim, ...)"""
        # Linear advection flux is simply the velocity field advecting the state.
        # Since velocity can be a vector, the flux takes shape (grid_ndim, ...)
        if np.isscalar(self.velocity):
            return np.stack([self.velocity * padded_state] * len(dx), axis=0)
        return self.velocity * padded_state

    def wave_speed(self, padded_state):
        return self.velocity


class WaveEquation(Equation):
    """Wave propagation equation."""
    scalar_label = "Displacement"

    def __init__(self, wave_speed):
        self.__name__ = 'wave'
        self.wave_speed_val = wave_speed
        self.spatial_order = 1

    def parity(self, grid_ndim):
        # State has exactly 2 components: [u (position), du/dt (velocity)].
        # Position is symmetric (+1) and velocity is antisymmetric (-1) at a reflecting wall.
        # This is independent of the number of spatial dimensions.
        return [1, -1]

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ in ('gradient', 'upwind'):
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Wave propagation is a 2nd-order spatial PDE. "
                "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
            )
        
        padded_state = boundary(state_data, len(dx), self.parity(len(dx)))
        u, v = padded_state
        
        d2udx2 = operator(u, dx)
        d2udt2 = (self.wave_speed_val**2) * d2udx2
        
        return np.stack([state_data[1], d2udt2], axis=0)

    def flux(self, padded_state, dx):
        """Returns system flux tensor: shape (num_components, grid_ndim, ...)"""
        u, v = padded_state
        grad_u = operators.gradient(u, dx)
        ndim = len(dx)
        pad_width = [(0, 0)] * (grad_u.ndim - ndim) + [(1, 1)] * ndim
        padded_grad_u = np.pad(grad_u, pad_width=pad_width, mode='edge')
        
        f1 = np.zeros_like(padded_grad_u)
        f2 = -(self.wave_speed_val**2) * padded_grad_u
        return np.stack([f1, f2], axis=0)

    #def source(self, padded_state, dx):
        #u, v = padded_state
        #return np.stack([v, np.zeros_like(v)], axis=0)

    def wave_speed(self, padded_state):
        return self.wave_speed_val

    def compute_energies(self, state_data, dx, boundary):
        dV = np.prod(dx)
        u, v = state_data[0], state_data[1]
        
        u_padded = boundary(u, len(dx))
        grad_u = operators.gradient(u_padded, dx)
        
        pe = ((self.wave_speed_val**2)/2) * np.sum(grad_u**2) * dV
        ke = (1/2) * np.sum(v**2) * dV
        return pe, ke, pe + ke


class DiffusionEquation(Equation):
    """Diffusion equation."""
    scalar_label = "Scalar"

    def __init__(self, diffusivity):
        self.__name__ = 'diffusion'
        self.diffusivity = diffusivity
        self.spatial_order = 2

    def parity(self, grid_ndim):
        return [1]

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ in ('gradient', 'upwind'):
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Diffusion is a 2nd-order spatial PDE. "
                "You cannot pass 'gradient' or 'upwind' (1st-order) as its operator."
            )
        
        ndim = len(dx)
        has_component_axis = state_data.ndim > ndim
        u = state_data[0] if has_component_axis else state_data

        padded_u = boundary(u, ndim, self.parity(ndim))
        d2udx2 = operator(padded_u, dx)
        du_dt = self.diffusivity * d2udx2
        
        if has_component_axis:
            return du_dt[np.newaxis, ...]
        return du_dt

    def flux(self, padded_state, dx):
        """Returns scalar flux vector: shape (grid_ndim, ...)"""
        grad_u = operators.gradient(padded_state, dx)
        ndim = len(dx)
        pad_width = [(0, 0)] * (grad_u.ndim - ndim) + [(1, 1)] * ndim
        padded_flux = np.pad(grad_u, pad_width=pad_width, mode='edge')
        return -self.diffusivity * padded_flux

    def wave_speed(self, padded_state):
        return 0.0


class ShallowWaterEquation(Equation):
    """Shallow water equation."""
    scalar_label = "Height (h)"

    def __init__(self):
        self.__name__ = 'shallow_water'
        self.spatial_order = 1
    def parity(self, grid_ndim):
        return [1] + [-1] * grid_ndim

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: Shallow water equation is a 1st-order spatial PDE. "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )
        
        padded_state = boundary(state_data, len(dx), self.parity(len(dx)))
        g = 9.81 
        eps = 1e-5
        
        grid_ndim = len(dx)
        v_unpadded = state_data[1:] # shape (grid_ndim, ...)
        h = padded_state[0]
        v = padded_state[1:]      # shape (grid_ndim, ...)
        q = h * v                 # shape (grid_ndim, ...)
        
        grad_q = operator(q, dx)
        if grid_ndim > 1 and grad_q.ndim == grid_ndim + 2:
            dh_dt = -np.sum([grad_q[i, i] for i in range(grid_ndim)], axis=0)
        else:
            dh_dt = -grad_q
        
        dv_dt = np.zeros_like(v_unpadded)
        
        for i in range(grid_ndim):
            v_grad_i = operator(v[i], dx)
            v_dot_grad = np.sum(v_unpadded * v_grad_i, axis=0)
            grad_h = operator(h, dx)
            dv_dt[i] = -v_dot_grad - g * grad_h[i]
            
        return np.concatenate([dh_dt[np.newaxis, ...], dv_dt], axis=0)

    def flux(self, padded_cons, dx, **kwargs):
        """Returns system flux tensor: shape (num_components, grid_ndim, ...)"""
        grid_ndim = len(dx)
        h = padded_cons[0]
        q = padded_cons[1:]
        g = 9.81
        eps = 1e-5
        
        F = np.zeros((1 + grid_ndim, grid_ndim) + h.shape)
        F[0, :] = q
        
        mask = h > eps
        for d in range(grid_ndim):
            for i in range(grid_ndim):
                q_i_q_d_over_h = np.zeros_like(h)
                q_i_q_d_over_h[mask] = (q[i][mask] * q[d][mask]) / h[mask]
                F[1+i, d] = q_i_q_d_over_h
                if i == d:
                    F[1+i, d] += 0.5 * g * h**2
                    
        return F

    def wave_speed(self, padded_state):
        h = padded_state[0]
        q = padded_state[1:]
        g = 9.81
        eps = 1e-5
        v_mag = np.zeros_like(h)
        mask = h > eps
        q_sq = np.sum(q**2, axis=0)
        v_mag[mask] = np.sqrt(q_sq[mask]) / h[mask]
        return v_mag + np.sqrt(g * h)
        
    def to_conservative(self, primitive_state):
        h = primitive_state[0]
        v = primitive_state[1:]
        q = h * v
        return np.concatenate([h[np.newaxis, ...], q], axis=0)
        
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
        return np.concatenate([h[np.newaxis, ...], v], axis=0)

    def compute_energies(self, state_data, dx, boundary):
        dV = np.prod(dx)
        h = state_data[0]
        v = state_data[1:]
        g = 9.81
        
        ke = 0.5 * np.sum(h * np.sum(v**2, axis=0)) * dV
        pe = 0.5 * g * np.sum(h**2) * dV
        return pe, ke, pe + ke


class BurgersEquation(Equation):
    """Burgers' equation."""
    scalar_label = "Velocity (u)"

    def __init__(self, viscosity):
        self.__name__ = 'burgers'
        self.viscosity = viscosity
        self.spatial_order = 1

    def parity(self, grid_ndim):
        return [1]

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: You are controlling the operator of the advection term in Burgers' equation "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )

        ndim = len(dx)
        has_component_axis = state_data.ndim > ndim
        u = state_data[0] if has_component_axis else state_data

        padded_u = boundary(u, ndim, self.parity(ndim))

        wave_term = self.viscosity * operators.laplacian(padded_u, dx)
        adv_term = -operator(0.5*(padded_u)**2, dx, velocity=padded_u)
        
        if ndim > 1:
            adv_term = np.sum(adv_term, axis=0)

        du_dt = wave_term + adv_term
        
        if has_component_axis:
            return du_dt[np.newaxis, ...]
        return du_dt

    def flux(self, padded_state, dx):
        """Returns scalar flux vector: shape (num_components, grid_ndim, ...)"""
        adv_flux = 0.5 * (padded_state)**2
        
        grid_ndim = len(dx)
        if grid_ndim > 1:
            F_adv = np.stack([adv_flux] * grid_ndim, axis=1)
        else:
            F_adv = adv_flux[np.newaxis, ...]
            
        # Compute viscous flux using a central difference that preserves the
        # full padded shape.
        grads = []
        for i in range(grid_ndim):
            spacing = dx[i]
            ax = -(grid_ndim - i)
            grad = (np.roll(padded_state, -1, axis=ax) - np.roll(padded_state, 1, axis=ax)) / (2.0 * spacing)
            grads.append(grad)
            
        if grid_ndim > 1:
            wave_flux = -self.viscosity * np.stack(grads, axis=1)
        else:
            wave_flux = -self.viscosity * grads[0][np.newaxis, ...]
            
        return F_adv + wave_flux

    def wave_speed(self, padded_state):
        return padded_state

    def compute_energies(self, state_data, dx, boundary):
        dV = np.prod(dx)
        total_e = np.sum(0.5 * state_data**2) * dV
        return 0.0, 0.0, total_e

class RossbyWave(Equation):
    """Rossby wave equation."""
    scalar_label = "Potential vorticity (q)"

    def __init__(self, beta, source=None):
        self.__name__ = 'rossby_wave'
        self.beta = beta
        self.poisson = None
        self.source = source
        self.spatial_order = 1

    def parity(self, grid_ndim):
        return [-1]

    def __call__(self, t, state_data, dx, boundary, operator):

        q = state_data[0]

        if self.poisson is None:
            self.poisson = operators.PoissonSolver(q.shape, dx)

        psi = self.poisson.solve(q)

        padded_psi = boundary(psi, len(dx), self.parity(len(dx)))

        dpsi_dx = operators.gradient(padded_psi, dx)[1]

        dq_dt = -self.beta*dpsi_dx

        rhs = np.stack([dq_dt], axis=0)

        if self.source is not None:
            src = self.source(t)
            rhs += src

        return rhs

    def compute_energies(self, state_data, dx, boundary):
        dV = np.prod(dx)
        q = state_data[0]

        if self.poisson is None:
            self.poisson = operators.PoissonSolver(q.shape, dx)

        psi = self.poisson.solve(q)

        padded_psi = boundary(psi, len(dx), self.parity(len(dx)))

        dpsi_dx = operators.gradient(padded_psi, dx)[1]
        dpsi_dy = operators.gradient(padded_psi, dx)[0]

        energy = 0.5 * np.sum(dpsi_dx**2 + dpsi_dy**2) * dV
        return 0.0, 0.0, energy

class BarotropicVorticity(Equation):
    scalar_label = "Vorticity (ζ)"

    def __init__(self, beta, nu, source=None):
        self.__name__ = 'barotropic_voricity'
        self.beta = beta
        self.nu = nu
        self.source = source
        self.spatial_order = 1
        self.poisson = None

    def parity(self, grid_ndim):
        return [1]

    def __call__(self, t, state_data, dx, boundary, operator):
        if operator.__name__ == 'laplacian':
            raise ValueError(
                "CRITICAL PHYSICS ERROR: You are controlling the operator of the voricity and advection term in Barotropic Voricity equation "
                "You cannot pass 'laplacian' (2nd-order) as its operator."
            )

        if not isinstance(boundary, Dirichlet):
            raise ValueError(
                "CRITICAL PHYSICS ERROR: BarotropicVorticity currently requires Dirichlet boundaries "
                "when using PoissonSolver."
            )

        zeta = state_data[0]

        if self.poisson is None:
            self.poisson = operators.PoissonSolver(zeta.shape, dx)

        psi = self.poisson.solve(zeta)

        padded_psi = boundary(psi, len(dx), self.parity(len(dx)))

        padded_zeta = boundary(zeta, len(dx), self.parity(len(dx)))

        dpsi = operator(padded_psi, dx)
        dzeta = operator(padded_zeta, dx)

        dpsi_dy, dpsi_dx = dpsi
        dzeta_dy, dzeta_dx = dzeta

        jacob = dpsi_dx * dzeta_dy - dpsi_dy * dzeta_dx

        dzeta_dt = -jacob - self.beta*dpsi_dx + self.nu*operators.laplacian(padded_zeta, dx)

        rhs = np.stack([dzeta_dt], axis=0)

        if self.source is not None:
            src = self.source(t)
            rhs += src

        return rhs

    def _solve_psi(self, state_data, dx):
        zeta = state_data[0]
        if self.poisson is None:
            self.poisson = operators.PoissonSolver(zeta.shape, dx)
        return self.poisson.solve(zeta)

    def get_surface_scalar(self, state_data, dx, boundary, operator):
        return self._solve_psi(state_data, dx)

    def get_velocity(self, state_data, dx, boundary, operator):
        psi = self._solve_psi(state_data, dx)
        padded_psi = boundary(psi, len(dx), self.parity(len(dx)))
        dpsi_dy, dpsi_dx = operator(padded_psi, dx)
        return -dpsi_dy, dpsi_dx