import matplotlib.pyplot as plt
import numpy as np

class RendererRegistry:
    _registry = []

    @classmethod
    def register(cls, renderer_class):
        cls._registry.append(renderer_class)
        return renderer_class

    @classmethod
    def resolve(cls, field):
        for renderer_class in cls._registry:
            if renderer_class.can_render(field):
                return renderer_class
        rank = getattr(field, 'rank', 'unknown')
        ndim = getattr(field, 'ndim', 'unknown')
        raise ValueError(f"No renderer found for field: rank={rank}, ndim={ndim}")

class BaseRenderer:
    @classmethod
    def can_render(cls, field):
        return False
        
    def __init__(self, fig, gridspec_slots, config, initial_state=None):
        """
        fig: The main matplotlib Figure.
        gridspec_slots: Dictionary containing GridSpec SubplotSpecs for positioning.
                        e.g., {'primary': gs[0, :], 'secondary1': gs[1, 0], ...}
        config: Dictionary containing simulation configuration (dt, final_time, max_frames, etc.)
        """
        self.fig = fig
        self.gridspec_slots = gridspec_slots
        self.config = config
        self.initial_state = initial_state
        self.artists = []
        
        # Calculate dynamic range from initial_state to keep condition close to the ground
        if self.initial_state is not None:
            data = self.initial_state.data if hasattr(self.initial_state, 'data') else self.initial_state
            ndim = self.initial_state.grid.ndim if hasattr(self.initial_state, 'grid') else self.initial_state.ndim
            display_y = data[0] if data.ndim > ndim else data
            min_val = np.min(display_y)
            max_val = np.max(display_y)
            span = max_val - min_val
            if span < 1e-6:
                span = 1.0
            self.vmin = min_val - 0.1 * span
            self.vmax = max_val + 0.1 * span
        else:
            self.vmin = -0.5
            self.vmax = 1.5
            
        self.setup()

    def setup(self):
        pass

    def update(self, frame_idx, state, current_time, energies):
        pass
        
    def get_artists(self):
        return self.artists

@RendererRegistry.register
class Scalar1DRenderer(BaseRenderer):
    @classmethod
    def can_render(cls, field):
        return getattr(field, 'rank', '') == 'scalar' and getattr(field, 'ndim', 0) == 1

    def setup(self):
        self.ax_primary = self.fig.add_subplot(self.gridspec_slots['primary'])
        self.ax_secondary1 = self.fig.add_subplot(self.gridspec_slots['secondary1'])
        
        nx = self.config['shape'][0]
        dx = self.config['spacing'][0]
        self.x = np.arange(nx) * dx
        
        # Primary: Live Line Plot
        self.ax_primary.set_xlim(0, nx * dx)
        self.ax_primary.set_ylim(self.vmin, self.vmax)
        self.ax_primary.grid(True, which='both', color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        self.ax_primary.set_xlabel("Spatial Domain (x)")
        self.ax_primary.set_ylabel("Amplitude")
        self.ax_primary.set_title("1D Scalar Field", fontsize=12, fontweight="bold")
        self.line_pos, = self.ax_primary.plot([], [], color="#00ffff", lw=2.5, label="Primary Component")
        self.artists.append(self.line_pos)
        
        # Secondary1: Space-Time Map
        self.ax_secondary1.set_facecolor('black')
        self.max_frames = self.config['max_frames']
        self.history_matrix = np.zeros((self.max_frames, nx))
        
        self.im = self.ax_secondary1.imshow(
            self.history_matrix,
            aspect="auto",
            cmap="inferno",
            extent=[0, nx * dx, self.config['final_time'], 0],
            vmin=self.vmin, vmax=self.vmax,
            interpolation="nearest"
        )
        self.ax_secondary1.set_title("Space-Time Fingerprint Matrix", fontsize=10, fontweight="bold")
        self.ax_secondary1.set_xlabel("Spatial Domain (x)")
        self.ax_secondary1.set_ylabel("t")
        self.fig.colorbar(self.im, ax=self.ax_secondary1, fraction=0.046, pad=0.04)
        self.artists.append(self.im)
        
        # Telemetry Text
        self.txt = self.fig.text(0.5, 0.015, "", color="white", fontsize=9, fontfamily="monospace",
                                 ha="center", bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none"))
        self.artists.append(self.txt)

    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        data = state.data if hasattr(state, 'data') else state
        display_y = data[0] if data.ndim > 1 else data
        
        self.line_pos.set_data(self.x, display_y)
        
        if frame_idx < self.max_frames:
            self.history_matrix[frame_idx] = np.abs(display_y)
            self.im.set_data(self.history_matrix)
                
        cfl = self.config['dt'] / self.config['characteristic_spacing']
        if frame_idx % 5 == 0:
            self.txt.set_text(
                f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
                f"ENERGY: {energies[2]:.3f}J"
            )

@RendererRegistry.register
class Scalar2DRenderer(BaseRenderer):
    @classmethod
    def can_render(cls, field):
        ndim = getattr(field, 'ndim', 0)
        rank = getattr(field, 'rank', '')
        components = getattr(field, 'components', 1)
        # Handles true scalar fields (rank=scalar) AND single-component vector
        # fields (e.g. GaussianIC(num_fields=1) for advection/diffusion/burgers)
        if ndim != 2:
            return False
        if rank == 'scalar':
            return True
        if rank == 'vector' and components == 1:
            return True
        return False

    def setup(self):
        # primary = heatmap
        # secondary1 = surface
        # secondary2 = horizontal cross section
        # secondary3 = vertical cross section
        self.ax_primary = self.fig.add_subplot(self.gridspec_slots['primary'])
        self.ax_surface = self.fig.add_subplot(self.gridspec_slots['secondary1'], projection='3d')
        self.ax_hcross = self.fig.add_subplot(self.gridspec_slots['secondary2'])
        self.ax_vcross = self.fig.add_subplot(self.gridspec_slots['secondary3'])
        
        ny, nx = self.config['shape']
        dy, dx = self.config['spacing']
        
        self.Y, self.X = np.meshgrid(
            np.arange(ny) * dy, np.arange(nx) * dx, indexing='ij'
        )
        
        self.current_map = np.zeros((ny, nx))
        
        # Heatmap
        self.im = self.ax_primary.imshow(
            self.current_map,
            aspect="auto",
            cmap="plasma",
            extent=[0, nx * dx, 0, ny * dy],
            vmin=self.vmin, vmax=self.vmax,
            origin="lower",
            interpolation="bicubic"
        )
        self.ax_primary.set_title("2D Field Heatmap", fontsize=12, fontweight="bold")
        self.ax_primary.set_xlabel("X Axis")
        self.ax_primary.set_ylabel("Y Axis")
        self.ax_primary.grid(True, which='both', color='white', linestyle='-', linewidth=1.5, alpha=0.5)
        self.ax_primary.set_axisbelow(False)
        self.artists.append(self.im)
        
        # Surface
        self.ax_surface.set_title("3D Surface", fontsize=10)
        self.ax_surface.set_zlim(self.vmin, self.vmax)
        self.ax_surface.grid(True, which='both', color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        self.surf = self.ax_surface.plot_surface(
            self.X, self.Y, self.current_map, cmap='plasma', edgecolor='none',
            rstride=2, cstride=2, vmin=self.vmin, vmax=self.vmax
        )
        self.artists.append(self.surf) # Note: surface blitting is tricky, might need re-adding
        
        # Cross sections
        self.ax_hcross.set_title("Horizontal Centerline", fontsize=10)
        self.ax_hcross.set_xlim(0, nx * dx)
        self.ax_hcross.set_ylim(self.vmin, self.vmax)
        self.ax_hcross.grid(True, which='both', color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        self.line_h, = self.ax_hcross.plot([], [], color="#00ffff", lw=2.5)
        self.artists.append(self.line_h)
        
        self.ax_vcross.set_title("Vertical Centerline", fontsize=10)
        self.ax_vcross.set_xlim(0, ny * dy)
        self.ax_vcross.set_ylim(self.vmin, self.vmax)
        self.ax_vcross.grid(True, which='both', color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
        self.line_v, = self.ax_vcross.plot([], [], color="#ff007f", lw=2.5)
        self.artists.append(self.line_v)
        
        # Telemetry
        self.txt = self.fig.text(0.5, 0.015, "", color="white", fontsize=9, fontfamily="monospace",
                                 ha="center", bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none"))
        self.artists.append(self.txt)
        
    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        data = state.data if hasattr(state, 'data') else state
        display_y = data[0] if data.ndim > 2 else data
        
        self.im.set_data(display_y)
        
        # Cross sections
        ny, nx = display_y.shape
        self.line_h.set_data(self.X[ny//2, :], display_y[ny//2, :])
        self.line_v.set_data(self.Y[:, nx//2], display_y[:, nx//2])
        
        # Surface recreation (blitting 3d is limited, standard replace)
        self.surf.remove()
        if self.surf in self.artists:
            self.artists.remove(self.surf)
        self.surf = self.ax_surface.plot_surface(
            self.X, self.Y, display_y, cmap='plasma', edgecolor='none',
            rstride=2, cstride=2, vmin=self.vmin, vmax=self.vmax
        )
        self.artists.append(self.surf)
            
        cfl = self.config['dt'] / self.config['characteristic_spacing']
        if frame_idx % 5 == 0:
            self.txt.set_text(
                f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
                f"ENERGY: {energies[2]:.3f}J"
            )

@RendererRegistry.register
class Vector1DRenderer(Scalar1DRenderer):
    @classmethod
    def can_render(cls, field):
        return getattr(field, 'rank', '') == 'vector' and getattr(field, 'ndim', 0) == 1

@RendererRegistry.register
class Vector2DRenderer(Scalar2DRenderer):
    @classmethod
    def can_render(cls, field):
        ndim = getattr(field, 'ndim', 0)
        rank = getattr(field, 'rank', '')
        components = getattr(field, 'components', 1)
        # Only claim multi-component vector fields (shallow water, etc.)
        return ndim == 2 and rank == 'vector' and components >= 2

    def setup(self):
        super().setup()
        # Override primary plot to clear imshow and prepare for contour/quiver
        self.ax_primary.clear()
        self.ax_primary.set_title("2D Contour & Velocity Field", fontsize=12, fontweight="bold")
        self.ax_primary.set_xlabel("X Axis")
        self.ax_primary.set_ylabel("Y Axis")
        self.ax_primary.grid(True, which='both', color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        # Update 3D Surface, Cross Sections, and Telemetry using base class update
        # (which expects state shape component 0 to render the 3D height)
        data = state.data if hasattr(state, 'data') else state
        display_y = data[0]
        
        # 3D surface update
        self.surf.remove()
        if self.surf in self.artists:
            self.artists.remove(self.surf)
        self.surf = self.ax_surface.plot_surface(
            self.X, self.Y, display_y, cmap='cool', edgecolor='k', linewidth=0.2,
            rstride=2, cstride=2, vmin=self.vmin, vmax=self.vmax
        )
        self.artists.append(self.surf)
        
        # Cross sections update
        ny, nx = display_y.shape
        self.line_h.set_data(self.X[ny//2, :], display_y[ny//2, :])
        self.line_v.set_data(self.Y[:, nx//2], display_y[:, nx//2])
        
        cfl = self.config['dt'] / self.config['characteristic_spacing']
        if frame_idx % 5 == 0:
            self.txt.set_text(
                f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
                f"ENERGY: {energies[2]:.3f}J"
            )
            
        # Draw Contour & Quiver plot on the primary 2D axis
        self.ax_primary.clear()
        self.ax_primary.set_title("2D Contour & Velocity Field", fontsize=12, fontweight="bold")
        self.ax_primary.set_xlabel("X Axis")
        self.ax_primary.set_ylabel("Y Axis")
        
        is_wave = self.config.get('eq_name') == 'wave'
        h = data[0]
        if is_wave:
            u = np.zeros_like(h)
            v = np.zeros_like(h)
        elif data.shape[0] >= 3:
            # Shallow Water: [h, qx, qy] or primitive [h, u, v]
            u = data[1]
            v = data[2]
        else:
            u = data[0]
            v = data[1]
            h = np.sqrt(u**2 + v**2)
            
        # Draw Contours of height/magnitude (25 levels for high resolution)
        span = self.vmax - self.vmin
        levels = np.linspace(self.vmin, self.vmax, 25)
        
        if span > 1e-4:
            # Filled contours
            contours_f = self.ax_primary.contourf(self.X, self.Y, h, levels=levels, cmap='cool', extend='both')
            # Labeled contour lines overlay
            if (np.max(h) - np.min(h)) > 1e-4:
                contours = self.ax_primary.contour(self.X, self.Y, h, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
                self.ax_primary.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            
            # Dynamic colorbar handling
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(contours_f, ax=self.ax_primary, fraction=0.046, pad=0.04)
                self.cbar.set_label("Magnitude / Depth")
            else:
                self.cbar.update_normal(contours_f)
            
        # Draw Quiver arrows for velocity vectors (subsample to avoid overcrowding, scaled, 20% faded, skipped for wave)
        if not is_wave:
            sub = max(1, min(nx, ny) // 20)
            self.ax_primary.quiver(
                self.X[::sub, ::sub], self.Y[::sub, ::sub],
                u[::sub, ::sub], v[::sub, ::sub],
                color='black', scale=None, width=0.003, alpha=0.8
            )
