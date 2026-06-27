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
        
    def __init__(self, fig, gridspec_slots, config):
        """
        fig: The main matplotlib Figure.
        gridspec_slots: Dictionary containing GridSpec SubplotSpecs for positioning.
                        e.g., {'primary': gs[0, :], 'secondary1': gs[1, 0], ...}
        config: Dictionary containing simulation configuration (dt, final_time, max_frames, etc.)
        """
        self.fig = fig
        self.gridspec_slots = gridspec_slots
        self.config = config
        self.artists = []
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
        self.ax_primary.set_ylim(-1.5, 1.5)
        self.ax_primary.grid(True, linestyle="--", alpha=0.5)
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
            vmin=0.0, vmax=1.0,
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
        return getattr(field, 'rank', '') == 'scalar' and getattr(field, 'ndim', 0) == 2

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
            cmap="inferno",
            extent=[0, nx * dx, 0, ny * dy],
            vmin=-0.5, vmax=1.5,
            origin="lower",
            interpolation="bicubic"
        )
        self.ax_primary.set_title("2D Field Heatmap", fontsize=12, fontweight="bold")
        self.ax_primary.set_xlabel("X Axis")
        self.ax_primary.set_ylabel("Y Axis")
        self.artists.append(self.im)
        
        # Surface
        self.ax_surface.set_title("3D Surface", fontsize=10)
        self.ax_surface.set_zlim(-0.5, 1.5)
        self.surf = self.ax_surface.plot_surface(
            self.X, self.Y, self.current_map, cmap='inferno', edgecolor='none',
            rstride=2, cstride=2, vmin=-0.5, vmax=1.5
        )
        self.artists.append(self.surf) # Note: surface blitting is tricky, might need re-adding
        
        # Cross sections
        self.ax_hcross.set_title("Horizontal Centerline", fontsize=10)
        self.ax_hcross.set_xlim(0, nx * dx)
        self.ax_hcross.set_ylim(-0.5, 1.5)
        self.line_h, = self.ax_hcross.plot([], [], color="#00ffff")
        self.artists.append(self.line_h)
        
        self.ax_vcross.set_title("Vertical Centerline", fontsize=10)
        self.ax_vcross.set_xlim(0, ny * dy)
        self.ax_vcross.set_ylim(-0.5, 1.5)
        self.line_v, = self.ax_vcross.plot([], [], color="#ff007f")
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
        self.surf = self.ax_surface.plot_surface(
            self.X, self.Y, display_y, cmap='inferno', edgecolor='none',
            rstride=2, cstride=2, vmin=-0.5, vmax=1.5
        )
            
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
        return getattr(field, 'rank', '') == 'vector' and getattr(field, 'ndim', 0) == 2
