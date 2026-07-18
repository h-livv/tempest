import matplotlib.pyplot as plt
import numpy as np
from src.visualization.theme import (
    AXES_FACE,
    GRID,
    PRIMARY,
    SCALAR_FIELD_CMAP,
    SECONDARY,
    SURFACE_MESH_WIDTH,
    TELEMETRY_BG,
    TELEMETRY_TEXT,
    scalar_field_facecolors,
    scalar_field_norm,
    style_3d_axis,
    style_axis,
    style_colorbar,
    scalar_axis_label,
    surface_magnitude_facecolors,
    surface_mesh_stride,
    format_solver_telemetry,
)

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
            self.initial_span = span
            self.vmin = min_val - 0.1 * span
            self.vmax = max_val + 0.1 * span
        else:
            self.initial_span = 2.0
            self.vmin = -0.5
            self.vmax = 1.5

        self.scalar_label = scalar_axis_label(
            self.config.get('eq_name'),
            self.config.get('scalar_label'),
        )
            
        self.setup()

    def setup(self):
        pass

    def update(self, frame_idx, state, current_time, energies):
        pass
        
    def get_artists(self):
        return self.artists


CONTOUR_LINE_THRESHOLD = 0.001
TOP_CONTOUR_CUTOFF = 0.20       # skip lowest 30% of [min, max] (top ~70% band)
TOP_CONTOUR_LEVEL_COUNT = 16    # dense levels within that band


def _top_contour_levels(values, cutoff=TOP_CONTOUR_CUTOFF, n_levels=TOP_CONTOUR_LEVEL_COUNT):
    """Contour levels in the upper portion of the field range, densely sampled."""
    values = np.asarray(values)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span < 1e-6:
        return np.array([vmax])
    vmin_floor = vmin + cutoff * span
    return np.linspace(vmin_floor, vmax, n_levels)


def _line_contour_levels(levels, threshold=CONTOUR_LINE_THRESHOLD):
    """Keep only contour line levels with |value| above the threshold."""
    levels = np.asarray(levels)
    return levels[np.abs(levels) > threshold]


def _plot_field_surface(ax, X, Y, z, zlim, norm, cmap=SCALAR_FIELD_CMAP):
    """Draw a 3D surface colored by scalar values with a shared norm."""
    ax.set_zlim(zlim)
    ny, nx = z.shape
    rstride = surface_mesh_stride(ny)
    cstride = surface_mesh_stride(nx)
    return ax.plot_surface(
        X, Y, z,
        facecolors=scalar_field_facecolors(z, norm, cmap),
        edgecolor='black',
        linewidth=SURFACE_MESH_WIDTH,
        shade=False,
        rstride=rstride,
        cstride=cstride,
    )


def _plot_gradient_surface(ax, X, Y, z, zlim):
    """Draw a 3D surface colored by |z| with a thin black mesh overlay."""
    ax.set_zlim(zlim)
    ny, nx = z.shape
    rstride = surface_mesh_stride(ny)
    cstride = surface_mesh_stride(nx)
    return ax.plot_surface(
        X, Y, z,
        facecolors=surface_magnitude_facecolors(z),
        edgecolor='black',
        linewidth=SURFACE_MESH_WIDTH,
        shade=False,
        rstride=rstride,
        cstride=cstride,
    )

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
        style_axis(self.ax_primary)
        self.ax_primary.set_xlabel("Spatial Domain (x)")
        self.ax_primary.set_ylabel("Amplitude")
        self.ax_primary.set_title("1D Scalar Field", fontsize=12, fontweight="bold")
        self.line_pos, = self.ax_primary.plot([], [], color=PRIMARY, lw=2.5, label="Primary Component")
        self.artists.append(self.line_pos)
        
        # Secondary1: Space-Time Map
        self.ax_secondary1.set_facecolor(AXES_FACE)
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
        style_axis(self.ax_secondary1, grid=False)
        self.ax_secondary1.set_title("Space-Time Fingerprint Matrix", fontsize=10, fontweight="bold")
        self.ax_secondary1.set_xlabel("Spatial Domain (x)")
        self.ax_secondary1.set_ylabel("t")
        cbar = self.fig.colorbar(self.im, ax=self.ax_secondary1, fraction=0.046, pad=0.04)
        style_colorbar(cbar)
        self.artists.append(self.im)
        
        # Telemetry Text
        self.txt = self.fig.text(
            0.5, 0.015, "", color=TELEMETRY_TEXT, fontsize=9, fontfamily="monospace",
            ha="center", bbox=dict(facecolor=TELEMETRY_BG, alpha=0.9, edgecolor=GRID),
        )
        self.artists.append(self.txt)

    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        data = state.data if hasattr(state, 'data') else state
        display_y = data[0] if data.ndim > 1 else data
        
        self.line_pos.set_data(self.x, display_y)
        
        if frame_idx < self.max_frames:
            self.history_matrix[frame_idx] = np.abs(display_y)
            self.im.set_data(self.history_matrix)
                
        if frame_idx % 5 == 0:
            self.txt.set_text(format_solver_telemetry(
                self.config.get('eq_name'), scheme_name, current_time, frame_idx,
                self.config['dt'] / self.config['characteristic_spacing'], energies,
            ))

@RendererRegistry.register
class Scalar2DRenderer(BaseRenderer):
    @classmethod
    def can_render(cls, field):
        ndim = getattr(field, 'ndim', 0)
        if ndim != 2:
            return False
        return not getattr(field, 'has_flow', False)

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
            cmap="cool",
            extent=[0, nx * dx, 0, ny * dy],
            vmin=self.vmin, vmax=self.vmax,
            origin="lower",
            interpolation="bicubic"
        )
        style_axis(self.ax_primary, grid_alpha=0.4)
        self.ax_primary.set_title("2D Field Heatmap", fontsize=12, fontweight="bold")
        self.ax_primary.set_xlabel("X Axis")
        self.ax_primary.set_ylabel("Y Axis")
        self.ax_primary.set_axisbelow(False)
        self.artists.append(self.im)
        
        # Surface
        self.ax_surface.set_title("3D Surface", fontsize=12)
        self.ax_surface.set_xlabel("X")
        self.ax_surface.set_ylabel("Y")
        self.ax_surface.set_zlabel(self.scalar_label)
        self.ax_surface.set_zlim(self.vmin, self.vmax)
        style_3d_axis(self.ax_surface)
        self.surf = _plot_gradient_surface(
            self.ax_surface, self.X, self.Y, self.current_map, (self.vmin, self.vmax),
        )
        self.artists.append(self.surf)
        
        # Cross sections
        self.ax_hcross.set_title("Horizontal Centerline", fontsize=10)
        self.ax_hcross.set_xlim(0, nx * dx)
        self.ax_hcross.set_ylim(self.vmin, self.vmax)
        style_axis(self.ax_hcross)
        self.line_h, = self.ax_hcross.plot([], [], color=PRIMARY, lw=2.5)
        self.artists.append(self.line_h)
        
        self.ax_vcross.set_title("Vertical Centerline", fontsize=10)
        self.ax_vcross.set_xlim(0, ny * dy)
        self.ax_vcross.set_ylim(self.vmin, self.vmax)
        style_axis(self.ax_vcross)
        self.line_v, = self.ax_vcross.plot([], [], color=SECONDARY, lw=2.5)
        self.artists.append(self.line_v)
        
        # Telemetry
        self.txt = self.fig.text(0.5, 0.015, "", color="white", fontsize=9, fontfamily="monospace",
                                 ha="center", bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none"))
        self.artists.append(self.txt)
        
    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        display_y = state.get_scalar() if hasattr(state, 'get_scalar') else (
            state.data[0] if state.data.ndim > 2 else state.data
        )

        self.im.set_data(display_y)

        ny, nx = display_y.shape
        self.line_h.set_data(self.X[ny // 2, :], display_y[ny // 2, :])
        self.line_v.set_data(self.Y[:, nx // 2], display_y[:, nx // 2])

        # Surface recreation (blitting 3d is limited, standard replace)
        self.surf.remove()
        if self.surf in self.artists:
            self.artists.remove(self.surf)

        self.surf = _plot_gradient_surface(
            self.ax_surface, self.X, self.Y, display_y, (self.vmin, self.vmax),
        )
        self.artists.append(self.surf)
            
        if frame_idx % 5 == 0:
            self.txt.set_text(format_solver_telemetry(
                self.config.get('eq_name'), scheme_name, current_time, frame_idx,
                self.config['dt'] / self.config['characteristic_spacing'], energies,
            ))

@RendererRegistry.register
class Vector1DRenderer(Scalar1DRenderer):
    @classmethod
    def can_render(cls, field):
        return getattr(field, 'rank', '') == 'vector' and getattr(field, 'ndim', 0) == 1

@RendererRegistry.register
class Vector2DRenderer(BaseRenderer):
    """Two-panel layout for 2D fields with flow: surface | contour heatmap."""

    @classmethod
    def can_render(cls, field):
        return getattr(field, 'ndim', 0) == 2 and getattr(field, 'has_flow', False)

    def setup(self):
        self.ax_surface = self.fig.add_subplot(self.gridspec_slots['surface'], projection='3d')
        self.ax_heatmap = self.fig.add_subplot(self.gridspec_slots['heatmap'])
        self.ax_hcross = self.fig.add_subplot(self.gridspec_slots['secondary2'])
        self.ax_vcross = self.fig.add_subplot(self.gridspec_slots['secondary3'])

        ny, nx = self.config['shape']
        dy, dx = self.config['spacing']
        self.Y, self.X = np.meshgrid(
            np.arange(ny) * dy, np.arange(nx) * dx, indexing='ij'
        )

        self.current_map = np.zeros((ny, nx))
        self.heatmap_cbar = None
        init_scalar = (
            self.initial_state.get_scalar()
            if self.initial_state is not None and hasattr(self.initial_state, 'get_scalar')
            else self.current_map
        )
        self.field_norm = scalar_field_norm(init_scalar)

        # 3D surface (left) — scalar field with shared norm coloring
        self.ax_surface.set_title("3D Surface", fontsize=12)
        self.ax_surface.set_xlabel("X")
        self.ax_surface.set_ylabel("Y")
        self.ax_surface.set_zlabel(self.scalar_label)
        self.ax_surface.set_zlim(self.vmin, self.vmax)
        style_3d_axis(self.ax_surface)
        self.surf = _plot_field_surface(
            self.ax_surface, self.X, self.Y, self.current_map, (self.vmin, self.vmax), self.field_norm,
        )
        self.artists.append(self.surf)

        # Heatmap (right) — contour fill + overlays drawn each frame
        style_axis(self.ax_heatmap, grid_style=":", grid_width=0.8, grid_alpha=0.5)
        self.ax_heatmap.set_title("2D Contour Field", fontsize=12)
        self.ax_heatmap.set_xlabel("X Axis")
        self.ax_heatmap.set_ylabel("Y Axis")

        # Cross sections
        self.ax_hcross.set_title("Horizontal Centerline", fontsize=10)
        self.ax_hcross.set_xlim(0, nx * dx)
        self.ax_hcross.set_ylim(self.vmin, self.vmax)
        style_axis(self.ax_hcross)
        self.line_h, = self.ax_hcross.plot([], [], color=PRIMARY, lw=2.5)
        self.artists.append(self.line_h)

        self.ax_vcross.set_title("Vertical Centerline", fontsize=10)
        self.ax_vcross.set_xlim(0, ny * dy)
        self.ax_vcross.set_ylim(self.vmin, self.vmax)
        style_axis(self.ax_vcross)
        self.line_v, = self.ax_vcross.plot([], [], color=SECONDARY, lw=2.5)
        self.artists.append(self.line_v)

        self.txt = self.fig.text(
            0.5, 0.015, "", color="white", fontsize=9, fontfamily="monospace",
            ha="center", bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none"),
        )
        self.artists.append(self.txt)

    def _update_surface(self, display_y, norm):
        self.surf.remove()
        if self.surf in self.artists:
            self.artists.remove(self.surf)

        self.surf = _plot_field_surface(
            self.ax_surface, self.X, self.Y, display_y, (self.vmin, self.vmax), norm,
        )
        self.artists.append(self.surf)

    def _update_heatmap(self, state, display_y, u, v, norm):
        self.ax_heatmap.clear()
        style_axis(self.ax_heatmap, grid_style=":", grid_width=0.8, grid_alpha=0.5)
        self.ax_heatmap.set_title("2D Contour Field", fontsize=12)
        self.ax_heatmap.set_xlabel("X Axis")
        self.ax_heatmap.set_ylabel("Y Axis")

        ny, nx = display_y.shape
        data_span = float(norm.vmax - norm.vmin)

        if data_span > 1e-4:
            top_levels = _top_contour_levels(display_y)
            fill_levels = np.unique(np.concatenate(([norm.vmin], top_levels, [norm.vmax])))
            contours_f = self.ax_heatmap.contourf(
                self.X, self.Y, display_y,
                levels=fill_levels,
                cmap=SCALAR_FIELD_CMAP,
                norm=norm,
                extend='both',
            )
            line_levels = _line_contour_levels(top_levels)
            if line_levels.size > 0:
                contours = self.ax_heatmap.contour(
                    self.X, self.Y, display_y, levels=line_levels,
                    colors='black', linewidths=0.5, alpha=0.9,
                )
                self.ax_heatmap.clabel(
                    contours, inline=True, fontsize=8, fmt='%.2f', colors='black',
                )

            if self.heatmap_cbar is None:
                self.heatmap_cbar = self.fig.colorbar(
                    contours_f, ax=self.ax_heatmap, fraction=0.04, pad=0.02, shrink=0.9,
                )
                style_colorbar(self.heatmap_cbar, label=self.scalar_label)
                self.fig.subplots_adjust(right=0.94)
            else:
                self.heatmap_cbar.update_normal(contours_f)

        if u is not None and v is not None and getattr(state, '_flow_velocity_fn', None) is None:
            sub = max(1, min(nx, ny) // 20)
            self.ax_heatmap.quiver(
                self.X[::sub, ::sub], self.Y[::sub, ::sub],
                u[::sub, ::sub], v[::sub, ::sub],
                color='black', scale=None, width=0.003, alpha=0.85,
            )

    def update(self, frame_idx, state, current_time, energies, scheme_name=""):
        display_y = state.get_scalar()
        velocity = state.get_velocity()
        u, v = velocity if velocity is not None else (None, None)

        norm = scalar_field_norm(display_y)
        self._update_heatmap(state, display_y, u, v, norm)
        self._update_surface(display_y, norm)

        ny, nx = display_y.shape
        self.line_h.set_data(self.X[ny // 2, :], display_y[ny // 2, :])
        self.line_v.set_data(self.Y[:, nx // 2], display_y[:, nx // 2])

        if frame_idx % 5 == 0:
            self.txt.set_text(format_solver_telemetry(
                self.config.get('eq_name'), scheme_name, current_time, frame_idx,
                self.config['dt'] / self.config['characteristic_spacing'], energies,
            ))
