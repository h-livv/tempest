"""Dark theme palette and styling helpers for the Tempest Unified Dashboard."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

# Core palette — black background, cyan data, muted grey chrome
BACKGROUND = "#000000"
AXES_FACE = "#000000"
TEXT = "#a9a9a9"
TITLE = "#cccccc"
GRID = "#808080"
PRIMARY = "#00CED1"
SECONDARY = "#ff6b9d"
SURFACE_FACE = "#001515"
SURFACE_EDGE = "#00CED1"
SURFACE_MESH_WIDTH = 0.2
TELEMETRY_BG = "#1a1a1a"
TELEMETRY_TEXT = "#cccccc"

# Shared scalar-field colormap for coupled 2D contour + 3D surface views
SCALAR_FIELD_CMAP = plt.get_cmap("cool")

ENERGY_PE = "#ffb347"
ENERGY_KE = "#ff6b9d"
ENERGY_TOTAL = "#00ff88"
ENERGY_LOSS = "#ff5555"

VORTICITY_EQ_NAMES = frozenset({'barotropic_voricity', 'rossby_wave'})
BVE_EQ_NAME = 'barotropic_voricity'

DASHBOARD_RCPARAMS = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": AXES_FACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TITLE,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
}


def apply_dashboard_theme():
    plt.rcParams.update(DASHBOARD_RCPARAMS)


def scalar_axis_label(eq_name, override=None):
    """Return the z-axis / colorbar label for a given equation."""
    if override:
        return override
    labels = {
        'shallow_water': 'Height (h)',
        'barotropic_voricity': 'Vorticity (ζ)',
        'barotropic_vorticity': 'Vorticity (ζ)',
        'rossby_wave': 'Potential vorticity (q)',
        'wave': 'Displacement',
        'advection': 'Scalar',
        'diffusion': 'Scalar',
        'burgers': 'Velocity (u)',
    }
    return labels.get(eq_name, 'State')


def format_solver_telemetry(eq_name, scheme_name, current_time, frame_idx, cfl, energies):
    """Footer telemetry string for the unified dashboard."""
    if eq_name == BVE_EQ_NAME:
        energy, enstrophy, _, peak_vorticity = energies
        return (
            f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
            f"ENERGY: {energy:.3e} | ENSTROPHY: {enstrophy:.3e} | PEAK |ζ|: {peak_vorticity:.3e}"
        )
    if eq_name in VORTICITY_EQ_NAMES:
        energy, enstrophy, _ = energies
        return (
            f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
            f"ENERGY: {energy:.3f}J | ENSTROPHY: {enstrophy:.3f}"
        )
    return (
        f"SOLVER: {scheme_name.upper()} | TIME: {current_time:.3f}s | FRAME: {frame_idx:03d} | CFL: {cfl:.3f} | "
        f"ENERGY: {energies[2]:.3f}J"
    )


# Target number of mesh cells on the 3D surface plot (independent of simulation N)
SURFACE_MESH_SIZE = 50


def surface_mesh_stride(n, target=SURFACE_MESH_SIZE):
    """Stride for plot_surface so the drawn mesh has ~target cells per axis."""
    return max(1, int(np.ceil(n / target)))


def style_axis(ax, grid=True, grid_alpha=0.5, grid_style="-", grid_width=1.0):
    ax.set_facecolor(AXES_FACE)
    if grid:
        ax.grid(
            True,
            which="both",
            color=GRID,
            linestyle=grid_style,
            linewidth=grid_width,
            alpha=grid_alpha,
        )
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TITLE)


def style_3d_axis(ax, grid=True, grid_alpha=0.5):
    ax.set_facecolor(AXES_FACE)
    if grid:
        ax.grid(True, which="both", color=GRID, linestyle="-", linewidth=1.0, alpha=grid_alpha)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.zaxis.label.set_color(TEXT)
    ax.title.set_color(TITLE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(0.3)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["alpha"] = grid_alpha


def style_colorbar(cbar, label=None):
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    cbar.outline.set_edgecolor(GRID)
    if label:
        cbar.set_label(label, color=TEXT)
    elif cbar.ax.yaxis.label:
        cbar.ax.yaxis.label.set_color(TEXT)


def scalar_field_norm(values):
    """Single normalization for a scalar field from its min/max."""
    values = np.asarray(values)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def scalar_field_facecolors(values, norm, cmap=None):
    """Map scalar field values to RGBA face colors using a shared norm."""
    if cmap is None:
        cmap = SCALAR_FIELD_CMAP
    return cmap(norm(np.asarray(values)))


# Legacy magnitude colormap for non-flow 2D scalar surfaces
SURFACE_MAG_CMAP = LinearSegmentedColormap.from_list(
    "tempest_surface_mag",
    [PRIMARY, "#ff00ff"],
)


def surface_magnitude_facecolors(values):
    """Map |z| from 0 (cyan) to max|z| (magenta) for 3D surface face colors."""
    abs_vals = np.abs(values)
    vmax = float(np.max(abs_vals))
    if vmax < 1e-12:
        vmax = 1.0
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    return SURFACE_MAG_CMAP(norm(abs_vals))
