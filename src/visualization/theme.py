"""Dark theme palette and styling helpers for the Tempest Unified Dashboard."""

import matplotlib.pyplot as plt
import numpy as np
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

# 3D surface: |value| mapped from zero (cyan) to |max| (magenta)
SURFACE_MAG_CMAP = LinearSegmentedColormap.from_list(
    "tempest_surface_mag",
    [PRIMARY, "#ff00ff"],
)

ENERGY_PE = "#ffb347"
ENERGY_KE = "#ff6b9d"
ENERGY_TOTAL = "#00ff88"
ENERGY_LOSS = "#ff5555"

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


def surface_magnitude_facecolors(values):
    """Map |z| from 0 (cyan) to max|z| (magenta) for 3D surface face colors."""
    abs_vals = np.abs(values)
    vmax = float(np.max(abs_vals))
    if vmax < 1e-12:
        vmax = 1.0
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    return SURFACE_MAG_CMAP(norm(abs_vals))
