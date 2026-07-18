import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from src.visualization.renderers import RendererRegistry
from src.visualization.theme import (
    apply_dashboard_theme,
    BVE_EQ_NAME,
    ENERGY_KE,
    ENERGY_LOSS,
    ENERGY_PE,
    ENERGY_TOTAL,
    VORTICITY_EQ_NAMES,
    style_axis,
)


class TempestVisualizer:
    def __init__(
        self,
        initial_state,
        dx,
        dt,
        eq_name,
        max_frames,
        steps_per_frame,
        start_delay=0.0,
        final_time=None,
        scalar_label=None,
    ):
        
        # Geometry extraction
        self.dt = dt
        self.max_frames = max_frames
        self.steps_per_frame = steps_per_frame
        self.start_delay = start_delay

        self.final_time = final_time if final_time is not None else max_frames * dt * steps_per_frame

        is_field = hasattr(initial_state, 'grid')
        self.ndim = initial_state.grid.ndim if is_field else initial_state.ndim
        
        # Configuration to pass to renderers
        self.config = {
            'dt': self.dt,
            'max_frames': self.max_frames,
            'final_time': self.final_time,
            'shape': initial_state.grid.shape if is_field else initial_state.shape,
            'spacing': initial_state.grid.spacing if is_field else (dx if isinstance(dx, tuple) else (dx,)),
            'characteristic_spacing': initial_state.grid.characteristic_spacing() if is_field else (min(dx) if isinstance(dx, tuple) else dx),
            'eq_name': eq_name,
            'scalar_label': scalar_label,
        }
        self.vorticity_diagnostics = eq_name in VORTICITY_EQ_NAMES
        self.bve_diagnostics = eq_name == BVE_EQ_NAME

        # Energy Trackers
        self.time_history = []
        self.pe_history = []
        self.ke_history = []
        self.total_history = []
        self.loss_history = []
        self.initial_energy = None
        self.initial_enstrophy = None
        self.initial_peak_vorticity = None

        # Resolve Renderer
        # We dummy-patch rank and ndim onto pure numpy arrays for fallback compatibility if they aren't fields yet
        if not is_field:
            setattr(initial_state, 'rank', 'scalar')
            setattr(initial_state, 'ndim', initial_state.ndim)

        renderer_class = RendererRegistry.resolve(initial_state)
        has_flow = getattr(initial_state, 'has_flow', False)

        apply_dashboard_theme()

        # Unified Figure & GridSpec
        self.fig = plt.figure(num="Tempest Unified Dashboard", figsize=(14, 8), dpi=100)
        
        if self.ndim == 1:
            gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[2, 1])
            self.slots = {
                'primary': gs[0, 0],
                'secondary1': gs[0, 1],
                'energy': gs[1, :]
            }
        elif has_flow:
            gs = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[2, 1])
            gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.25)
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.3)
            self.slots = {
                'surface': gs_top[0, 0],
                'heatmap': gs_top[0, 1],
                'secondary2': gs_bottom[0, 0],
                'secondary3': gs_bottom[0, 1],
                'energy': gs_bottom[0, 2],
            }
        else:
            gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[2, 1], width_ratios=[1, 1, 2])
            self.slots = {
                'primary': gs[0, 2],
                'secondary1': gs[0, :2],
                'secondary2': gs[1, 0],
                'secondary3': gs[1, 1],
                'energy': gs[1, 2],
            }

        self.renderer = renderer_class(self.fig, self.slots, self.config, initial_state)
        
        # Energy panel
        self.ax_energy = self.fig.add_subplot(self.slots['energy'])
        self.ax_energy.set_xlabel("Elapsed Time (t)")
        style_axis(self.ax_energy, grid_style="--", grid_alpha=0.5)
        self.ax_energy.set_xlim(0, self.final_time)
        self.energy_axis_initialized = False

        if self.bve_diagnostics:
            self.ax_energy.set_ylabel("Normalized Metrics")
            self.line_energy, = self.ax_energy.plot([], [], color=ENERGY_TOTAL, lw=2.5, label="Energy")
            self.line_enstrophy, = self.ax_energy.plot([], [], color=ENERGY_KE, lw=2, label="Enstrophy")
            self.line_peak, = self.ax_energy.plot([], [], color=ENERGY_PE, lw=2, label="Peak |ζ|")
            self.ax_energy.legend(loc="upper right")
            self.line_pe = self.line_ke = self.line_total = self.line_loss = None
        elif self.vorticity_diagnostics:
            self.ax_energy.set_ylabel("Energy / Enstrophy")
            self.line_energy, = self.ax_energy.plot([], [], color=ENERGY_TOTAL, lw=2.5, label="Energy")
            self.line_enstrophy, = self.ax_energy.plot([], [], color=ENERGY_KE, lw=2, label="Enstrophy")
            self.ax_energy.legend(loc="upper right")
            self.line_peak = None
            self.line_pe = self.line_ke = self.line_total = self.line_loss = None
        else:
            self.ax_energy.set_ylabel("Energy")
            self.line_pe, = self.ax_energy.plot([], [], color=ENERGY_PE, lw=2, label="PE")
            self.line_ke, = self.ax_energy.plot([], [], color=ENERGY_KE, lw=2, label="KE")
            self.line_total, = self.ax_energy.plot([], [], color=ENERGY_TOTAL, lw=2.5, label="Total E")
            self.line_loss, = self.ax_energy.plot([], [], color=ENERGY_LOSS, linestyle="--", lw=2, label="Loss")
            self.ax_energy.legend(loc="upper right")
            self.line_energy = self.line_enstrophy = self.line_peak = None

        self.fig.tight_layout(rect=[0, 0.02, 0.95, 1])

    def render_frame(self, frame_idx, state, current_time, scheme_name, energies=None):
        energy, enstrophy, total_e = energies[:3]
        peak_vorticity = energies[3] if len(energies) > 3 else None

        if frame_idx == 0:
            self.time_history.clear()
            self.pe_history.clear()
            self.ke_history.clear()
            self.total_history.clear()
            self.loss_history.clear()
            if self.bve_diagnostics:
                self.initial_energy = energy if energy != 0 else 1.0
                self.initial_enstrophy = enstrophy if enstrophy != 0 else 1.0
                self.initial_peak_vorticity = peak_vorticity if peak_vorticity != 0 else 1.0
            elif not self.vorticity_diagnostics:
                self.initial_energy = total_e

        self.renderer.update(frame_idx, state, current_time, energies, scheme_name)

        if self.bve_diagnostics:
            if not self.energy_axis_initialized:
                self.ax_energy.set_ylim(0.0, 1.1)
                self.energy_axis_initialized = True

            norm_energy = energy / self.initial_energy
            norm_enstrophy = enstrophy / self.initial_enstrophy
            norm_peak = peak_vorticity / self.initial_peak_vorticity

            self.time_history.append(current_time)
            self.pe_history.append(norm_energy)
            self.ke_history.append(norm_enstrophy)
            self.total_history.append(norm_peak)

            self.line_energy.set_data(self.time_history, self.pe_history)
            self.line_enstrophy.set_data(self.time_history, self.ke_history)
            self.line_peak.set_data(self.time_history, self.total_history)

            if frame_idx % 15 == 0 and len(self.time_history) > 10:
                all_values = self.pe_history + self.ke_history + self.total_history
                min_e, max_e = min(all_values), max(all_values)
                if not (np.isnan(min_e) or np.isnan(max_e) or np.isinf(min_e) or np.isinf(max_e)):
                    margin = max(1e-6, 0.15 * max(abs(max_e - 1.0), abs(min_e - 1.0), 0.1))
                    self.ax_energy.set_ylim(min_e - margin, max_e + margin)

            energy_artists = (self.line_energy, self.line_enstrophy, self.line_peak)
        elif self.vorticity_diagnostics:
            if not self.energy_axis_initialized:
                estimated_max = max(abs(energy), abs(enstrophy), 1.0)
                self.ax_energy.set_ylim(-0.1 * estimated_max, 1.1 * estimated_max)
                self.energy_axis_initialized = True

            self.time_history.append(current_time)
            self.pe_history.append(energy)
            self.ke_history.append(enstrophy)

            self.line_energy.set_data(self.time_history, self.pe_history)
            self.line_enstrophy.set_data(self.time_history, self.ke_history)

            if frame_idx % 15 == 0 and len(self.time_history) > 10:
                all_values = self.pe_history + self.ke_history
                min_e, max_e = min(all_values), max(all_values)
                if not (np.isnan(min_e) or np.isnan(max_e) or np.isinf(min_e) or np.isinf(max_e)):
                    margin = max(1e-6, 0.15 * max(abs(max_e), abs(min_e)))
                    self.ax_energy.set_ylim(min_e - margin, max_e + margin)

            energy_artists = (self.line_energy, self.line_enstrophy)
        else:
            energy_loss = self.initial_energy - total_e

            if not self.energy_axis_initialized:
                estimated_max = max(abs(energy), abs(enstrophy), abs(total_e), 1.0)
                self.ax_energy.set_ylim(-0.1 * estimated_max, 1.1 * estimated_max)
                self.energy_axis_initialized = True

            self.time_history.append(current_time)
            self.pe_history.append(energy)
            self.ke_history.append(enstrophy)
            self.total_history.append(total_e)
            self.loss_history.append(energy_loss)

            self.line_pe.set_data(self.time_history, self.pe_history)
            self.line_ke.set_data(self.time_history, self.ke_history)
            self.line_total.set_data(self.time_history, self.total_history)
            self.line_loss.set_data(self.time_history, self.loss_history)

            if frame_idx % 15 == 0 and len(self.time_history) > 10:
                all_energies = self.pe_history + self.ke_history + self.total_history + self.loss_history
                min_e, max_e = min(all_energies), max(all_energies)
                if not (np.isnan(min_e) or np.isnan(max_e) or np.isinf(min_e) or np.isinf(max_e)):
                    margin = max(1e-6, 0.15 * max(abs(max_e), abs(min_e)))
                    self.ax_energy.set_ylim(min_e - margin, max_e + margin)

            energy_artists = (self.line_pe, self.line_ke, self.line_total, self.line_loss)

        if frame_idx == 0 and self.start_delay > 0:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            import time
            time.sleep(self.start_delay)

        return tuple(self.renderer.get_artists()) + energy_artists

    def close(self):
        try:
            plt.close(self.fig)
        except AttributeError:
            pass