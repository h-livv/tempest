import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from src.visualization.renderers import RendererRegistry

class TempestVisualizer:
    def __init__(
        self,
        initial_state,
        dx,
        dt,
        eq_name, # Ignored for logic, kept for backward compatibility if signature is frozen
        max_frames,
        steps_per_frame,
        start_delay=0.0,
        final_time=None,
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
            'characteristic_spacing': initial_state.grid.characteristic_spacing() if is_field else (min(dx) if isinstance(dx, tuple) else dx)
        }

        # Energy Trackers
        self.time_history = []
        self.pe_history = []
        self.ke_history = []
        self.total_history = []
        self.loss_history = []
        self.initial_energy = None

        # Resolve Renderer
        # We dummy-patch rank and ndim onto pure numpy arrays for fallback compatibility if they aren't fields yet
        if not is_field:
            setattr(initial_state, 'rank', 'scalar')
            setattr(initial_state, 'ndim', initial_state.ndim)

        renderer_class = RendererRegistry.resolve(initial_state)

        # Unified Figure & GridSpec
        self.fig = plt.figure(num="Tempest Unified Dashboard", figsize=(14, 8), dpi=100)
        
        if self.ndim == 1:
            gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[2, 1])
            self.slots = {
                'primary': gs[0, 0],
                'secondary1': gs[0, 1],
                'energy': gs[1, :]
            }
        else:
            gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[2, 1], width_ratios=[1, 1, 2])
            self.slots = {
                'primary': gs[0, 2],        # Heatmap as a square on the right (slightly extended)
                'secondary1': gs[0, :2],    # 3D Surface as the main attraction on the left
                'secondary2': gs[1, 0],     # Horiz Cross
                'secondary3': gs[1, 1],     # Vert Cross
                'energy': gs[1, 2]          # Energy Stats
            }

        self.renderer = renderer_class(self.fig, self.slots, self.config)
        
        # Energy Axis Setup
        self.ax_energy = self.fig.add_subplot(self.slots['energy'])
        self.ax_energy.set_xlabel("Elapsed Time (t)")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.grid(True, linestyle="--", alpha=0.5)

        self.line_pe, = self.ax_energy.plot([], [], color="#ffaa00", lw=2, label="PE")
        self.line_ke, = self.ax_energy.plot([], [], color="#ff007f", lw=2, label="KE")
        self.line_total, = self.ax_energy.plot([], [], color="#00ff00", lw=2.5, label="Total E")
        self.line_loss, = self.ax_energy.plot([], [], color="#ff3333", linestyle="--", lw=2, label="Loss")
        self.ax_energy.legend(loc="upper right")
        self.ax_energy.set_xlim(0, self.final_time)
        self.energy_axis_initialized = False

        self.fig.tight_layout()

    def render_frame(self, frame_idx, state, current_time, scheme_name, energies=None):
        pe, ke, total_e = energies

        if frame_idx == 0:
            self.time_history.clear()
            self.pe_history.clear()
            self.ke_history.clear()
            self.total_history.clear()
            self.loss_history.clear()
            self.initial_energy = total_e

        # Delegate Rendering
        self.renderer.update(frame_idx, state, current_time, energies, scheme_name)

        # Update Energy
        energy_loss = self.initial_energy - total_e
        
        if not self.energy_axis_initialized:
            estimated_max = max(abs(pe), abs(ke), abs(total_e), 1.0)
            self.ax_energy.set_ylim(-0.1 * estimated_max, 1.1 * estimated_max)
            self.energy_axis_initialized = True

        self.time_history.append(current_time)
        self.pe_history.append(pe)
        self.ke_history.append(ke)
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

        if frame_idx == 0 and self.start_delay > 0:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            import time
            time.sleep(self.start_delay)

        return tuple(self.renderer.get_artists()) + (self.line_pe, self.line_ke, self.line_total, self.line_loss)

    def close(self):
        try:
            plt.close(self.fig)
        except AttributeError:
            pass