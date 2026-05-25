import numpy as np
import matplotlib.pyplot as plt

class TempestVisualizer:
    def __init__(self, initial_state, dx, dt, equation_name):
        self.dx = dx
        self.dt = dt
        self.eq_name = equation_name.upper()
        
        # Unpack array metadata dimensions
        self.ndim = initial_state.ndim
        self.n = initial_state.shape[-1]
        self.x = np.linspace(0, self.n * dx, self.n)
        
        # Create a modern split-screen panel layout
        self.fig, (self.ax_live, self.ax_hist) = plt.subplots(1, 2, figsize=(15, 6))
        
        # -------------------------------------------------------------
        # PANEL 1: LIVE MOTION GRAPH
        # -------------------------------------------------------------
        init_y = initial_state[0] if self.ndim > 1 else initial_state
        
        # Primary position displacement line
        self.line_pos, = self.ax_live.plot(self.x, init_y, lw=2.5, color='#00f0ff', label='Displacement (u)')
        
        # Secondary kinetic velocity line (Only visible for 2D system matrices)
        self.line_vel, = self.ax_live.plot(self.x, np.zeros(self.n), lw=1.5, color='#ff007f', linestyle='--', label='Velocity (v)')
        self.line_vel.set_visible(True if self.ndim > 1 else False)
        
        # Format live canvas layout
        max_amp = np.max(np.abs(init_y)) if np.max(np.abs(init_y)) > 0 else 1.0
        self.ax_live.set_xlim(0, self.n * dx)
        self.ax_live.set_ylim(-max_amp * 1.6, max_amp * 1.6)
        self.ax_live.grid(True, linestyle='--', alpha=0.4)
        self.ax_live.set_xlabel('Spatial Domain (x)', fontsize=11)
        self.ax_live.set_ylabel('Amplitude (u)', fontsize=11)
        self.ax_live.legend(loc='upper right')
        
        # -------------------------------------------------------------
        # PANEL 2: SPACE-TIME FINGERPRINT (x-t Plot)
        # -------------------------------------------------------------
        self.max_frames = 1000
        self.history_matrix = np.zeros((self.max_frames, self.n))
        self.history_matrix[0] = init_y
        
        # Extent coordinates mapping: [Left, Right, Bottom, Top]
        self.im = self.ax_hist.imshow(
            self.history_matrix, 
            cmap='inferno', 
            aspect='auto', 
            extent=[0, self.n * dx, self.max_frames * dt, 0]
        )
        self.ax_hist.set_xlabel('Spatial Domain (x)', fontsize=11)
        self.ax_hist.set_ylabel('Elapsed Time (t)', fontsize=11)
        self.ax_hist.set_title('Space-Time Fingerprint Matrix', fontsize=12, fontweight='bold')
        self.fig.colorbar(self.im, ax=self.ax_hist, label='Displacement Intensity')
        
        # -------------------------------------------------------------
        # LIVE TELEMETRY AND THEMES
        # -------------------------------------------------------------
        self._apply_dynamic_theme()
        self.txt = self.ax_live.text(
            0.02, 0.05, '', 
            transform=self.ax_live.transAxes, 
            family='monospace', fontsize=10, color='white',
            bbox=dict(facecolor='#1e1e1e', alpha=0.8, edgecolor='none')
        )

    def _apply_dynamic_theme(self):
        """Changes graphics identity based on the physical model running."""
        if "WAVE" in self.eq_name:
            self.line_pos.set_color('#00f0ff')  # Neon Cyan
            self.ax_live.set_title('Acoustic Wave Propagation (2nd Order PDE)', fontsize=12, fontweight='bold')
        elif "ADVECTION" in self.eq_name:
            self.line_pos.set_color('#00ff66')  # Matrix Green
            self.ax_live.set_title('Linear Advection Transport (1st Order PDE)', fontsize=12, fontweight='bold')
        else:
            self.line_pos.set_color('#ffffff')
            self.ax_live.set_title(f'{self.eq_name} Engine Simulation', fontsize=12, fontweight='bold')
            
        self.fig.suptitle('Project Tempest: Computational Fluid & Wave Dashboard', fontsize=14, fontweight='bold')
        self.fig.tight_layout()

    def render_frame(self, frame_idx, state, current_time, scheme_name):
        """Feeds updated scientific computational arrays straight onto screen assets."""
        display_y = state[0] if self.ndim > 1 else state
        
        # 1. Refresh live graph curves
        self.line_pos.set_ydata(display_y)
        if self.ndim > 1:
            self.line_vel.set_ydata(state[1])
            
        # 2. Append profile row data into the space-time history grid
        if frame_idx < self.max_frames:
            self.history_matrix[frame_idx] = display_y
            self.im.set_data(self.history_matrix)
            
        # 3. Process live math tracking diagnostics
        cfl = (1.0 * self.dt / self.dx)
        self.txt.set_text(
            f"TIME: {current_time:.3f}s\n"
            f"FRAME: {frame_idx:03d}\n"
            f"SOLVER: {scheme_name.upper()}\n"
            f"CFL RATIO: {cfl:.3f}"
        )
        
        return self.line_pos, self.line_vel, self.im,