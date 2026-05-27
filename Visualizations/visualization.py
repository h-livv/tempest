import numpy as np
import matplotlib.pyplot as plt

class TempestVisualizer:
    def __init__(self, initial_state, dx, dt, eq_name):
        self.dx = dx
        self.dt = dt
        self.eq_name = eq_name
        self.ndim = initial_state.ndim
        self.max_frames = 500
        self.nx = initial_state.shape[-1]
        

        # 1. Setup Live Data History Vaults
        self.time_history = []
        self.pe_history = []
        self.ke_history = []
        self.total_history = []
        
        # Space-Time Matrix Allocation
        display_y = initial_state[0] if self.ndim > 1 else initial_state
        self.history_matrix = np.zeros((self.max_frames, self.nx))
        self.history_matrix[0] = display_y

        # 2. Configure Widescreen 3-Panel Layout (1 Row, 3 Columns)
        self.fig, (self.ax_live, self.ax_map, self.ax_energy) = plt.subplots(
            1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1]}
        )
        self.fig.suptitle("Project Tempest: Computational Fluid & Wave Dashboard", fontsize=14, fontweight='bold')

        # --- PANEL 1: LIVE SIMULATION ---
        self.ax_live.set_xlim(0, self.nx)
        self.ax_live.set_ylim(-1.5, 1.5)
        self.ax_live.grid(True, linestyle='--', alpha=0.5)
        self.ax_live.set_xlabel("Spatial Domain (x)")
        self.ax_live.set_ylabel("Amplitude (u)")
        
        # Style the displacement line based on equation properties
        if self.eq_name == 'diffusion':
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Temperature (u)")
            self.ax_live.set_title("Diffusion (2nd Order PDE)", fontsize=12, fontweight='bold')
        elif self.eq_name == 'advection':
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Displacement (u)")
            self.ax_live.set_title("Linear Advection (1st Order PDE)", fontsize=12, fontweight='bold')
        else:
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Displacement (u)")
            self.ax_live.set_title("Wave Engine Simulation", fontsize=12, fontweight='bold')

        self.line_vel, = self.ax_live.plot([], [], color='#ff007f', linestyle='--', lw=1.5, label="Velocity (v)")
        self.ax_live.legend(loc="upper right")

        # Telemetry Box
        self.txt = self.ax_live.text(
            0.02, 0.05, "", transform=self.ax_live.transAxes,
            color="white", fontsize=10, fontfamily="monospace",
            bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none")
        )

        # --- PANEL 2: SPACE-TIME FINGERPRINT ---
        self.im = self.ax_map.imshow(
            self.history_matrix, aspect='auto', cmap='inferno',
            extent=[0, self.nx, self.max_frames * self.dt, 0], vmin=0.0, vmax=1.0
        )
        self.ax_map.set_title("Space-Time Fingerprint Matrix", fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel("Spatial Domain (x)")
        self.ax_map.set_ylabel("Elapsed Time (t)")
        cbar = self.fig.colorbar(self.im, ax=self.ax_map)
        cbar.set_label("Displacement Intensity")

        # --- PANEL 3: LIVE ENERGY TRACKER ---
        self.ax_energy.set_xlabel("Elapsed Time (t)")
        self.ax_energy.grid(True, linestyle='--', alpha=0.5)
        
        # Instantiate line artists
        self.line_pe, = self.ax_energy.plot([], [], color='#ffaa00', lw=2, label="Potential Energy (PE)")
        self.line_ke, = self.ax_energy.plot([], [], color='#ff007f', lw=2, label="Kinetic Energy (KE)")
        self.line_total, = self.ax_energy.plot([], [], color='#00ff00', lw=2.5, label="Total Energy (E)")

        if self.eq_name in ['diffusion', 'advection']:
            self.ax_energy.set_title("Real-Time Magnitude Conservation", fontsize=12, fontweight='bold')
            self.ax_energy.set_ylabel("System Magnitude (L2 Norm)")
            
            self.line_pe.set_visible(False)
            self.line_ke.set_visible(False)
            self.line_total.set_label("Total Magnitude (L2 Norm)")
            
            # FIX: Force the legend to only register the visible total line object
            self.ax_energy.legend(handles=[self.line_total], loc="upper right")
        else:
            self.ax_energy.set_title("Real-Time Energy Conservation", fontsize=12, fontweight='bold')
            self.ax_energy.set_ylabel("Energy Magnitude")
            
            # For the wave equation, let auto-generation pick up all three handles
            self.ax_energy.legend(loc="upper right")
        
        
        plt.tight_layout()
        self.fig.subplots_adjust(wspace=0.3)

    def render_frame(self, frame_idx, state, current_time, scheme_name, energies):
        """Feeds updated computational arrays and energy metrics straight onto screen assets."""
        display_y = state[0] if self.ndim > 1 else state
        pe, ke, total_e = energies
        
        # 1. Update standard profile plots
        self.line_pos.set_data(np.arange(self.nx), display_y)
        if self.ndim > 1:
            self.line_vel.set_data(np.arange(self.nx), state[1])
            
        # 2. Append history matrix data
        if frame_idx < self.max_frames:
            self.history_matrix[frame_idx] = display_y
            self.im.set_data(self.history_matrix)
            
        # 3. Append and update Energy History Lines
        self.time_history.append(current_time)
        self.pe_history.append(pe)
        self.ke_history.append(ke)
        self.total_history.append(total_e)
        
        self.line_pe.set_data(self.time_history, self.pe_history)
        self.line_ke.set_data(self.time_history, self.ke_history)
        self.line_total.set_data(self.time_history, self.total_history)
        
        # Dynamically scale energy axis viewing windows
        if len(self.time_history) > 1:
            self.ax_energy.set_xlim(0, max(self.time_history))
            
            # CONDITIONAL CHANGE: Only calculate limits based on curves that are actually visible
            if self.eq_name in ['diffusion', 'advection']:
                all_energies = self.total_history
            else:
                all_energies = self.pe_history + self.ke_history + self.total_history
                
            min_e, max_e = min(all_energies), max(all_energies)
            margin = max(0.1, 0.2 * (max_e - min_e))
            self.ax_energy.set_ylim(min_e - margin, max_e + margin)

        # 4. Text diagnostics
        cfl = (1.0 * self.dt / self.dx)
        self.txt.set_text(
            f"TIME: {current_time:.3f}s\n"
            f"FRAME: {frame_idx:03d}\n"
            f"SOLVER: {scheme_name.upper()}\n"
            f"CFL RATIO: {cfl:.3f}\n"
            f"TOTAL E: {total_e:.4f}"
        )
        
        # CRITICAL: Return all newly modified artists to the blitting engine
        return self.line_pos, self.line_vel, self.im, self.txt, self.line_pe, self.line_ke, self.line_total