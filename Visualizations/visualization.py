import numpy as np
import matplotlib.pyplot as plt


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
    ):
        self.dx = dx
        self.dt = dt
        self.eq_name = eq_name
        self.max_frames = max_frames
        self.steps_per_frame = steps_per_frame
        self.start_delay = start_delay
        self.final_time = final_time if final_time is not None else max_frames * dt * steps_per_frame
        self.ndim = initial_state.ndim
        self.nx = initial_state.shape[-1]
        self.x = np.arange(self.nx) * self.dx
        
        # 1. Setup Live Data History Vaults
        self.time_history = []
        self.pe_history = []
        self.ke_history = []
        self.total_history = []
        self.loss_history = []         
        self.initial_energy = None     
        
        # Space-Time Matrix Allocation
        display_y = initial_state[0] if self.ndim > 1 else initial_state
        self.history_matrix = np.zeros((self.max_frames, self.nx))
        self.history_matrix[0] = display_y

        # 2. Configure Widescreen 3-Panel Layout
        self.fig, (self.ax_live, self.ax_map, self.ax_energy) = plt.subplots(
            1, 3, figsize=(14, 6.5), dpi=100, gridspec_kw={'width_ratios': [1, 1, 1]},
            layout='constrained'
        )
        self.fig.suptitle("Project Tempest: Computational Fluid & Wave Dashboard", fontsize=14, fontweight='bold')

        # --- PANEL 1: LIVE SIMULATION ---
        self.ax_live.set_xlim(0, self.nx * self.dx)
        self.ax_live.grid(True, linestyle='--', alpha=0.5)
        self.ax_live.set_xlabel("Spatial Domain (x)")
        
        if self.eq_name == 'diffusion':
            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Temperature (u)")
            self.ax_live.set_title("Diffusion (2nd Order PDE)", fontsize=12, fontweight='bold')
        elif self.eq_name == 'advection':
            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Displacement (u)")
            self.ax_live.set_title("Linear Advection (1st Order PDE)", fontsize=12, fontweight='bold')
        elif self.eq_name == 'wave':
            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Displacement (u)")
            self.ax_live.set_title("Wave Propagation (2nd Order PDE)", fontsize=12, fontweight='bold')
        elif self.eq_name == 'shallow_water':
            self.ax_live.set_ylim(-0.2, 5.0)  # Expanded scale for sharp dam break peaks
            self.ax_live.set_ylabel("Fluid Profile Scale")
            self.line_pos, = self.ax_live.plot([], [], color='#00ffff', lw=2.5, label="Water Depth (h)")
            self.ax_live.set_title("Shallow Water Equations (Non-linear System)", fontsize=12, fontweight='bold')

        velocity_label = "Velocity (u)" if self.eq_name == 'shallow_water' else "Velocity (v)"
        self.line_vel, = self.ax_live.plot([], [], color='#ff007f', linestyle='--', lw=1.5, label=velocity_label)
        self.ax_live.legend(loc="upper right")

        # Telemetry Box Layout
        self.txt = self.ax_live.text(
            0.02, 0.96, "", transform=self.ax_live.transAxes, verticalalignment='top',    # Anchors the text from the top edge of the bounding box
            horizontalalignment='left', color="white", fontsize=9, fontfamily="monospace",
            bbox=dict(facecolor="#333333", alpha=0.8, edgecolor="none")
        )

        # --- PANEL 2: SPACE-TIME FINGERPRINT ---
        vmax_val = 3.5 if self.eq_name == 'shallow_water' else 1.0
        self.im = self.ax_map.imshow(
            self.history_matrix, aspect='auto', cmap='inferno',
            extent=[0, self.nx * self.dx, self.final_time, 0],
            vmin=0.0, vmax=vmax_val
        )
        self.ax_map.set_title("Space-Time Fingerprint Matrix", fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel("Spatial Domain (x)")
        self.ax_map.set_ylabel("Elapsed Time (t)")
        cbar = self.fig.colorbar(self.im, ax=self.ax_map)
        
        cbar_label = "Water Depth (h)" if self.eq_name == 'shallow_water' else "Displacement Intensity"
        cbar.set_label(cbar_label)

        # --- PANEL 3: LIVE ENERGY TRACKER ---
        self.ax_energy.set_xlabel("Elapsed Time (t)")
        self.ax_energy.set_ylabel("Energy State Value")
        self.ax_energy.grid(True, linestyle='--', alpha=0.5)
        
        self.line_pe, = self.ax_energy.plot([], [], color='#ffaa00', lw=2, label="Potential Energy (PE)")
        self.line_ke, = self.ax_energy.plot([], [], color='#ff007f', lw=2, label="Kinetic Energy (KE)")
        self.line_total, = self.ax_energy.plot([], [], color='#00ff00', lw=2.5, label="Total Energy (E)")
        self.line_loss, = self.ax_energy.plot([], [], color='#ff3333', linestyle='--', lw=2, label="Energy Loss (ΔE)")
        
        # FORCE ABSOLUTE VALUES (Turns off the confusing negative offsets)
        self.ax_energy.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax_energy.get_yaxis().get_major_formatter().set_useOffset(False)
        self.ax_energy.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_energy.get_yaxis().get_major_formatter().set_scientific(False)
        
        if self.eq_name in ['wave', 'shallow_water']:
            self.ax_energy.set_title("System Energy Separation", fontsize=12, fontweight='bold')
        else:
            self.ax_energy.set_title("Total Energy Track (Scalar Field)", fontsize=12, fontweight='bold')
            
        self.ax_energy.legend(loc="upper right")
    

    def render_frame(self, frame_idx, state, current_time, scheme_name, energies=None):
        pe, ke, total_e = energies
        
        # Flush caches completely if the animation loops back to zero
        if frame_idx == 0:
            self.time_history.clear()
            self.pe_history.clear()
            self.ke_history.clear()
            self.total_history.clear()
            self.loss_history.clear()        
            self.initial_energy = total_e    
            self.history_matrix.fill(0) 
            
        display_y = state[0] if self.ndim > 1 else state
        
        # 1. Update primary field plot (Height / Displacement)
        self.line_pos.set_data(self.x, display_y)
        
        # 2. Update velocity field line if running a multi-variable system
        if state.shape[0] > 1:
            # FIXED: Removed the redundant q/h division since the incoming state is already primitive [h, v]
            self.line_vel.set_data(self.x, state[1])
            
        # 3. Append history matrix row
        if frame_idx < self.max_frames:
            self.history_matrix[frame_idx] = display_y
            self.im.set_data(self.history_matrix)
            
        # 4. Process Energy Diagnostics
        energy_loss = self.initial_energy - total_e   
        
        self.time_history.append(current_time)
        self.pe_history.append(pe)
        self.ke_history.append(ke)
        self.total_history.append(total_e)
        self.loss_history.append(energy_loss)         
        
        self.line_pe.set_data(self.time_history, self.pe_history)
        self.line_ke.set_data(self.time_history, self.ke_history)
        self.line_total.set_data(self.time_history, self.total_history)
        self.line_loss.set_data(self.time_history, self.loss_history) 
        
        # Dynamic axis autoscaling for energy lines
        if len(self.time_history) > 1:
            # Explicitly force the left boundary to stay locked to absolute 0
            self.ax_energy.set_xlim(0, max(max(self.time_history), 0.1))
            
            if self.eq_name in ['diffusion', 'advection']:
                all_energies = self.total_history + self.loss_history
            else:
                all_energies = self.pe_history + self.ke_history + self.total_history + self.loss_history
            
            min_e, max_e = min(all_energies), max(all_energies)
            margin = max(0.1, 0.2 * (max_e - min_e))
            self.ax_energy.set_ylim(min_e - margin, max_e + margin)
        else:
            # Safe baseline limit for frame 0
            self.ax_energy.set_xlim(0, 0.1)

        # 5. Push data string to screen telemetry
        cfl = (1.0 * self.dt / self.dx)
        self.txt.set_text(
            f"TIME: {current_time:.3f}s\n"
            f"FRAME: {frame_idx:03d}\n"
            f"SOLVER: {scheme_name.upper()}\n"
            f"CFL RATIO: {cfl:.3f}\n"
            f"NET ENERGY: {total_e:.2f}J\n"
            f"ENERGY LOSS: {energy_loss:.2f}J"
        )
        
        # ... your existing step 5 code: self.txt.set_text(...) ...
        
        # 6. Recording Guard: Catch frame 0, paint canvas, and hold
        if frame_idx == 0 and self.start_delay > 0:
            self.fig.canvas.draw_idle() 
            plt.pause(0.001) # Forces Matplotlib to completely render the window UI
            
            import time
            print(f"Dashboard primed. Holding Frame 0 static for {self.start_delay}s for screen capture...")
            time.sleep(self.start_delay)
            print("Simulation timeline released!")

        return (self.line_pos, self.line_vel, self.im, 
                self.line_pe, self.line_ke, self.line_total, self.line_loss, self.txt)