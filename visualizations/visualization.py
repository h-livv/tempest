import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np



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

        # ============================================================
        # CORE SIMULATION METADATA
        # ============================================================

        self.dx = dx
        self.dt = dt
        self.eq_name = eq_name
        self.max_frames = max_frames
        self.steps_per_frame = steps_per_frame
        self.start_delay = start_delay

        self.final_time = (
            final_time
            if final_time is not None
            else max_frames * dt * steps_per_frame
        )

        self.ndim = initial_state.ndim
        self.nx = initial_state.shape[-1]
        self.x = np.arange(self.nx) * self.dx

        # ============================================================
        # HISTORY BUFFERS
        # ============================================================

        self.time_history = []
        self.pe_history = []
        self.ke_history = []
        self.total_history = []
        self.loss_history = []

        self.initial_energy = None

        # ============================================================
        # SPACETIME MATRIX
        # ============================================================

        display_y = initial_state[0] if self.ndim > 1 else initial_state

        self.history_matrix = np.zeros((self.max_frames, self.nx))
        self.history_matrix[0] = display_y

        # ============================================================
        # FIGURE LAYOUT
        # ============================================================

        self.fig, (self.ax_live, self.ax_map, self.ax_energy) = plt.subplots(
            1,
            3,
            figsize=(14, 6.5),
            dpi=100,
            gridspec_kw={"width_ratios": [1, 1, 1]},
        )

        self.fig.subplots_adjust(
        left=0.05,
        right=0.97,
        bottom=0.10,
        top=0.86,
        wspace=0.28
    )

        self.fig.suptitle(
            "Tempest: Computational Fluid & Wave Dashboard",
            fontsize=14,
            fontweight="bold",
            y=0.97
        )

        # ============================================================
        # PANEL 1 — LIVE FIELD
        # ============================================================

        self.ax_live.set_xlim(0, self.nx * self.dx)
        self.ax_live.grid(True, linestyle="--", alpha=0.5)

        self.ax_live.set_xlabel("Spatial Domain (x)")

        if self.eq_name == "diffusion":

            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")

            self.line_pos, = self.ax_live.plot(
                [],
                [],
                color="#00ffff",
                lw=2.5,
                label="Temperature (u)"
            )

            self.ax_live.set_title(
                "Diffusion (2nd Order PDE)",
                fontsize=12,
                fontweight="bold"
            )

        elif self.eq_name == "advection":

            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")

            self.line_pos, = self.ax_live.plot(
                [],
                [],
                color="#00ffff",
                lw=2.5,
                label="Displacement (u)"
            )

            self.ax_live.set_title(
                "Linear Advection (1st Order PDE)",
                fontsize=12,
                fontweight="bold"
            )

        elif self.eq_name == "wave":

            self.ax_live.set_ylim(-1.5, 1.5)
            self.ax_live.set_ylabel("Amplitude (u)")

            self.line_pos, = self.ax_live.plot(
                [],
                [],
                color="#00ffff",
                lw=2.5,
                label="Displacement (u)"
            )

            self.ax_live.set_title(
                "Wave Propagation (2nd Order PDE)",
                fontsize=12,
                fontweight="bold"
            )

        elif self.eq_name == "shallow_water":

            self.ax_live.set_ylim(-0.2, 5.0)
            self.ax_live.set_ylabel("Fluid Profile Scale")

            self.line_pos, = self.ax_live.plot(
                [],
                [],
                color="#00ffff",
                lw=2.5,
                label="Water Depth (h)"
            )

            self.ax_live.set_title(
                "Shallow Water Equations (Non-linear System)",
                fontsize=12,
                fontweight="bold"
            )

        velocity_label = (
            "Velocity (u)"
            if self.eq_name == "shallow_water"
            else "Velocity (v)"
        )

        self.line_vel, = self.ax_live.plot(
            [],
            [],
            color="#ff007f",
            linestyle="--",
            lw=1.5,
            label=velocity_label
        )

        self.ax_live.legend(loc="upper right")

        # ============================================================
        # TELEMETRY BOX
        # ============================================================

        self.txt = self.ax_live.text(
            0.02,
            0.96,
            "",
            transform=self.ax_live.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            color="white",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(
                facecolor="#333333",
                alpha=0.8,
                edgecolor="none"
            )
        )

        # ============================================================
        # PANEL 2 — SPACE-TIME MAP
        # ============================================================

        vmax_val = 3.5 if self.eq_name == "shallow_water" else 1.0

        self.im = self.ax_map.imshow(
            self.history_matrix,
            aspect="auto",
            cmap="inferno",
            extent=[0, self.nx * self.dx, self.final_time, 0],
            vmin=0.0,
            vmax=vmax_val,
            interpolation="nearest"
        )

        self.ax_map.set_title(
            "Space-Time Fingerprint Matrix",
            fontsize=12,
            fontweight="bold"
        )

        self.ax_map.set_xlabel("Spatial Domain (x)")
        self.ax_map.set_ylabel("Elapsed Time (t)")

        cbar = self.fig.colorbar(self.im, ax=self.ax_map)

        cbar_label = (
            "Water Depth (h)"
            if self.eq_name == "shallow_water"
            else "Displacement Intensity"
        )

        cbar.set_label(cbar_label)

        # ============================================================
        # PANEL 3 — ENERGY TRACKER
        # ============================================================

        self.ax_energy.set_xlabel("Elapsed Time (t)")
        self.ax_energy.set_ylabel("Energy")

        self.ax_energy.grid(True, linestyle="--", alpha=0.5)

        self.line_pe, = self.ax_energy.plot(
            [],
            [],
            color="#ffaa00",
            lw=2,
            label="Potential Energy (PE)"
        )

        self.line_ke, = self.ax_energy.plot(
            [],
            [],
            color="#ff007f",
            lw=2,
            label="Kinetic Energy (KE)"
        )

        self.line_total, = self.ax_energy.plot(
            [],
            [],
            color="#00ff00",
            lw=2.5,
            label="Total Energy (E)"
        )

        self.line_loss, = self.ax_energy.plot(
            [],
            [],
            color="#ff3333",
            linestyle="--",
            lw=2,
            label="Energy Loss (ΔE)"
        )

        self.ax_energy.legend(loc="upper right")

        # FIXED AXIS LIMITS FOR BLITTING STABILITY

        self.ax_energy.set_xlim(0, self.final_time)

        self.energy_axis_initialized = False

        # Disable scientific notation weirdness

        self.ax_energy.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax_energy.get_yaxis().get_major_formatter().set_useOffset(False)

        self.ax_energy.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_energy.get_yaxis().get_major_formatter().set_scientific(False)

        # ============================================================
        # INTERNAL UPDATE THROTTLES
        # ============================================================

        self.telemetry_update_rate = 5
        self.energy_rescale_rate = 15
        self.map_update_rate = 6

    # ================================================================
    # FRAME RENDERER
    # ================================================================

    def render_frame(
        self,
        frame_idx,
        state,
        current_time,
        scheme_name,
        energies=None
    ):

        pe, ke, total_e = energies

        # ============================================================
        # RESET ON LOOP
        # ============================================================

        if frame_idx == 0:

            self.time_history.clear()
            self.pe_history.clear()
            self.ke_history.clear()
            self.total_history.clear()
            self.loss_history.clear()

            self.initial_energy = total_e

            self.history_matrix.fill(0)

        # ============================================================
        # FIELD EXTRACTION
        # ============================================================

        display_y = state[0] if self.ndim > 1 else state

        # ============================================================
        # LIVE FIELD UPDATE
        # ============================================================

        self.line_pos.set_data(self.x, display_y)

        if self.ndim > 1 and state.shape[0] > 1:
            self.line_vel.set_data(self.x, state[1])

        # ============================================================
        # SPACETIME MATRIX UPDATE
        # ============================================================

        if frame_idx < self.max_frames:

            self.history_matrix[frame_idx] = display_y

            # Reduce expensive full texture uploads

            if frame_idx % self.map_update_rate == 0:
                self.im.set_data(self.history_matrix)

        # ============================================================
        # ENERGY TRACKING
        # ============================================================

        energy_loss = self.initial_energy - total_e
        
        # ============================================================
        # INITIAL ENERGY AXIS CALIBRATION
        # ============================================================

        if not self.energy_axis_initialized:

            estimated_max = max(
                abs(pe),
                abs(ke),
                abs(total_e),
                1.0
            )

            self.ax_energy.set_ylim(
                -0.1 * estimated_max,
                1.1 * estimated_max
            )

            self.energy_axis_initialized = True

        self.time_history.append(current_time)
        self.pe_history.append(pe)
        self.ke_history.append(ke)
        self.total_history.append(total_e)
        self.loss_history.append(energy_loss)

        self.line_pe.set_data(
            self.time_history,
            self.pe_history
        )

        self.line_ke.set_data(
            self.time_history,
            self.ke_history
        )

        self.line_total.set_data(
            self.time_history,
            self.total_history
        )

        self.line_loss.set_data(
            self.time_history,
            self.loss_history
        )

        # ============================================================
        # OCCASIONAL ENERGY RESCALING
        # ============================================================

        if (
            frame_idx % self.energy_rescale_rate == 0
            and len(self.time_history) > 10
        ):

            if self.eq_name in ["diffusion", "advection"]:

                all_energies = (
                    self.total_history
                    + self.loss_history
                )

            else:

                all_energies = (
                    self.pe_history
                    + self.ke_history
                    + self.total_history
                    + self.loss_history
                )

            min_e = min(all_energies)
            max_e = max(all_energies)

            if np.isnan(min_e) or np.isnan(max_e) or np.isinf(min_e) or np.isinf(max_e):
                pass
            else:
                dynamic_range = max(abs(max_e), abs(min_e))

                margin = max(1e-6, 0.15 * dynamic_range)

                self.ax_energy.set_ylim(
                    min_e - margin,
                    max_e + margin
                )

        # ============================================================
        # TELEMETRY UPDATE
        # ============================================================

        if frame_idx % self.telemetry_update_rate == 0:

            cfl = self.dt / self.dx

            self.txt.set_text(
                f"TIME: {current_time:.3f}s\n"
                f"FRAME: {frame_idx:03d}\n"
                f"SOLVER: {scheme_name.upper()}\n"
                f"CFL RATIO: {cfl:.3f}\n"
                f"NET ENERGY: {total_e:.3f}J\n"
                f"ENERGY LOSS: {energy_loss:.3f}J"
            )

        # ============================================================
        # START DELAY
        # ============================================================

        if frame_idx == 0 and self.start_delay > 0:

            self.fig.canvas.draw_idle()
            plt.pause(0.001)

            import time

            print(
                f"Dashboard primed. "
                f"Holding Frame 0 for {self.start_delay}s..."
            )

            time.sleep(self.start_delay)

            print("Simulation timeline released!")

        # ============================================================
        # RETURN ARTISTS FOR BLITTING
        # ============================================================

        return (
            self.line_pos,
            self.line_vel,
            self.im,
            self.line_pe,
            self.line_ke,
            self.line_total,
            self.line_loss,
            self.txt,
        )
    
    # ================================================================
    # CLEAN SHUTDOWN
    # ================================================================

    def close(self):
        """
        Safely destroys the figure window while bypassing the 
        matplotlib animation teardown race-condition bug.
        """
        try:
            plt.close(self.fig)
        except AttributeError:
            pass # The timer is already dead, ignore the error