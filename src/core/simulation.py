"""
Simulation – high-level Tempest pipeline orchestrator.
"""

from __future__ import annotations

import numpy as np

from src.mesh.grid import Grid
from src.mesh.fields import Field, ScalarField, VectorField
from src.core.config import SimulationConfig
from src.core.results import SimulationResults
from src.diagnostics.tracker import DataTracker
from src.validation import validation
from src.diagnostics import stability

from src.visualization.visualization import TempestVisualizer


class Simulation:
    """
    Simulation orchestrator.
    
    Think of the Simulation object as the grand conductor of the Tempest orchestra.
    It owns and manages the lifecycle of a simulation, coordinating between:
      - The spatial domain context (Grid and Field abstractions)
      - The numerical progression schemes (Integrators and operators)
      - The analytical validation & stability metrics (Validation and Stability diagnostics)
      - The output visualizations (TempestVisualizer renderers)
      
    Attributes:
        config (SimulationConfig): Container representing initial options and configuration parameters.
        grid (Grid): Discretised spatial domain coordinate ruler.
        state (Field): Wrapped array representing current simulation physics variables.
        time (float): Current elapsed simulation clock time.
        step (int): Count of elapsed integrator advancement loops.
        equation (Equation): The instantiated PDE physics rules to evaluate.
        tracker (DataTracker): Diagnostics snapshot accumulator.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        =======================================================================
        SECTION 1: Simulation Setup
        =======================================================================
        Sets up the grid, constructs initial states, instantiates the physics
        models, and logs initial structures.
        """
        self.config = config

        # Construct spatial coordinates
        self.grid = Grid(shape=config.shape, spacing=config.spacing)

        # Build initial physical state using injected InitialCondition callable
        raw = config.initial_condition(self.grid)
        if isinstance(raw, Field):
            self.state = raw
        elif raw.ndim == self.grid.ndim:
            self.state = ScalarField(self.grid, raw)
        else:
            self.state = VectorField(self.grid, raw)

        # Clock initiation
        self.time: float = 0.0
        self.step: int = 0
        
        # Pull equation object from config
        self.equation = self.config.equation

        # Tracker for diagnostics outputs
        self.tracker = DataTracker(
            final_time=config.final_time,
            dt=config.dt,
            record_interval=config.record_interval,
            grid_size=self.grid.shape,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _extract_field(self) -> np.ndarray:
        """Extracts the primary 1-D/N-D array from the current Field for diagnostic plotting."""
        data = self.state.data if hasattr(self.state, "data") else self.state
        return data[0] if data.ndim > self.grid.ndim else data

    def _append_snapshot(self) -> None:
        """
        =======================================================================
        SECTION 3: Diagnostics & Validation Analysis
        =======================================================================
        Invokes analytical validation targets and monitors total system energy conservation.
        """
        actual_u = self._extract_field()

        # Compute exact physical analytical solution for error matching
        true_u = validation.validation(
            self.equation,
            self.state.data,
            self.config.initial_condition,
            self.grid,
            self.time,
            self.config.boundary.__name__,
        )

        # Track stability diagnostics (potential, kinetic, and total energy)
        _, _, total_e = stability.tracking(
            self.state,
            self.grid,
            self.config.boundary,
            self.equation,
        )

        self.tracker.record(self.time, actual_u, true_u, total_e)

    def _advance_one_step(self) -> None:
        """
        =======================================================================
        SECTION 2: Solver Execution (Time-Stepping)
        =======================================================================
        Invokes the numerical time-integrator to advance the PDE one clock tick forward.
        """
        next_state = self.config.integrator(
            self.state,
            self.time,
            self.config.dt,
            self.grid,
            self.config.boundary,
            self.config.operator,
            self.equation,
        )
        
        # Re-wrap in grid Field objects if solver returned standard ndarray
        if hasattr(self.state, "grid") and not hasattr(next_state, "grid"):
            self.state = self.state.__class__(self.state.grid, next_state)
        else:
            self.state = next_state
            
        self.step += 1
        self.time = self.step * self.config.dt

    def _advance_to(self, target_step: int) -> None:
        """Drives the time-loop stepping execution up to a target step count."""
        total_steps = int(self.config.final_time / self.config.dt)
        target_step = min(target_step, total_steps)
        record_interval = max(1, int(self.config.record_interval))
        while self.step < target_step:
            self._advance_one_step()
            if self.step % record_interval == 0 or self.step == total_steps:
                self._append_snapshot()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResults:
        """
        Runs the full PDE integration simulation.
        
        It orchestrates the initialization snapshot, drives the time stepping loop,
        updates the output visualizations in real-time or headlessly, and packages
        the output arrays.
        
        Returns:
            SimulationResults: Dataclass wrapping historical trajectory states.
        """
        import matplotlib
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        # Setup constants and instantiate renderer
        config = self.config
        grid = self.grid
        total_steps = int(config.final_time / config.dt)
        steps_per_frame = max(1, int(config.steps_per_frame))
        max_frames = max(1, total_steps // steps_per_frame)

        # Build TempestVisualizer to orchestrate plotting output files
        visualizer = TempestVisualizer(
            self.state,
            config.spacing[0] if grid.ndim == 1 else config.spacing,
            config.dt,
            self.equation.__name__,
            max_frames,
            steps_per_frame,
            final_time=config.final_time,
        )

        # Initial Snapshot before time integration loops
        self._append_snapshot()
        
        # Compute energy stability at t=0
        init_energies = stability.tracking(
            self.state.data,
            grid,
            config.boundary,
            self.equation,
        )
        
        # Render initial frame
        visualizer.render_frame(
            0,
            self.state,
            self.time,
            config.integrator.__name__,
            init_energies,
        )

        # ------------------------------------------------------------------
        # SECTION 5: Visualization & Animation update loop
        # ------------------------------------------------------------------
        def update_frame(frame: int):
            if frame > 0:
                self._advance_to(frame * steps_per_frame)

            current_energies = stability.tracking(
                self.state.data,
                grid,
                config.boundary,
                self.equation,
            )
            updated = visualizer.render_frame(
                frame, self.state, self.time, config.integrator.__name__, current_energies
            )

            if frame == visualizer.max_frames - 1:
                if self.step < total_steps:
                    self._advance_to(total_steps)

                is_headless = matplotlib.get_backend().lower() == "agg"
                msg = (
                    f"Simulation complete. Recorded "
                    f"{len(self.tracker.time[:self.tracker.idx])} snapshots "
                    f"(every {max(1, int(config.record_interval))} step(s))."
                )
                if is_headless:
                    print(msg + " Closing plot window...")
                    visualizer.close_timer = visualizer.fig.canvas.new_timer(
                        interval=50
                    )
                    visualizer.close_timer.single_shot = True
                    visualizer.close_timer.add_callback(visualizer.close)
                    visualizer.close_timer.start()
                else:
                    print(msg + " Keep plot window open.")
            return updated

        # Execute headless sweep or run Matplotlib dashboard
        if matplotlib.get_backend().lower() == "agg":
            for f in range(visualizer.max_frames):
                update_frame(f)
        else:
            anim = animation.FuncAnimation(
                visualizer.fig,
                update_frame,
                frames=visualizer.max_frames,
                interval=0,
                blit=False,
                repeat=False,
            )
            plt.show()

        # ------------------------------------------------------------------
        # SECTION 4: Data Export packaging
        # ------------------------------------------------------------------
        tracker = self.tracker

        return SimulationResults(
            grid=grid,
            final_state=self.state,
            history=tracker.get_history_dataframe(),
            final_numerical=(
                tracker.numerical[tracker.idx - 1] if tracker.idx > 0 else None
            ),
            final_analytical=(
                tracker.analytical[tracker.idx - 1] if tracker.idx > 0 else None
            ),
            raw_tensor_data=tracker.numerical[: tracker.idx],
            energy_history={
                "time": visualizer.time_history.copy(),
                "pe": visualizer.pe_history.copy(),
                "ke": visualizer.ke_history.copy(),
                "total": visualizer.total_history.copy(),
                "loss": visualizer.loss_history.copy(),
            },
        )
