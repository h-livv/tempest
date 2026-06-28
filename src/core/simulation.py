"""
Simulation – encapsulates all state and logic needed to run one Tempest
simulation.

Usage::

    config = SimulationConfig(...)
    sim    = Simulation(config)
    results = sim.run()

The class owns:
  - config       the SimulationConfig supplied at construction time
  - grid         the Grid built from config.shape / config.spacing
  - state        the current Field (ScalarField or VectorField)
  - time         the current simulation time (float)
  - step         the current step count (int)
  - tracker      the DataTracker instance

The existing ``solver()`` function in src/solver.py is left intact.
Both code-paths are verified to produce identical results.
"""

from __future__ import annotations

import numpy as np

from src.grid import Grid
from src.fields import Field, ScalarField, VectorField
from src.core.config import SimulationConfig
from src.core.results import SimulationResults
from diagnostics.tracker import DataTracker
from diagnostics import stability, validation
from visualizations.visualization import TempestVisualizer


class Simulation:
    """Run a single Tempest simulation described by *config*."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        # ------------------------------------------------------------------
        # Build the grid
        # ------------------------------------------------------------------
        self.grid = Grid(shape=config.shape, spacing=config.spacing)

        # ------------------------------------------------------------------
        # Build the initial field
        # ------------------------------------------------------------------
        raw = config.initial_condition(self.grid)
        if isinstance(raw, Field):
            self.state = raw
        elif raw.ndim == self.grid.ndim:
            self.state = ScalarField(self.grid, raw)
        else:
            self.state = VectorField(self.grid, raw)

        # ------------------------------------------------------------------
        # Simulation clock
        # ------------------------------------------------------------------
        self.time: float = 0.0
        self.step: int = 0

        # ------------------------------------------------------------------
        # Tracker  (same arguments the old solver used)
        # ------------------------------------------------------------------
        self.tracker = DataTracker(
            final_time=config.final_time,
            dt=config.dt,
            record_interval=config.record_interval,
            grid_size=self.grid.shape,
        )

    # ------------------------------------------------------------------
    # Internal helpers  (mirrored from solver.py without modification)
    # ------------------------------------------------------------------

    # Extracts the primary 1-D/N-D array from the current Field for diagnostic use.
    def _extract_field(self) -> np.ndarray:
        """Return the raw array from the current state, unwrapping 1-component stacks."""
        data = self.state.data if hasattr(self.state, "data") else self.state
        return data[0] if data.ndim > self.grid.ndim else data

    # Records the current numerical and analytical state into the DataTracker.
    def _append_snapshot(self) -> None:
        """Record one diagnostic snapshot into the tracker."""
        actual_u = self._extract_field()

        # Analytical reference (returns zeros when no closed form exists).
        #
        # validation.validation() expects an init_condition callable with the
        # OLD signature (N, x) -> array, not the new (grid) -> Field|ndarray
        # signature.  We bridge the gap with a thin adapter so that
        # validation.py does not need to change in this step.
        x_compat = (
            self.grid.coordinates[0]
            if self.grid.ndim == 1
            else self.grid.coordinates
        )
        N_compat = (
            self.grid.shape[0] if self.grid.ndim == 1 else self.grid.shape
        )
        dx_compat = (
            self.grid.spacing[0] if self.grid.ndim == 1 else self.grid.spacing
        )

        # Adapter: calls the new-style IC with a temporary single-axis Grid
        # built from the N and x arguments supplied by validation.validation.
        def _ic_adapter(N_arg, x_arg):
            """Evaluate initial_condition at the exact x_arg coordinates.

            validation.validation() passes (N, x_shifted) to evaluate the IC
            at specific spatial positions (e.g. shifted for advection).  We
            build a minimal duck-typed proxy that gives the IC access to those
            exact positions via the grid.coordinates interface, so no Grid
            reconstruction is needed.
            """
            class _GridProxy:
                ndim = 1
                shape = (len(x_arg),)
                spacing = (dx_compat,)
                coordinates = [x_arg]  # IC reads grid.coordinates[0]

            result = self.config.initial_condition(_GridProxy())
            data = result.data if hasattr(result, "data") else np.asarray(result)
            # Ensure (1, N) shape – validation.py indexes with [0].
            # Use squeeze rather than data[0] to avoid NumPy 2.x view restrictions
            # on non-owned arrays.
            flat = np.squeeze(data, axis=0) if data.ndim > 1 else data
            return flat[np.newaxis]

        true_u = validation.validation(
            self.config.equation,
            self.state.data,
            _ic_adapter,
            N_compat,
            x_compat,
            self.time,
            self.config.coefficient,
            self.config.boundary.__name__,
            dx_compat,
        )

        _, _, total_e = stability.tracking(
            self.state,
            self.grid,
            self.config.boundary,
            self.config.equation.__name__,
            self.config.coefficient,
        )

        self.tracker.record(self.time, actual_u, true_u, total_e)

    # Calls the integrator once and advances the clock by one time step.
    def _advance_one_step(self) -> None:
        """Advance the simulation by a single time step."""
        next_state = self.config.integrator(
            self.state,
            self.time,
            self.config.dt,
            self.grid,
            self.config.boundary,
            self.config.operator,
            self.config.equation,
            self.config.coefficient,
        )
        # Preserve Field type when the integrator returns a raw array
        if hasattr(self.state, "grid") and not hasattr(next_state, "grid"):
            self.state = self.state.__class__(self.state.grid, next_state)
        else:
            self.state = next_state
        self.step += 1
        self.time = self.step * self.config.dt

    # Drives the time-loop until the simulation reaches a specified step count.
    def _advance_to(self, target_step: int) -> None:
        """Run the time-loop up to (but not beyond) *target_step*."""
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
        Run the full simulation and return a :class:`SimulationResults`.

        The internal logic is identical to the original ``solver()`` function.
        The visualiser, animation loop, and diagnostic recording all behave
        the same way.  The only difference is the return type: instead of a
        plain dict the caller receives a ``SimulationResults`` dataclass.
        """
        import matplotlib
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------------
        # Setup – derive timing constants and build the visualiser
        # ------------------------------------------------------------------
        config = self.config
        grid = self.grid
        total_steps = int(config.final_time / config.dt)
        steps_per_frame = max(1, int(config.steps_per_frame))
        max_frames = max(1, total_steps // steps_per_frame)

        visualizer = TempestVisualizer(
            self.state,
            config.spacing[0] if grid.ndim == 1 else config.spacing,
            config.dt,
            config.equation.__name__,
            max_frames,
            steps_per_frame,
            final_time=config.final_time,
        )

        # ------------------------------------------------------------------
        # Initial snapshot – record t=0 before any time stepping begins
        # ------------------------------------------------------------------
        self._append_snapshot()
        visualizer.render_frame(
            0,
            self.state,
            self.time,
            config.integrator.__name__,
            stability.tracking(
                self.state.data,
                grid,
                config.boundary,
                config.equation.__name__,
                config.coefficient,
            ),
        )

        # ------------------------------------------------------------------
        # Simulation loop – advance and render frame-by-frame
        # ------------------------------------------------------------------
        def update_frame(frame: int):
            if frame > 0:
                self._advance_to(frame * steps_per_frame)

            energies = stability.tracking(
                self.state.data,
                grid,
                config.boundary,
                config.equation.__name__,
                config.coefficient,
            )
            updated = visualizer.render_frame(
                frame, self.state, self.time, config.integrator.__name__, energies
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
        # Package results – collect outputs into SimulationResults
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
