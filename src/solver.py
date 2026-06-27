import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from visualizations.visualization import TempestVisualizer
from diagnostics import stability, validation
from diagnostics.tracker import DataTracker
from src.grid import Grid
from src.fields import ScalarField, VectorField

def solver(
    N,
    init_state,
    boundary,
    operator,
    equation,
    integrator,
    coefficient,
    dt,
    dx,
    FINAL_TIME,
    STEPS_PER_FRAME,
    RECORD_INTERVAL=1,
):
    # Construct unified Grid
    # Phase 2 Backward compatibility
    # Ensure N and dx are sequences if they are passed as scalars.
    actual_shape = N if isinstance(N, tuple) else (N,)
    actual_spacing = dx if isinstance(dx, tuple) else (dx,)
    grid = Grid(shape=actual_shape, spacing=actual_spacing)

    # Initial state initialization agnostic of dimensions
    if isinstance(actual_shape, tuple) and len(actual_shape) > 1:
        if equation.__name__ == 'burgers':
            state_data = init_state(*actual_shape, *grid.coordinates, nu=coefficient)
        else:
            state_data = init_state(*actual_shape, *grid.coordinates)
    else:
        if equation.__name__ == 'burgers':
            state_data = init_state(actual_shape[0], grid.coordinates[0], nu=coefficient)
        else:
            state_data = init_state(actual_shape[0], grid.coordinates[0])

    # Wrap raw data in appropriate Field abstraction
    if state_data.shape == grid.shape or (state_data.ndim == grid.ndim + 1 and state_data.shape[0] == 1):
        if state_data.ndim == grid.ndim + 1 and state_data.shape[0] == 1:
            state_data = state_data[0] # Unpack single field
        state = ScalarField(grid, state_data)
    else:
        state = VectorField(grid, state_data)

    total_steps = int(FINAL_TIME / dt)
    record_interval = max(1, int(RECORD_INTERVAL))
    steps_per_frame = max(1, int(STEPS_PER_FRAME))
    max_frames = max(1, total_steps // steps_per_frame)

    current_time = 0.0
    step = 0

    visualizer = TempestVisualizer(
        state,
        dx,
        dt,
        equation.__name__,
        max_frames,
        steps_per_frame,
        final_time=FINAL_TIME,
    )

    if equation.__name__ == 'advection' and (boundary.__name__ == 'reflect' or boundary.__name__ == 'edge' or boundary.__name__ == 'constant'):
        raise ValueError(
            '''BOUNDARY CONDITION ERROR: It is advised to only use the condition 'periodic' with advection as this most closely replicates physical
            behaviour and enables accurate validation and convergence study.'''
        )

    # Initialize the Tracker
    # Tracker currently assumes N, will be refactored later if needed
    tracker = DataTracker(FINAL_TIME, dt, RECORD_INTERVAL, N)

    def _extract_field(s):
        data = s.data if hasattr(s, 'data') else s
        return data[0] if data.ndim > grid.ndim else data

    def _append_snapshot():
        actual_u = _extract_field(state)
        # Fetch clean analytical state
        # Pass raw grid coordinate for compatibility
        x_compat = grid.coordinates[0] if grid.ndim == 1 else grid.coordinates
        true_u = validation.validation(
            equation, state.data, init_state, N, x_compat, current_time, coefficient, boundary.__name__, dx
        )
        
        _, _, total_e = stability.tracking(state, grid, boundary, equation.__name__, coefficient)
        
        # Delegate storage and error computation
        tracker.record(current_time, actual_u, true_u, total_e)

    def _advance_to(target_step):
        nonlocal state, current_time, step
        target_step = min(target_step, total_steps)
        while step < target_step:
            next_state = integrator(
                state, current_time, dt, grid, boundary, operator, equation, coefficient
            )
            if hasattr(state, 'grid') and not hasattr(next_state, 'grid'):
                state = state.__class__(state.grid, next_state)
            else:
                state = next_state
            step += 1
            current_time = step * dt
            if step % record_interval == 0 or step == total_steps:
                _append_snapshot()

    # Record and render the initial condition once (guards duplicate frame-0 callbacks).
    _append_snapshot()
    visualizer.render_frame(0, state, current_time, integrator.__name__, stability.tracking(
        state.data, grid, boundary, equation.__name__, coefficient
    ))

    def update_frame(frame):
        if frame > 0:
            _advance_to(frame * steps_per_frame)

        energies = stability.tracking(state.data, grid, boundary, equation.__name__, coefficient)
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__, energies)

        if frame == visualizer.max_frames - 1:
            if step < total_steps:
                _advance_to(total_steps)
            
            is_headless = matplotlib.get_backend().lower() == 'agg'
            if is_headless:
                print(
                    f"Simulation complete. Recorded {len(tracker.time[:tracker.idx])} snapshots "
                    f"(every {record_interval} step(s)). Closing plot window..."
                )
                visualizer.close_timer = visualizer.fig.canvas.new_timer(interval=50) 
                visualizer.close_timer.single_shot = True
                visualizer.close_timer.add_callback(visualizer.close)
                visualizer.close_timer.start()
            else:
                print(
                    f"Simulation complete. Recorded {len(tracker.time[:tracker.idx])} snapshots "
                    f"(every {record_interval} step(s)). Keep plot window open."
                )
        return updated

    import matplotlib
    if matplotlib.get_backend().lower() == 'agg':
        # Headless mode: run the simulation loop manually
        for f in range(visualizer.max_frames):
            update_frame(f)
    else:
        # Animation (visual cadence only)
        anim = animation.FuncAnimation(
            visualizer.fig,
            update_frame,
            frames=visualizer.max_frames,
            interval=0,
            blit=False,
            repeat=False
        )
        plt.show()
    

    return {
        "grid": grid,
        "x": grid.coordinates[0] if grid.ndim == 1 else grid.coordinates,
        "final_numerical": tracker.numerical[-1] if tracker.idx > 0 else None,
        "final_analytic": tracker.analytical[-1] if tracker.idx > 0 else None,
        "history_dataframe": tracker.get_history_dataframe(),
        "raw_tensor_data": tracker.numerical[:tracker.idx]
    }
