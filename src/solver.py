import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from visualizations.visualization import TempestVisualizer
from diagnostics import stability, validation
from diagnostics.tracker import DataTracker

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
    # Initial grid structure
    x = np.arange(N, dtype=float) * dx

    # Initial state - multidimensional array based on number of fields
    state = init_state(N, x)

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
    tracker = DataTracker(FINAL_TIME, dt, RECORD_INTERVAL, N)

    def _extract_field(s):
        return s[0] if s.ndim > 1 else s

    def _append_snapshot():
        actual_u = _extract_field(state)
        # Fetch clean analytical state
        true_u = validation.validation(
            equation, state, init_state, N, x, current_time, coefficient, boundary.__name__, dx
        )
        
        _, _, total_e = stability.tracking(state, dx, boundary, equation.__name__, coefficient)
        
        # Delegate storage and error computation
        tracker.record(current_time, actual_u, true_u, total_e)

    def _advance_to(target_step):
        nonlocal state, current_time, step
        target_step = min(target_step, total_steps)
        while step < target_step:
            state = integrator(
                state, current_time, dt, dx, boundary, operator, equation, coefficient
            )
            step += 1
            current_time = step * dt
            if step % record_interval == 0 or step == total_steps:
                _append_snapshot()

    # Record and render the initial condition once (guards duplicate frame-0 callbacks).
    _append_snapshot()
    visualizer.render_frame(0, state, current_time, integrator.__name__, stability.tracking(
        state, dx, boundary, equation.__name__, coefficient
    ))

    def update_frame(frame):
        if frame > 0:
            _advance_to(frame * steps_per_frame)

        energies = stability.tracking(state, dx, boundary, equation.__name__, coefficient)
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__, energies)

        if frame == visualizer.max_frames - 1:
            if step < total_steps:
                _advance_to(total_steps)
            print(
                f"Simulation complete. Recorded {len(tracker.time[:tracker.idx])} snapshots "
                f"(every {record_interval} step(s)). Closing plot window..."
            )
            
            # THE FIX: Attach the timer to the visualizer so the Garbage Collector doesn't kill it!
            visualizer.close_timer = visualizer.fig.canvas.new_timer(interval=50) 
            visualizer.close_timer.single_shot = True
            visualizer.close_timer.add_callback(visualizer.close)
            visualizer.close_timer.start()

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
        "x": x,
        "final_numerical": tracker.numerical[-1] if tracker.idx > 0 else None,
        "final_analytic": tracker.analytical[-1] if tracker.idx > 0 else None,
        "history_dataframe": tracker.get_history_dataframe(),
        "raw_tensor_data": tracker.numerical[:tracker.idx]
    }
