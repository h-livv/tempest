import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from visualizations.visualization import TempestVisualizer
from diagnostics import stability, validation

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

    # Lists for data collection (independent of animation cadence)
    time_data = []
    l2_data = []
    l1_data = []
    max_err_data = []
    numerical_history = []
    analytic_history = []

    def _extract_field(s):
        return s[0] if s.ndim > 1 else s

    def _append_snapshot():
        actual_u = _extract_field(state)
        results = validation.validation(
            equation, state, init_state, N, x, current_time, coefficient, boundary.__name__, dx
        )
        time_data.append(current_time)
        l2_data.append(results["l2_error"])
        l1_data.append(results["l1_error"])
        max_err_data.append(results["max_error"])
        numerical_history.append(actual_u.copy())
        analytic_history.append(results["analytic_state"].copy())

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
                f"Simulation complete. Recorded {len(time_data)} snapshots "
                f"(every {record_interval} step(s)). Closing plot window..."
            )
            plt.close(visualizer.fig)

        return updated

    # Animation (visual cadence only)
    ani = animation.FuncAnimation(
        visualizer.fig,
        update_frame,
        frames=visualizer.max_frames,
        interval=0,
        blit=False,
        repeat=False
    )

    plt.show()

    history_df = pd.DataFrame({
        "time": time_data,
        "l2_error": l2_data,
        "l1_error": l1_data,
        "max_error": max_err_data
    })

    return {
        "x": x,
        "final_numerical": numerical_history[-1],
        "final_analytic": analytic_history[-1],
        "history_dataframe": history_df
    }
