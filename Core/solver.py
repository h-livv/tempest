import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from Visualizations.visualization import TempestVisualizer
from Core import stability
from Validation import validation
from Experiments import init_conditions

def solver(N, init_state, boundary, operator, equation, integrator, coefficient, dt, dx):
    #Initial grid structure
    x = np.arange(N, dtype=float) * dx
    
    state = init_state(N, x)
    
    if equation.__name__ == 'diffusion':
        STEPS_PER_FRAME = 2000
    else:
        STEPS_PER_FRAME = 2000
        
    FINAL_TIME = 100.0
    
    total_steps = int(FINAL_TIME / dt)
    
    max_frames = total_steps // STEPS_PER_FRAME
    
    current_time = 0.0
    
    #Visualizations handled by the graph file
    visualizer = TempestVisualizer(state, dx, dt, equation.__name__, max_frames, STEPS_PER_FRAME)
    
    if equation.__name__ == 'advection' and (boundary.__name__ == 'reflect' or boundary.__name__ == 'edge' or boundary.__name__ == 'constant'):
        raise ValueError(
            '''BOUNDARY CONDITION ERROR: It is advised to only use the condition 'periodic' with advection as this most closely replicates physical
            behaviour and enables validation and convergence study.'''
        )

        
    time_data = []
    l2_data = []
    l1_data = []
    max_err_data = []
    numerical_history = []
    analytic_history = []
    
    def update_frame(frame):
        nonlocal state, current_time      #Creates a new variable, does not modify original
        
        #State evolution using preffered integrator
        for _ in range(STEPS_PER_FRAME):
            state = integrator(state, current_time, dt, dx, boundary, operator, equation, coefficient)
            current_time += dt
            
        actual_u = state[0] if (state.ndim > 1 and state.shape[0] == 2) else state
        
        results = validation.validation(equation, actual_u, init_state, N, x, current_time, coefficient, boundary, dx)
        
        time_data.append(current_time)
        l2_data.append(results["l2_error"])
        l1_data.append(results["l1_error"])
        max_err_data.append(results["max_error"])
        numerical_history.append(actual_u.copy())
        analytic_history.append(results["analytic_state"].copy())
        
        print(f"Frame {frame} | Max Error: {results['max_error']:.5e}")
            
        energies = stability.tracking(state, dx, boundary, equation.__name__, coefficient)
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__, energies)
        
        if frame == visualizer.max_frames - 1:
            print("Simulation complete. Saving final state and closing plot window...")
            plt.close(visualizer.fig)
            
        return updated  
  
    #Animation
    ani = animation.FuncAnimation(
        visualizer.fig, 
        update_frame, 
        frames=visualizer.max_frames, 
        interval=0, 
        blit=True,
        repeat =False
    )
    
    plt.show()
    
# Package fine-grained frame-by-frame history data
    history_df = pd.DataFrame({
        "time": time_data,
        "l2_error": l2_data,
        "l1_error": l1_data,
        "max_error": max_err_data
    })
    
    # Return the final state matrices, coordinate grid, and full histories back to the pipeline
    return {
        "x": x,
        "final_numerical": numerical_history[-1],
        "final_analytic": analytic_history[-1],
        "history_dataframe": history_df
    }
        

    
