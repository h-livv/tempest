import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from Visualizations.visualization import TempestVisualizer
from Core import stability
from Validation import validation
from Experiments import init_conditions

def grid1d(init_state, boundary, operator, equation, integrator, coefficient, dt, dx):
    #Initial grid structure
    
    N = 1000
    x = np.arange(N)
    
    state = init_state(N, x)
    
    #Visualizations handled by the graph file
    visualizer = TempestVisualizer(state, dx, dt, equation.__name__)
    
    if equation.__name__ == 'advection' and boundary.__name__ == 'reflect' and operator.__name__ == 'gradient':
        raise ValueError(
            "BOUNDARY CONDITION ERROR: It is advised to not use the condition 'reflect' and operator 'gradient' with advection as this violates physical behaviour"
        )
    
    if equation.__name__ == 'diffusion':
        STEPS_PER_FRAME = 2000
    else:
        STEPS_PER_FRAME = 500
        
    time_data = []
    disp = []
    l2_data = []
    rel_data = []
    relative_state = []
    analytic_state = []
    

    
    def update_frame(frame):
        nonlocal state      #Creates a new variable, does not modify original
        current_time = frame * dt * STEPS_PER_FRAME  #Current time
        
        #State evolution using preffered integrator
        for _ in range(STEPS_PER_FRAME):
            state = integrator(state, current_time, dt, dx, boundary, operator, equation, coefficient)
        
        results = validation.validation(equation, state[0], init_conditions.advec_gauss, N, x, current_time, coefficient)
        
        time_data.append(current_time)
        disp.append(x)
        l2_data.append(results["l2_error"])
        rel_data.append(results["relative_error"])
        relative_state.append(results["relative"])
        analytic_state.append(results["analytic_state"])
        
        print("Numerical peak:", np.argmax(state))
        print("Expected peak:", N/2 + coefficient*current_time)
            
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
    
           
    df = pd.DataFrame({
        "time": time_data,
        "l2_error": l2_data,
        "relative_error": rel_data
            })
        
    df.to_csv("Results/advection/advection_validation.csv", index=False)
    
    np.save(
    "Results/advection/numerical_states.npy",
    np.array(relative_state)
)

    np.save(
        "Results/advection/analytic_states.npy",
        np.array(analytic_state)
    )
        

    
