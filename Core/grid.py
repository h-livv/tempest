# Core/grid.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Visualizations.graphs import TempestVisualizer

def grid1d(initial_state, boundary, operator, equation, integrator, dt, dx):
    #Initial grid structure
    state = np.copy(initial_state)
    
    #Visualizations handled by the graph file
    visualizer = TempestVisualizer(state, dx, dt, equation.__name__)
    
    def update_frame(frame):
        nonlocal state      #Creates a new variable, does not modify original
        current_time = frame * dt   #Current time
        
        #State evolution using preffered integrator
        state = integrator(state, current_time, dt, dx, boundary, operator, equation)
        
        #Ship updated arrays to dedicated visualization file
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__)
        return updated
            
    #Animation
    ani = animation.FuncAnimation(
        visualizer.fig, 
        update_frame, 
        frames=visualizer.max_frames, 
        interval=20, 
        blit=True
    )

    plt.show()