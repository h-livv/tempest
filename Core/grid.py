import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Visualizations.visualization import TempestVisualizer
from Core import stability

def grid1d(boundary, operator, equation, integrator, coefficient, dt, dx):
    #Initial grid structure
    
    N = 250
    x = np.linspace(0, N, N)
    
    if equation.__name__ == 'diffusion':
        init_pos = np.where(x < N // 5, 0.5, 0.0)
    
    else:
        init_pos = np.exp(-0.01 * (x - N/2)**2)
    
    init_vel = np.zeros(N)
    init_state = np.vstack([init_pos, init_vel])
    
    state = init_state
    
    #Visualizations handled by the graph file
    visualizer = TempestVisualizer(state, dx, dt, equation.__name__)
    
    def update_frame(frame):
        nonlocal state      #Creates a new variable, does not modify original
        current_time = frame * dt   #Current time
        
        #State evolution using preffered integrator
        state = integrator(state, current_time, dt, dx, boundary, operator, equation, coefficient)
        
        '''center = state.shape[-1] // 2
        if state.ndim > 1:
            state[0][center] = 1.0
        else:
            state[center] = 1.0'''
            
        energies = stability.tracking(state, dx, boundary, equation.__name__, coefficient)
        
        #Ship updated arrays to dedicated visualization file
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__, energies)
        return updated
            
    #Animation
    ani = animation.FuncAnimation(
        visualizer.fig, 
        update_frame, 
        frames=visualizer.max_frames, 
        interval=10, 
        blit=True,
        repeat = True
    )

    plt.show()