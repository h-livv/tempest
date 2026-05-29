import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Visualizations.visualization import TempestVisualizer
from Core import stability
from Core import operators

def grid1d(init_state, boundary, operator, equation, integrator, coefficient, dt, dx):
    #Initial grid structure
    
    N = 250
    x = np.linspace(0, N, N)
    
    state = init_state(N, x)
    
    #Visualizations handled by the graph file
    visualizer = TempestVisualizer(state, dx, dt, equation.__name__)
    
    if equation.__name__ == 'advection' and boundary.__name__ == 'reflect' and operator.__name__ == 'gradient':
        raise ValueError(
            "BOUNDARY CONDITION ERROR: It is advised to not use the condition 'reflect' and operator 'gradient' with advection as this violates physical behaviour"
        )
    
    if equation.__name__ == 'diffusion':
        STEPS_PER_FRAME = 100
    else:
        STEPS_PER_FRAME = 50
    
    def update_frame(frame):
        nonlocal state      #Creates a new variable, does not modify original
        current_time = frame * dt * STEPS_PER_FRAME  #Current time
        
        #State evolution using preffered integrator
        for _ in range(STEPS_PER_FRAME):
            state = integrator(state, current_time, dt, dx, boundary, operator, equation, coefficient)
            
        energies = stability.tracking(state, dx, boundary, equation.__name__, coefficient)
        
        updated = visualizer.render_frame(frame, state, current_time, integrator.__name__, energies)
        return updated  
            
    #Animation
    ani = animation.FuncAnimation(
        visualizer.fig, 
        update_frame, 
        frames=visualizer.max_frames, 
        interval=0, 
        blit=True,
        repeat = True
    )

    plt.show()