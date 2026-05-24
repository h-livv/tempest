import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def grid1d(n, boundary, operator, equation, integrator, dt, dx):
    
    #Past, present, future states of the cells
    # u(x, t-dt), u(x, t), u(x, t + dt)
    u_pres = np.zeros(n)
    
    x_axis = np.arange(n)

    u_pres = np.exp(-((x_axis - 125) / 10)**2)
    
    #Visualization animation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, n)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("1D")
    ax.grid(True)
    
    x_axis = np.arange(n)
    line, = ax.plot(x_axis, u_pres, color='dodgerblue', lw=2)
    
    def update_frame(t):
        nonlocal u_pres #Creates new variables inside function, does not modify the original
        
        line.set_ydata(u_pres)
        
        u_pres = integrator(u_pres, t, dt, dx, boundary, operator, equation)
        
        return line,
            
    ani = animation.FuncAnimation(
    fig, 
    update_frame, 
    frames=1000, 
    interval=20, 
    blit=True
    )

    plt.show()
    
    