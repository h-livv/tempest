import numpy as np

def wave_gauss(N, x):
    center = 0.5 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    init_vel = np.zeros(N)
    
    init_state = np.vstack([init_pos, init_vel])
    
    return init_state

def shallow_gauss(N, x):
    init_height = 1.0 + np.exp(-0.0001 * (x - N/5)**2)
    init_vel = np.zeros(N)
    
    init_state = np.vstack([init_height, init_vel])
    
    return init_state

def advec_gauss(N, x):
    center = 0.5 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    
    init_state = np.vstack([init_pos])
    
    return init_state

def diff_gauss(N, x):
    init_temp = np.exp(-0.01 * (x - N/2)**2)
    
    init_state = np.vstack([init_temp])
    
    return init_state

def diff_rod(N, x):
    init_temp = np.where(x < N // 5, 0.5, 0.0)
    
    init_state = np.vstack([init_temp])
    
    return init_state

def shallow_dam(N, x):
    init_h = np.where(x < N/2, 3.5, 1.0)
    init_v = np.zeros(N)
    
    init_state = np.vstack([init_h, init_v])
    
    return init_state

def shallow_collision(N, x):
    init_h= np.ones(N) * 1.5
    init_v = np.where(x < N/2, 2.0, -2.0)
    
    init_state = np.vstack([init_h, init_v])
    
    return init_state

def shallow_ripples(N, x):
    ambient = 1.5
    
    init_h = ambient + 0.5 * np.sin(2 * np.pi * x / N) + 0.15 * np.cos(10 * np.pi * x / N)
    init_v = np.ones(N) * 0.5
    
    init_state = np.vstack([init_h, init_v])
    
    return init_state

def constant(N, x):
    
    init_h = np.ones(N)
    init_v = np.zeros(N)
    
    init_state = np.vstack([init_h, init_v])
    
    return init_state

def shallow_peak(N, x):
    ambient_depth = 1.0
    center_idx = N // 2
    half_width = 1
    
    init_h = np.ones(N) * ambient_depth
    init_h[center_idx - half_width : center_idx + half_width] = 100
    
    init_v = np.zeros(N)
    init_state = np.vstack([init_h, init_v])
    
    return init_state