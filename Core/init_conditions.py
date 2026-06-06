import numpy as np

#Gaussian waves

def wave_gauss(N, x):
    center = 0.5 * x.max()
    sigma = 10.0

    init_pos = np.exp(-((x - center)**2) / (2 * sigma**2))
    init_vel = np.zeros(N)
    
    init_state = np.vstack([init_pos, init_vel])
    return init_state

def shallow_gauss(N, x):
    center = 0.2 * x.max()

    init_height = 1.0 + np.exp(-0.0001 * (x - center)**2)
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
    center = 0.5 * x.max()
    
    init_temp = np.exp(-0.01 * (x - center)**2)
    
    init_state = np.vstack([init_temp])
    return init_state

#Sharp peaks (Diraq delta function + square shapes)

def shallow_peak(N, x):
    ambient_depth = 1.0
    center_idx = N // 2
    half_width = 1
    
    init_h = np.ones(N) * ambient_depth

    init_h[center_idx - half_width : center_idx + half_width] = 100.0
    
    init_v = np.zeros(N)
    init_state = np.vstack([init_h, init_v])
    return init_state

def wave_peak(N, x):
    init_pos = np.zeros(N)
    center_idx = N // 2
    init_pos[center_idx] = 2.0
    
    init_vel = np.zeros(N)
    return np.vstack([init_pos, init_vel])

def wave_square(N, x):
    init_pos = np.where((x > 0.4 * x.max()) & (x < 0.6 * x.max()), 1.0, 0.0)
    init_vel = np.zeros(N)
    
    return np.vstack([init_pos, init_vel])

def advec_peak(N, x):
    init_pos = np.zeros(N)
    center_idx = N // 2
    init_pos[center_idx] = 1.0
    
    return np.vstack([init_pos])

def advec_square(N, x):
    init_pos = np.where((x > 0.4 * x.max()) & (x < 0.6 * x.max()), 1.0, 0.0)
    
    return np.vstack([init_pos])

def diff_peak(N, x):
    init_temp = np.zeros(N)
    center_idx = N // 2
    init_temp[center_idx] = 10.0
    
    return np.vstack([init_temp])

#A rod heated on one end
def diff_rod(N, x):
    init_temp = np.where(x < 0.2 * x.max(), 0.5, 0.0)
    
    init_state = np.vstack([init_temp])
    return init_state

#Dam breaking
def shallow_dam(N, x):
    init_h = np.where(x < 0.5 * x.max(), 3.5, 1.0)
    init_v = np.zeros(N)
    
    init_state = np.vstack([init_h, init_v])
    return init_state

#Two waves colliding at the center
def shallow_collision(N, x):
    init_h = np.ones(N) * 1.5
    init_v = np.where(x < 0.5 * x.max(), 2.0, -2.0)
    
    init_state = np.vstack([init_h, init_v])
    return init_state

#Constant for diagnostic purposes
def constant(N, x):
    init_h = np.ones(N)
    init_v = np.zeros(N)
    
    init_state = np.vstack([init_h, init_v])
    
    return init_state