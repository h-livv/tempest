import numpy as np

#Acts as an open region
#Works by creating a zero-gradient boundary condition. Index 0 stretches to the left and index -1 stretches to the right.
def edge(state):
    #Dynamic pad width to accommodate both first and second order PDEs
    #Basically adds one "ghost cell" to either side of the grid
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='edge')
    
#Acts as a fixed wall
#Sets boundary cells to absolute zero
def constant(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='constant')

#Periodic
#Maps right boundary to the left and vice versa
def periodic(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='wrap')

#Reflects
#MUST EXPLORE IN DETAIL MATHEMATICALLY
def reflect(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    
    #edge creates a zero-gradient condition, telling the wave that the exact same shape exists on the other side of the boundary
    padded = np.pad(state, pad_width=pad_width, mode='edge')
    
    #At a high level, inverts sign of velocity, forcing an anti-symmetric boundary reflection.
    if padded.ndim > 1 and padded.shape[0] == 2:
        padded[1, 0] = -padded[1, 1]
        padded[1, -1] = -padded[1, -2]
        
    return padded