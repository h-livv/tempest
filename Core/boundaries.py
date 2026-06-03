import numpy as np

#Acts as an open region
def edge(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)] #Dynamic pad width to accommodate both first and second order PDEs
    return np.pad(state, pad_width=pad_width, mode='edge')
    
#Acts as a fixed wall
def constant(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='constant')

#periodic
def periodic(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='wrap')

#Reflects
def reflect(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    padded = np.pad(state, pad_width=pad_width, mode='edge')
    
    if padded.ndim > 1 and padded.shape[0] == 2:
        padded[1, 0] = -padded[1, 1]
        padded[1, -1] = -padded[1, -2]
        
    return padded