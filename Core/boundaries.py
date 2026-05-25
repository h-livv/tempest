import numpy as np

def edge(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)] #Dynamic pad width to accommodate both first and second order PDEs
    return np.pad(state, pad_width=pad_width, mode='edge')
    
def constant(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='constant')

def reflect(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='reflect')

def symmetric(state):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='symmetric')