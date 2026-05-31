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
    # 1. Use your elegant dynamic pad width logic
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    
    # 2. Use 'edge' mode to copy boundary values outward symmetrically
    padded = np.pad(state, pad_width=pad_width, mode='edge')
    
    # 3. Dynamic Physics Check: If it's a multi-variable coupled system (like wave/shallow water)
    if padded.ndim > 1 and padded.shape[0] == 2:
        # Invert only the second row's ghost cells (velocity/momentum) to create a solid wall bounce
        padded[1, 0] = -padded[1, 1]
        padded[1, -1] = -padded[1, -2]
        
    return padded