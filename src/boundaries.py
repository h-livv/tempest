import numpy as np

#Acts as an open region
#Works by creating a zero-gradient boundary condition. Index 0 stretches to the left and index -1 stretches to the right.
def edge(state, parity):
    #Dynamic pad width to accommodate both first and second order PDEs
    #Basically adds one "ghost cell" to either side of the grid
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='edge')
    
#Acts as a fixed wall
#Sets boundary cells to absolute zero
def constant(state, parity):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='constant')

#Periodic
#Maps right boundary to the left and vice versa
def periodic(state, parity):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    return np.pad(state, pad_width=pad_width, mode='wrap')

#Reflects
#MUST EXPLORE IN DETAIL MATHEMATICALLY
def reflect(state, parity):
    pad_width = [(0, 0)] * (state.ndim - 1) + [(1, 1)]
    
    #edge creates a zero-gradient condition, telling the wave that the exact same shape exists on the other side of the boundary
    #this handles symmetric fields
    padded = np.pad(state, pad_width=pad_width, mode='edge')
    
    if parity is not None:
        if state.ndim == 1:
            # 1D single-field optimization: parity is applied as a pure scalar
            padded[0] *= parity[0]
            padded[-1] *= parity[0]
        else:
            # Multi-field N-dimensional broadcasting:
            # Dynamically shapes the mask to align perfectly with axis 0 (fields)
            # and injects 1s to broadcast automatically across all spatial axes.
            mask_shape = [len(parity)] + [1] * (state.ndim - 2)
            parity_mask = np.asarray(parity).reshape(mask_shape)
            
            padded[..., 0] *= parity_mask
            padded[..., -1] *= parity_mask
        
    return padded