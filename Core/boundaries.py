import numpy as np

def edge(u_pres):
    u_padded = np.pad(u_pres, pad_width=1, mode='edge')
    return u_padded
    
def constant(u_pres):
    u_padded = np.pad(u_pres, pad_width=1, mode='constant')
    return u_padded

def reflect(u_pres):
    u_padded = np.pad(u_pres, pad_width=1, mode='reflect')
    return u_padded

def symmetric(u_pres):
    u_padded = u_padded = np.pad(u_pres, pad_width=1, mode='symmetric')
    return u_padded