import numpy as np

def gradient(u_pres, dx, boundary):
    u_x = boundary(u_pres)
    left = u_x[0:-2]
    center = u_x[1:-1]
    right = u_x[2:]
    
    grad = (right - left)/(2*dx)
    return grad

def laplacian(u_pres, dx, boundary):
    u_x = boundary(u_pres)
    left = u_x[0:-2]
    center = u_x[1:-1]
    right = u_x[2:]
    
    lap = (right - 2*center + left)/(dx*dx)
    return lap

def upwind(u_pres, dx, boundary):
    padded = boundary(u_pres)
    
    left = padded[0:-2]
    center = padded[1:-1]
    
    return (center - left) / dx