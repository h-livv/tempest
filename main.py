import numpy as np
from Core import grid
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators

# Spark the engine!
grid.grid1d(
    n=250,
    boundary=boundaries.reflect,          # Protects edges by copying boundary values
    operator=operators.gradient,        # Calculates first spatial derivative (u_x)
    equation=equations.advection,       # Applies du/dt = -c * u_x
    integrator=integrators.euler,       # Explicitly uses the 1D Euler time-stepper
    dt=0.1,                             # Continuous time-step increment
    dx=1.0                              # Continuous cell spacing scale
)