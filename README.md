# Project Tempest
A modular computational framework for simulating nonlinear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib. 

## Current capabilities:
1. A 1D grid system for field evolution over time.
2. Finite-difference spatial operators.
3. Governing evolution laws, currently including advection.
4. Integration methods, currently including Euler.
5. Customizable boundary conditions.
6. Visualization.

---

## Module overview:
1. **grid.py**
* Describes the 1D grid and initial conditions, handles visualization.
  
2. **boundaries.py**
* Various boundary conditions to experiment with. Includes edge, constant, reflected, symmetric. More to come.

3. **operators.py**
* Finite-difference spatial operators. Includes the gradient, laplacian, and upwind.

4. **equations.py**
* Contains the governing physical equations. Currently includes advection, more to come.

5. **integrators.py**
* Contains the integrators used to evolve the grid state over time. Currently includes Euler, more to come.

---

## Planned features:
1. RK4 and Leapfrog integrators.
2. Wave and diffusion equation implementations.
3. Stability analysis.
4. Coupled multi-field systems.
5. Shallow water equations.

---

**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators.
