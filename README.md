# Project Tempest
A modular computational framework for simulating nonlinear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib. 

## Current capabilities:
1. A 1D grid system for field evolution over time.
2. Finite-difference spatial operators.
3. Governing evolution laws, currently including advection and wave propagation.
4. Integration methods, currently including Euler and RK4.
5. Customizable boundary conditions.
6. Visualization.

---

## Visualization of wave propagation on a string (Sped up 4x):
<img width="900" height="500" alt="Video Project 6" src="https://github.com/user-attachments/assets/d2fd7b71-93f0-468a-ac58-b58ff810a6c8" />

---

## Module overview:
1. **grid.py**
* Describes the 1D grid and initial conditions.
  
2. **boundaries.py**
* Various boundary conditions to experiment with. Includes edge, constant, reflected, symmetric.

3. **operators.py**
* Finite-difference spatial operators. Includes the gradient, laplacian, and upwind.

4. **equations.py**
* Contains the governing physical equations. Currently includes advection and wave propagation.

5. **integrators.py**
* Contains the integrators used to evolve the grid state over time. Currently includes Euler and RK4.

6. **visualizations.py**
* Handles all visualization.

---

## Planned features:
1. Leapfrog integrator.
2. Diffusion equation implementation.
3. Stability analysis.
4. Coupled multi-field systems.
5. Shallow water equations.

---

**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs.
