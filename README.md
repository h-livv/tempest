# Project Tempest
A modular computational framework for simulating nonlinear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib. 

## Current capabilities:
1. A 1D grid system for field evolution over time.
2. Finite-difference spatial operators.
3. Governing evolution laws, including advection, wave propagation, and diffusion.
4. Integration methods, including Euler, RK4, and Leapfrog.
5. Customizable boundary conditions.
7. Energy graph for stability tracking.

---

## Visualization of wave propagation:
<img width="800" height="400" alt="Figure 1 2026-05-26 16-44-40 - Trim" src="https://github.com/user-attachments/assets/0a8c2be2-bbe6-4165-9851-7b4c54830935" />

## Visualization of diffusion (can be thought of as heat transfer on a rod):
<img width="800" height="400" alt="Figure 1 2026-05-26 16-41-44 - Trim" src="https://github.com/user-attachments/assets/25ed40e4-c6c2-4476-b0d7-cd2bf81901ec" />

---

## Module overview:
1. **grid.py**
* Describes the 1D grid and initial conditions.
  
2. **boundaries.py**
* Various boundary conditions to experiment with. Includes edge, constant, reflected, symmetric.

3. **operators.py**
* Finite-difference spatial operators. Includes the gradient, laplacian, and upwind.

4. **equations.py**
* Contains the governing physical equations. Includes advection, wave propagation, and diffusion.

5. **integrators.py**
* Contains the integrators used to evolve the grid state over time. Includes Euler, RK4, and Leapfrog.

6. **visualizations.py**
* Handles all visualization.

---

## Planned features:
1. Detailed stability analysis.
2. Coupled multi-field systems.
3. Shallow water equations.

---

**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs.
