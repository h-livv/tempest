# Project Tempest
A modular computational framework for simulating nonlinear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib. 

## Current capabilities:
1. A 1D grid system for field evolution over time.
2. Finite-difference spatial operators.
3. Governing evolution laws, including advection, wave propagation, and diffusion.
4. Integration methods, including Euler, RK4, and Leapfrog.
5. Customizable boundary conditions.
7. Energy graph for stability tracking.
8. Stable configurations to experiment with.

---

## Visualization of wave propagation:
<img width="800" height="400" alt="wave" src="https://github.com/user-attachments/assets/0bed2341-91ac-4105-8c7a-5daf75434808" />

## Visualization of diffusion (can be thought of as heat transfer on a rod):
<img width="800" height="400" alt="diffusion" src="https://github.com/user-attachments/assets/0499a55e-ed35-4412-908b-2be0d51250c2" />

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
1. Implement shallow water equations in 1D.
2. Generalize multi-field systems.
3. Extend simulation to 2D.

---

**configurations.txt** includes stable configurations that can directly be implemented. <br>
**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs.
