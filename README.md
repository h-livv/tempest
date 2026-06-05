# Project Tempest
A modular computational framework for simulating and validating non-linear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib.

---

## Visualization of wave propagation:
<img width="800" height="400" alt="wave" src="https://github.com/user-attachments/assets/0bed2341-91ac-4105-8c7a-5daf75434808" />

## Visualization of diffusion (can be thought of as heat transfer on a rod):
<img width="800" height="400" alt="diffusion" src="https://github.com/user-attachments/assets/0499a55e-ed35-4412-908b-2be0d51250c2" />

---

## Current capabilities:

Grid Infrastructure
- 1D structured grids
- Custom initial conditions
- Configurable boundary conditions

Numerical Methods
- Upwind gradients
- Central gradients
- Lax-Friedrichs scheme
- Euler, RK4 and Leapfrog integration

Physical Systems
- Linear advection
- Diffusion
- Wave propagation
- Shallow water equations

Diagnostics
- Energy tracking
- Stability monitoring
- Analytical validation tools
- Automated data extraction pipeline

---

## Validation & Numerical Findings

Implemented:
- Linear advection
- Wave propagation
- Diffusion

Key findings:
- Energy conservation alone is not a sufficient measure of numerical accuracy.
- Boundary consistency between analytical and numerical solutions is essential for meaningful validation.
- Upwind schemes exhibit numerical diffusion.
- Central schemes exhibit numerical dispersion.
- Numerical solutions converge toward analytical solutions under mesh refinement.

Detailed validation studies are available in /Results and /docs

---

## Convergence study

Implemented convergence study for:
- Advection
- Wave propagation

These studies validate the numerical accuracy of the simulation and verify that the solutions converge.

Observed that the slope of:

**log(error) vs log(dx)**

approaches the theoretical order of convergence, confirming correct implementation and expected behaviour.

Detailed convergence studies available in /Results and /docs

---

## Module overview:
1. **solver.py**
* The main PDE evolution engine.
  
2. **boundaries.py**
* Various boundary conditions to experiment with. Includes edge, constant, periodic and reflected.

3. **operators.py**
* Finite-difference spatial operators. Includes the gradient, laplacian, upwind.

4. **equations.py**
* Contains the governing physical equations. Includes advection, diffusion, wave propagation, and shallow water equations.

5. **integrators.py**
* Contains the integrators used to evolve the grid state over time. Includes Euler, RK4, Leapfrog, and Lax-Friedrichs.

6. **visualization.py**
* Handles all visualization.

7. **stability.py**
* Computes diagnostics such as energy, conservation parameters, and numerical stability indicators.

8. **validation.py**
* Code for comparison with analytical solutions.

---

## Roadmap
### Near-term
1. Validation against analytical solutions for wave propagation, diffusion, and shallow water equations.
2. Higher-order and conservative numerical schemes.
3. Convergence study.
4. Generalized multi-field framework.
5. 2D structured grids.

### Long-term ML research directions
- Neural PDE surrogates
- Physics-informed neural networks
- Neural operators
- Hybrid numerical-learned solvers

---

**configurations.txt** includes stable configurations that can directly be implemented. <br>
**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs.<br>
**Results** contains validation studies, numerical comparisons and outputs.
