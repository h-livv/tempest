# Tempest
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

These studies validate the numerical accuracy of the simulation and verify that the solutions converge.

Observed that the slope of:

**log(error) vs log(dx)**

approaches the theoretical order of convergence, confirming correct implementation and expected behaviour.

Detailed convergence studies available in /Results and /docs

---

## Module overview:
1. **main.py**
* Consists of the main data pipeline in which multiple combinations of parameters can be passed into.

2. **direct_solver.py**
* Directly view the visualization without any diagnostic data generated.

3. **solver.py**
* The main PDE evolution engine.
  
4. **boundaries.py**
* Various boundary conditions to experiment with. Includes edge, constant, periodic and reflected.

5. **operators.py**
* Finite-difference spatial operators. Includes the gradient, laplacian, upwind.

6. **equations.py**
* Contains the governing physical equations. Includes advection, diffusion, wave propagation, and shallow water equations.

7. **integrators.py**
* Contains the integrators used to evolve the grid state over time. Includes Euler, RK4, Leapfrog, and Lax-Friedrichs.

8. **init_conditions.py**
* Contains multiple intial conditions to experiment with. Both uniform and irregular.

9. **visualization.py**
* Handles all visualization.

10. **stability.py**
* Computes diagnostics such as energy, conservation parameters, and numerical stability indicators.

11. **validation.py**
* Code for comparison with analytical solutions.

---

## Roadmap
### Near-term
1. Validation and convergence study for wave propagation and diffusion
2. Multi-field generalization
3. Higher-order and conservative numerical schemes
4. Addition of PDEs such as Burgers' equation
5. Detailed validation and convergence
6. Proceeding towards 2D expansion or ML research

### Long-term ML research directions
- Neural PDE surrogates
- Physics-informed neural networks
- Neural operators
- Hybrid numerical-learned solvers

---

**docs** consists of details analysis of studies carried out on validation and convergence. <br>
**Results** contains validation studies, numerical comparisons and outputs. <br>
**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs. <br?
**configurations.txt** includes stable configurations that can directly be implemented.
