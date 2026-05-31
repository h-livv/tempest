# Project Tempest
A modular computational framework for simulating and validating nonlinear dynamical systems using numerical PDE evolution methods. Implemented from scratch in Python using NumPy and Matplotlib.

## Current capabilities:
1. A 1D grid system for field evolution over time.
2. Finite-difference schemes such as Lax-Friedrichs.
3. Governing evolution laws, including advection, diffusion, wave propagation, and shallow water equations.
4. Integration methods, including Euler, RK4, and Leapfrog.
5. Customizable boundary conditions.
6. Energy graph for stability tracking.
7. Stable configurations to experiment with.
8. Validation against analytical solutions. Currently includes advection.

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

## Validation & Numerical Findings

### Linear Advection Validation

Project Tempest includes validation against analytical solutions where available. For the linear advection equation, a Gaussian pulse is compared against the exact periodically-shifted analytical solution throughout the simulation.

#### Key Observations

1. Initial validation exhibited large error spikes during boundary crossings.
2. Investigation revealed a mismatch between the analytical and numerical domains. The numerical solver used periodic boundary conditions, while the analytical solution assumed an infinite domain.
3. After enforcing the same periodicity in the analytical solution, the boundary spikes disappeared.
4. Remaining error was attributed primarily to numerical discretization effects.

#### Upwind vs Central Gradient

A comparison was performed between the upwind and central-gradient spatial operators.
<img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/bf938514-bdcb-42d8-99a9-9a60dee6a96f" />

| Property            | Upwind      | Central Gradient |
| ------------------- | ----------- | ---------------- |
| Energy Conservation | Lower       | Higher           |
| Shape Preservation  | Better      | Worse            |
| Numerical Behaviour | Dissipative | Dispersive       |
| L2 Error            | Lower       | Higher           |
| Relative Error      | Lower       | Higher           |

The central gradient operator preserved energy more effectively but introduced dispersive oscillations that distorted the pulse shape. The upwind operator introduced numerical diffusion and energy loss, but maintained a smoother pulse profile and achieved lower error metrics.

#### Numerical Conclusion

The experiments demonstrate a classical tradeoff in numerical PDE methods:

* **Central Gradient → Dispersive**
* **Upwind → Dissipative**

Furthermore, the study highlights that energy conservation alone is not a sufficient measure of numerical accuracy. A scheme may conserve energy while producing larger deviations from the analytical solution.

---

## Planned Features
1. Validation against analytical solutions for wave propagation, diffusion, and shallow water equations.
2. Higher-order conservative schemes (Lax-Wendroff, Rusanov).
3. Convergence and accuracy studies.
4. Generalized multi-field framework.
5. 2D structured grids.

---

**configurations.txt** includes stable configurations that can directly be implemented. <br>
**observations.txt** includes behaviour observed using different initial conditions, operators, equations, and integrators. <br>
**logs.txt** contains progress logs.
**Results** contains validation studies, numerical comparisons and outputs.
