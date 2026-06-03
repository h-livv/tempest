### Linear Advection Validation

Project Tempest includes validation against analytical solutions where available. For the linear advection equation, a Gaussian pulse is compared against the exact periodically-shifted analytical solution throughout the simulation.

#### Observations

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
