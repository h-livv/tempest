# Tempest
A modular framework for simulating, validating, and learning PDE evolution operators.

---

## Wave Equation
<img width="1280" height="600" alt="wave" src="https://github.com/user-attachments/assets/84a809dc-0ec8-484e-bbde-0da33aac43fb" />

## Diffusion
<img width="1280" height="600" alt="diffusion" src="https://github.com/user-attachments/assets/55179cc2-cd2d-41d5-bd87-f2702a6a3791" />

## Linear advection
<img width="1280" height="600" alt="advection" src="https://github.com/user-attachments/assets/1c102fd2-15c2-4175-a05b-8aedca4f13a6" />

---
## Quick start

```bash
git clone https://github.com/h-livv/tempest.git
cd tempest
pip install -r requirements.txt
```
```bash
python main.py configs/1d/advection/simulation.py
```

### Direct API Usage (Simulation)

You can also bypass the pipeline and use the core simulation engine directly:

```python
from src.core.config import SimulationConfig
from src.core.simulation import Simulation
from src.physics.init_conditions import GaussianIC
from src.physics.equations import AdvectionEquation
from src.numerics.operators import upwind
from src.mesh.boundaries import periodic
from src.numerics.integrators import rk4

config = SimulationConfig(
    shape=(100,),
    spacing=(0.01,),
    dt=0.005,
    final_time=2.0,
    steps_per_frame=10,
    equation=AdvectionEquation(velocity=1.0),
    operator=upwind,
    boundary=periodic,
    integrator=rk4,
    initial_condition=GaussianIC(sigma=10.0, center_ratio=0.5)
)

results = Simulation(config).run()
```
---

## Validation & Convergence

Tempest includes an automated validation and convergence testing pipeline designed to rigorously verify physical fidelity and asymptotic grid convergence across hyperbolic, parabolic, and conservative PDE systems.

**Key Findings from the validation study**

* **The Diffusion-Dispersion Tradeoff (Advection):** Contrasts the severe artificial dissipation of upwind schemes against the dispersive nature of central differencing.
* **Spatial Error Dominance (Diffusion):** Under parabolic stability constraints, spatial truncation error overwhelmingly dominates. Computationally expensive higher-order time integrators (like RK4) offer no practical advantage over Forward Euler for explicitly integrated diffusion.
* **Hamiltonian Conservation (Wave Equation):** While standard RK4 introduces truncation-induced energy fluctuations, Tempest's symplectic Leapfrog implementation perfectly preserves the shadow Hamiltonian, maintaining total system energy.
* **Shock-Capturing Limitations (Shallow Water Equations):** Captures the fundamental breakdown of standard linear schemes in discontinuous regimes (e.g., dam breaks). Artificial viscosity in Lax-Friedrichs yields sub-first-order convergence, while Lax-Wendroff suffers from severe numerical dispersion and Gibbs oscillations in the presence of infinite gradients.
* **Limitations of boundary conditions in shock-based systems (Burgers' Equation):** Periodic boundary conditions encounter critical physics discrepancies with shock-based PDEs such as the Burgers' equation. The periodic expansion jump forms a rarefaction fan which alters the shock so it no longer represents the same physical problem. A Dirichlet boundary condition strictly holds the boundaries at the values the analytical domain requires.

The full formal methodology paper is available in [docs/validation_study_final.md](./docs/validation_study_final.md). 

Burgers' equation validation and convergence: [docs/burgers_validation.md](./docs/burgers_validation.md)

(Detailed numerical outputs, comparisons, and convergence CSVs are available in the `/outputs` directory).

---

## Neural Surrogate Model

Tempest includes an experimental machine learning pipeline for learning numerical evolution operators directly from simulated PDE trajectories.

Currently features an experimental ML surrogate model that learns to emulate the linear advection equation much faster than solving it numerically step-by-step.

Starting from a simple one-step predictor, the model was gradually optimized into a stable long-horizon transport surrogate through iterative testing and physics-informed constraints.

**Key highlights:**

* Lightweight 1D CNN architecture
* Stable autoregressive rollout over 10,000+ timesteps
* Strong shape and energy conservation
* Generalization to previously unseen initial conditions
* Translation-consistent wave transport
* Near-instant rollout generation after training

The full optimization process, experiments, and failure analysis are documented in:

* [docs/surrogate_setup.md](./docs/surrogate_setup.md)
* [docs/surrogate_evolution.md](./docs/advec_surrogate.md)

---

## Current capabilities:

Grid Infrastructure
- 1D and 2D supported structured grids
- Custom initial conditions (object-oriented, grid-native)
- Configurable boundary conditions

Numerical Methods
- **Integration**: Explicit Euler, Runge-Kutta 4 (RK4), Leapfrog, Lax-Friedrichs, Lax-Wendroff
- **Spatial Operators**: Upwind gradients, Central gradients, Laplacian

Physical Systems
- Linear advection
- Diffusion
- Wave propagation
- Shallow water equations
- Burgers' equation

Validation Diagnostics
- Energy tracking
- Stability monitoring
- Error analysis
- Convergence study

Machine Learning
- Lightweight model for quick training and outputs
- Ongoing development of generalized PDE evolution surrogates

---

## Module overview:

```text
Tempest/
│
├── configs/                  # Stored configurations for stable PDE runs
├── docs/                     # Formal mathematical documentation and studies
├── ml/                       # Code and outputs related to machine learning
├── outputs/                  # Structured CI/CT CSV outputs and validation data
├── src/                      # Core PDE Evolution Engine
│   ├── core/                 # Simulation clock, state management, and orchestration
│   ├── mesh/                 # Grid, Fields, and Boundary condition abstractions
│   ├── physics/              # Equations and InitialConditions encapsulated as objects
│   ├── numerics/             # Finite difference operators, Integrators, and Direct solvers
│   ├── validation/           # Analytical validation solutions
│   ├── visualization/        # Decoupled visualization/plotting architecture
│   └── diagnostics/          # Energy tracking and stability diagnostics
│
├── main.py                   # Automated data pipeline entry point
├── README.md                 # Project documentation and quick start guide
└── requirements.txt          # Minimal dependencies (numpy, pandas, matplotlib)
```
---

## Roadmap
### Near-term
1. Extend shock-capturing methods to 2D systems
2. Expand neural surrogate models to multi-dimensional PDEs
3. Investigate neural operators for generalized evolution learning

### Long-term ML research directions
- Generalized PDE evolution framework
- Learned numerical operators
- Physics-informed machine learning
- Reduced-order modeling
- Hybrid numerical–learned solvers

---
