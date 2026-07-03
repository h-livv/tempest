# Tempest
A modular framework for simulating, validating, and learning PDE evolution operators.<br>

Tempest combines numerical simulation, rigorous validation, and operator learning to study time-dependent physical systems governed by partial differential equations.

---

## Gravity Wave Propagation
*Shallow Water Equations*

<img width="1280" height="500" alt="shallow" src="https://github.com/user-attachments/assets/1ad0bf57-f510-482f-9b89-9f377fb27b77" />

## Nonlinear Shock Formation
*Burgers' Equation*

<img width="1280" height="500" alt="burgers" src="https://github.com/user-attachments/assets/e4171ddf-0994-47a6-8774-ccc0ac57a860" />

## Standing Waves
*Wave Equation*

<img width="1280" height="500" alt="wave" src="https://github.com/user-attachments/assets/c559920b-d273-43e8-b4fc-4e6cae927e0f" />

---

## Validation & Convergence

Tempest includes an automated validation and convergence testing pipeline designed to rigorously verify physical fidelity and asymptotic grid convergence across hyperbolic, parabolic, and conservative PDE systems.

**Numerical findings**

* **The Diffusion-Dispersion Tradeoff (Advection):** Contrasts the severe artificial dissipation of upwind schemes against the dispersive nature of central differencing.
* **Spatial Error Dominance (Diffusion):** Under parabolic stability constraints, spatial truncation error overwhelmingly dominates. Computationally expensive higher-order time integrators (like RK4) offer no practical advantage over Forward Euler for explicitly integrated diffusion.
* **Hamiltonian Conservation (Wave Equation):** While standard RK4 introduces truncation-induced energy fluctuations, Tempest's symplectic Leapfrog implementation perfectly preserves the shadow Hamiltonian, maintaining total system energy.
* **Shock-Capturing Limitations (Shallow Water Equations):** Captures the fundamental breakdown of standard linear schemes in discontinuous regimes (e.g., dam breaks). Artificial viscosity in Lax-Friedrichs yields sub-first-order convergence, while Lax-Wendroff suffers from severe numerical dispersion and Gibbs oscillations in the presence of infinite gradients.
* **Limitations of boundary conditions in shock-based systems (Burgers' Equation):** Periodic boundary conditions encounter critical physics discrepancies with shock-based PDEs such as the Burgers' equation. The periodic expansion jump forms a rarefaction fan which alters the shock so it no longer represents the same physical problem. A Dirichlet boundary condition strictly holds the boundaries at the values the analytical domain requires.

The full formal methodology paper is available in [docs/validation_study_final.md](./docs/validation_study_final.md)

Burgers' equation validation and convergence: [docs/burgers_validation.md](./docs/burgers_validation.md)

(Detailed numerical outputs, comparisons, and convergence CSVs are available in the `/outputs` directory).

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

## Scientific Machine Learning

Tempest includes an experimental machine learning pipeline for learning numerical evolution operators directly from simulated PDE trajectories.

Current work focuses on an autoregressive neural surrogate for the linear advection equation, developed as a first step toward generalized learned PDE evolution operators.

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

Simulation infrastructure
- Structured 1D and 2D grids
- Configurable boundary conditions
- Object-oriented initial conditions
- Automated experiment pipeline

Implemented PDEs
- Linear advection
- Diffusion
- Wave propagation
- Burgers' equation
- Shallow water equations

Numerical Methods
- **Integration**: Explicit Euler, Runge-Kutta 4 (RK4), Leapfrog, Lax-Friedrichs, Lax-Wendroff
- **Spatial Operators**: Upwind gradients, Central gradients, Laplacian

Validation & Diagnostics
- Energy tracking
- Stability monitoring
- Error analysis
- Convergence study

Scientific Machine Learning
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

### Scientific Applications

**Atmospheric Dynamics**
- Gravity waves
- Rossby waves

**Fluid Instabilities**
- Kelvin–Helmholtz
- Rayleigh–Taylor

**Geophysical Flows**
- Tsunami propagation

### Framework Development

- Additional PDE families
- Higher-dimensional validation
- GPU acceleration

### Scientific Machine Learning

- Multi-dimensional surrogates
- Neural operators
- Fourier operator learning
- Hybrid numerical–learned solvers

---

## Long-term Vision

Tempest aims to become a unified framework for studying the numerical and learned evolution of physical systems governed by partial differential equations.

---
