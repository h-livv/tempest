# Tempest
### A computational physics laboratory for implementing and exploring numerical methods and scientific machine learning for partial differential equations.

Tempest provides a modular environment for implementing numerical methods, reproducing results from literature, validating algorithms against analytical solutions, and exploring scientific machine learning techniques.

---

## Gravity Wave Propagation
*Shallow Water Equations*

<img width="800" height="450" alt="shallow" src="https://github.com/user-attachments/assets/4eabee3e-07dc-462d-8a30-7ed781453bd7" />

*Simulation of free-surface gravity waves and nonlinear dam-break evolution.*

## Planetary Wave Dynamics
*Rossby Wave Equation*

<img width="800" height="450" alt="rossby" src="https://github.com/user-attachments/assets/858071c7-f7e5-4a2c-b85f-504fbadf9754" />

*Westward-propagating planetary waves induced by the β-effect.*

## Vortex Dynamics in Hurricane Eyewalls
*Barotropic Vorticity Equation*

<img width="800" height="450" alt="BVE" src="https://github.com/user-attachments/assets/fafc635c-1e9a-4431-9a3e-fa2267f5ad9a" />

*Idealized annular vortex instability producing coherent vortex structures relevant to hurricane eyewall dynamics.*

---

## Design Philosophy

Every physical model in Tempest is built from interchangeable components:

- Governing equations
- Numerical operators
- Time integrators
- Boundary conditions
- Initial conditions
- Sources
- Diagnostics

This modular design enables rapid experimentation with new physical systems while reusing a common numerical infrastructure.

---

## Validation Studies

Tempest includes an automated validation and convergence framework for verifying physical fidelity, stability, and asymptotic accuracy across implemented PDEs.

**Key numerical findings**

- **Advection:** Demonstrates the classical diffusion–dispersion tradeoff between upwind and central difference schemes.
- **Diffusion:** Confirms spatial error dominates under parabolic stability constraints, making higher-order time integrators unnecessary.
- **Wave Equation:** Leapfrog preserves the system's shadow Hamiltonian, while RK4 exhibits long-term energy drift.
- **Shallow Water Equations:** Investigates shock-capturing limitations of Lax–Friedrichs and Lax–Wendroff schemes.
- **Burgers' Equation:** Highlights the influence of boundary conditions on shock evolution and analytical agreement.

Further details are available in:

- [Validation Study](docs/validation_study_final.md)
- [Burgers Validation](docs/burgers_validation.md)

Detailed validation outputs, convergence studies, and numerical diagnostics are generated in the `/outputs` directory.

---

## Implemented methods

**Simulation infrastructure**
- Structured 1D and 2D grids
- Configurable boundary conditions
- Object-oriented initial conditions
- Automated experiment pipeline

**Implemented PDEs**
- Linear advection
- Diffusion
- Wave propagation
- Burgers' equation
- Shallow water equations
- Rossby wave equation
- Barotropic Vorticity Equation

**Time Integrators**
- Explicit Euler
- Runge–Kutta 4 (RK4)
- Leapfrog

**Finite Difference Schemes**
- Upwind
- Central Difference
- Lax–Friedrichs
- Lax–Wendroff
- Laplacian

**Validation & Diagnostics**
- Energy tracking
- Stability monitoring
- Error analysis
- Convergence study

**Scientific Machine Learning**
- Experimental CNN and spectral surrogate models for PDE evolution
- Simplified one-layer spectral model
- Basic FNO model

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

Tempest includes an experimental Scientific Machine Learning (SciML) framework for learning PDE evolution operators directly from high-fidelity numerical simulations.

Experimental investigation of neural surrogates for PDE evolution, including studies of autoregressive stability, spectral representations, and generalization.

**Current capabilities:**

- Lightweight 1D CNN surrogate with stable autoregressive rollout
- Experimental Fourier-domain surrogate operating directly on spectral coefficients
- Long-horizon prediction over 10,000+ timesteps
- Strong conservation of transported waveforms
- Generalization to unseen initial conditions and waveform combinations
- Systematic analysis of stability, numerical artifacts, and translation equivariance
- Near real-time autoregressive inference after training

The complete development process, experiments, and analyses are documented in:

- [CNN Surrogate](docs/CNN_surrogate.md)
- [Spectral Surrogate](docs/spectral_surrogate.md)

---

## Module overview:

```text
Tempest/
│
├── configs/                  # Stored configurations for stable PDE runs
├── docs/                     # Formal mathematical documentation and studies
├── ml/                       # Code and outputs related to machine learning
├── outputs/                  # Structured CI/CT CSV outputs and validation data
├── src/                      # Core Simulation Engine
│   ├── core/                 # Simulation clock, state management, and orchestration
│   ├── mesh/                 # Grid, Fields, and Boundary condition abstractions
│   ├── physics/              # Physical models, governing equations, initial conditions, and sources
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

## Research Directions

### Numerical Methods
- High-resolution finite-volume schemes
- Adaptive mesh refinement
- Conservative discretizations
- Structure-preserving integrators

### Scientific Machine Learning

- Neural operators
- Long-horizon stability
- Physics-informed architectures
- Hybrid numerical–learned solvers

### Physical Systems
- Turbulence
- Geophysical flows
- Electromagnetism
- Nonlinear wave dynamics

---

## Long-term Vision

Tempest serves as an experimental platform for computational physics, bringing together numerical simulation, validation, and scientific machine learning in a modular environment for exploring PDE-governed systems.

---
