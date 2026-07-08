# Tempest
### A modular framework for simulating, validating, and learning PDE evolution operators.<br>

Tempest is a modular framework for numerical simulation, validation, and scientific machine learning of time-dependent partial differential equations.

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

Detailed validation outputs, convergence studies, and numerical diagnostics are available in the `/outputs` directory.

---

## Features

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
- Foundation for generalized neural operators and Fourier Neural Operators (FNOs)

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

Current work investigates both convolutional and spectral surrogate models for the linear advection equation, serving as a stepping stone toward neural operator architectures such as Fourier Neural Operators (FNOs).

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

These investigations form the foundation for future work on neural operators, spectral PDE solvers, and learned scientific simulators within Tempest.

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
│   ├── physics/              # Physical models, governing equations, and initial conditions
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

### Scientific Machine Learning

- Multi-dimensional surrogates
- Neural operators
- Fourier operator learning
- Hybrid numerical–learned solvers

### Framework Development

- Additional PDE families
- Higher-dimensional validation
- GPU acceleration

---

## Long-term Vision

Tempest aims to become a unified framework for studying the numerical and learned evolution of physical systems governed by partial differential equations.

---
