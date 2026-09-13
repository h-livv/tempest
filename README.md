# Tempest

### A computational physics laboratory for studying **numerical and learned methods for partial differential equations**.

Tempest provides a common environment for implementing PDEs, comparing numerical
schemes, validating solutions, and experimentally studying learned
approximations to PDE evolution.

> **Status:** Archived

## What It Explores

The central question is:

> **How do different computational approximations reproduce the dynamics of
> physical systems described by PDEs?**

The repository experiments with both conventional numerical methods and
scientific machine-learning approaches.

## Example Simulations

### Shallow Water Equations

<img width="800" height="450"
alt="shallow"
src="https://github.com/user-attachments/assets/4eabee3e-07dc-462d-8a30-7ed781453bd7" />

Gravity-wave and dam-break dynamics.

### Rossby Wave Equation

<img width="800" height="450"
alt="rossby"
src="https://github.com/user-attachments/assets/858071c7-f7e5-4a2c-b85f-504fbadf9754" />

Planetary-wave propagation from the β-effect.

### Barotropic Vorticity Equation

<img width="800" height="450" alt="BVE" src="https://github.com/user-attachments/assets/fafc635c-1e9a-4431-9a3e-fa2267f5ad9a" />

Idealized vortex dynamics and coherent-structure formation.

## Implemented Methods

### Implemented physical models include:

* advection and diffusion
* wave and Burgers' equations
* shallow-water dynamics
* Rossby-wave dynamics
* barotropic vorticity

### Numerical methods include:

* Euler and RK4
* Leapfrog
* Upwind, central-difference, Lax-Friedrichs, and Lax-Wendroff schemes
* finite-difference spatial operators

## Learned Solvers

Tempest also contains experimental work on learned approximations to PDE
evolution, including:

* CNN surrogates
* spectral surrogates
* basic Fourier Neural Operators
* autoregressive long-horizon prediction
* generalization across unseen initial conditions

These experiments examine questions such as stability, spectral representation,
translation structure, and long-horizon error accumulation.

See [`CNN Surrogate`](docs/CNN_surrogate.md) and
[`Spectral Surrogate`](docs/spectral_surrogate.md).

## Validation

Numerical methods are compared against analytical solutions where available,
with supporting:

* convergence studies
* error analysis
* stability diagnostics
* energy diagnostics

See [`Validation Study`](docs/validation_study_final.md) and
[`Burgers Validation`](docs/burgers_validation.md).

## Architecture

Simulations are assembled from interchangeable:

```text
Governing equation
       │
       ▼
Spatial discretization
       │
       ▼
Time integrator
       │
       ▼
Boundary / initial conditions
       │
       ▼
Diagnostics
```

This makes different physical systems and computational methods directly
comparable within the same environment.

## Quick Start

```bash
git clone https://github.com/h-livv/tempest.git
cd tempest
pip install -r requirements.txt

python main.py configs/2d/advection/simulation.py
```
