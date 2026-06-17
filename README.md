# Tempest
A framework for simulating and validating non-linear dynamical systems using numerical PDE evolution methods.

---

## Shallow water equations - Dam break
<img width="2000" height="400" alt="shallow_dam" src="https://github.com/user-attachments/assets/af5b4136-dcfb-4e13-9077-32decffff339" />

## Diffusion - Heat transfer on a rod
<img width="2000" height="400" alt="diffusion" src="https://github.com/user-attachments/assets/825feab3-8dcf-4c4b-88e2-2863c8853a0c" />

## Wave propagation - Disturbance on a string
<img width="2000" height="400" alt="wave_prop" src="https://github.com/user-attachments/assets/f0800786-4009-43ce-a661-33874ac1b61d" />

---
## Quick start

```bash
git clone https://github.com/h-livv/tempest.git
cd tempest
pip install -r requirements.txt
```
Run the direct visualization (no diagnostics):
```
python direct_solver.py
```
Run the full automated data pipeline:
```
python main.py configs/{any available configuration}
```
---

## Validation & Convergence

Tempest includes an automated continuous testing pipeline to verify physical fidelity and asymptotic convergence.

**Key Findings from the Advection, Diffusion, and Wave Studies:**

* The slope of log(error) vs log(dx) perfectly approaches theoretical truncation limits, confirming implementation integrity.
* Energy conservation alone is not a sufficient measure of numerical accuracy; boundary consistency between analytical and numerical solutions is equally critical.
* Spatial operator choice dictates physical artifacts: Upwind schemes introduce numerical diffusion (smearing), while Central schemes introduce numerical dispersion (oscillations).

The full formal methodology paper is available in [docs/validation_study.md](./docs/validation_study.md)

(Detailed numerical outputs, comparisons, and convergence CSVs are available in the /pipeline_results).

---

## Neural Surrogate Model

Tempest features an experimental ML surrogate model that learns to emulate the linear advection equation much faster than solving it numerically step-by-step.

Starting from a simple one-step predictor, the model was gradually optimized into a stable long-horizon transport surrogate through iterative testing and physics-informed constraints.

**Key highlights:**

* Lightweight 1D CNN architecture
* Stable autoregressive rollout over 10,000+ timesteps
* Strong shape and energy conservation
* Generalization to previously unseen initial conditions
* Translation-consistent wave transport
* Near-instant rollout generation after training

The full optimization process, experiments, and failure analysis are documented in:

* [docs/surrogate_setup.md](https://github.com/h-livv/tempest/blob/main/docs/surrogate_setup.md)
* [docs/surrogate_evolution.md](https://github.com/h-livv/tempest/blob/main/docs/advec_surrogate.md)


---

## Current capabilities:

Grid Infrastructure
- 1D structured grids
- Custom initial conditions
- Configurable boundary conditions

Numerical Methods
- **Integration**: Explicit Euler, Runge-Kutta 4 (RK4), Leapfrog, Lax-Friedrichs
- **Spatial Operators**: Upwind gradients, Central gradients, Laplacian

Physical Systems
- Linear advection
- Diffusion
- Wave propagation
- Shallow water equations

Diagnostics
- Energy tracking
- Stability monitoring
- Automated data extraction pipeline

---

## Module overview:

```
Tempest/
│
├── configs/                  # Stored configurations for stable PDE runs
├── diagnostics/              # Analysis & Verification Tools (energy, stability)
├── docs/                     # Formal mathematical documentation and studies
├── pipeline_results/         # Automated CI/CT CSV outputs and validation data
├── src/                      # Core PDE Evolution Engine (equations, integrators)
├── visualizations/           # Decoupled matplotlib plotting architecture
│
├── main.py                   # Automated data pipeline entry point
├── README.md                 # Project documentation and quick start guide
└── requirements.txt          # Minimal dependencies (numpy, pandas, matplotlib)
```
---

## Roadmap
### Near-term
1. Higher-order and conservative numerical schemes
2. Addition of PDEs such as Burgers' equation
3. Detailed validation and convergence
4. Proceeding towards 2D expansion or ML research

### Long-term ML research directions
- Neural PDE surrogates
- Physics-informed neural networks
- Neural operators
- Hybrid numerical-learned solvers

---
