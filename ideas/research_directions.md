# TEMPEST — Potential Research Directions

This document captures research trajectories the TEMPEST project could evolve into, grounded in the existing composable architecture, validation pipeline, and experimental observations.

---

## 1. Numerical Analysis of Dissipation vs. Dispersion (Immediate, High Yield)

Existing advection and wave experiments already surfaced a central tradeoff:

- **Central gradient** → dispersive, better energy retention, worse shape fidelity
- **Upwind** → dissipative, smoother profiles, lower L2 error on smooth pulses

### Research Questions

| Question | What TEMPEST Already Provides |
|---|---|
| When does L2 error favor dissipation over dispersion? | Validation + transient error plots |
| How do errors scale under mesh refinement? | Convergence pipeline in `main.py` |
| Is energy conservation a misleading accuracy metric? | `stability.py` + documented upwind vs. gradient results |

### Concrete Extensions

- **Modified equation analysis** — derive effective PDEs (numerical viscosity/dispersion terms) for each operator and compare to measured error growth
- **Dispersion relations** — plot numerical phase speed vs. wavenumber for each scheme (explains ripple formation at boundaries)
- **Long-time drift** — wave instability after `t > 1000`; study accumulated phase/amplitude error vs. symplectic or energy-preserving integrators

### Potential Output

A methods paper or thesis chapter on *accuracy metrics beyond energy conservation in 1D hyperbolic/parabolic solvers*.

---

## 2. Higher-Resolution Shock Hydrodynamics (Shallow Water → Riemann Benchmarks)

Shallow water is where the physics becomes genuinely hard. The project has already moved from unstable RK4 to **Lax–Friedrichs**, with artificial diffusion identified as the cost. Dam-break data collection is flagged in `Experiments/experiments.txt`.

### Evolution Path

```
Lax-Friedrichs baseline
  → MacCormack / Lax-Wendroff
    → MUSCL / TVD limiters
      → Exact / approximate Riemann solvers
        → Dam-break benchmark suite
```

### Research Outputs

- Compare **mass, momentum, and energy budgets** across schemes (not just total energy)
- Benchmark against **analytical Riemann solutions** (dam break, symmetric collision — `shallow_dam`, `shallow_collision` ICs)
- Quantify **smearing vs. oscillation** near shocks — ties to the advection dissipation/dispersion story in a nonlinear setting
- Study **CFL sensitivity** near discontinuities (pipeline already sweeps `N`, `dx`, `dt`)

---

## 3. Boundary-Condition Science (Under-Explored, Mathematically Deep)

`boundaries.py` notes that `reflect` must be explored in detail mathematically. Wave validation already shows spikes during interference and at walls.

### Research Questions

- Which BCs are **well-posed** for each equation/operator pair?
- **Reflect vs. constant vs. edge** — when do ghost-cell implementations introduce spurious modes?
- **Standing waves and resonance** — reflective domains with tuned domain length / wavelength ratios (on `Experiments/experiments.txt` list)
- **Validation consistency** — the advection periodicity fix (numerical domain must match analytical domain) generalizes to a methodology for domain-consistent validation

### Deliverable

A systematic **BC × equation × operator** error matrix using the existing `itertools.product` config pattern.

---

## 4. Time-Integration and Structure-Preserving Methods

The architecture cleanly separates `integrators.py` from spatial operators, enabling structured comparison studies.

| System | Current Integrator | Research Extension |
|---|---|---|
| Wave | RK4, leapfrog | Symplectic integrators, Störmer–Verlet, energy drift over 10⁴ steps |
| Advection | RK4 | SSP-RK, IMEX for stiff diffusion limits |
| Shallow water | Lax (finite volume style) | Operator splitting: flux step + source step |
| Diffusion | RK4 | Implicit / Crank–Nicolson for stiff `dt` limits |

### Key Question

RK4 stabilizes wave but may not preserve structure long-term. A convergence study on **temporal** order (fix `dx`, refine `dt`) would complement existing spatial convergence work.

---

## 5. Coupled and Multi-Physics PDEs

The composable `equation(t, state, ...)` interface scales to coupled systems without architectural rewrites.

- **Advection–diffusion** `u_t + c u_x = ν u_xx` — Péclet number studies (when does upwind diffusion dominate physical diffusion?)
- **Reaction–diffusion** (Fisher, Allen–Cahn) — pattern formation; no analytical solution → motivates manufactured solutions
- **Burgers' equation** — shock formation from smooth ICs; bridge between linear advection and shallow water
- **Variable-coefficient wave** `u_tt = c(x)² u_xx` — standing modes, WKB limits

Each adds a new equation module but reuses the full validation/convergence pipeline.

---

## 6. Verification Methodology as the Research Product

The pipeline (`master_metrics.csv`, per-run archives, log–log convergence regression) is already a **reproducible experiment harness**. Elevate it into a research contribution:

- **Method of Manufactured Solutions (MMS)** for shallow water and diffusion where no closed-form benchmark exists
- **Grid Convergence Index (GCI)** following Roache — formalize empirical slope fits
- **Error metric taxonomy** — L2 vs. L1 vs. max vs. energy vs. shape momenta; when does each matter?
- **Automated stability screening** — CFL violation detection, blow-up classification across parameter sweeps

The advection convergence doc (`docs/advection_convergence_study.md`) is already halfway to a reproducible study template; generalizing it across all equations is a coherent program.

---

## 7. Dimension Extension: 1D → 2D Structured Grids

The ghost-cell + slicing pattern in `operators.py` generalizes to 2D with modest refactoring.

- **2D advection** on periodic domains (vortex filament deformation)
- **2D wave equation** on rectangles (Chladni-style standing patterns)
- **2D shallow water** — dam break on a slope, radial collapse

Larger engineering lift, but unlocks visualization-rich phenomena while keeping the same validation philosophy.

---

## 8. Data-Driven and Inverse Problems (Longer Horizon)

Once the sweep pipeline generates enough labeled data (`spatial_data.npz`, `time_history.csv`):

- Train **neural surrogates** (PINNs, FNOs) on TEMPEST runs and compare generalization vs. classical FD
- **Parameter identification** — infer `c`, `ν`, or bathymetry from noisy observations
- **Optimal scheme selection** — given `(equation, Peclet, CFL, smoothness of IC)`, predict which operator minimizes error

The composable architecture makes TEMPEST a **data generator** for ML baselines, with analytical truth built in.

---

## Suggested Sequencing

### Phase A — Finish the Numerical-Analysis Story (3–6 months)

Upwind vs. gradient vs. higher-order advection; wave long-time stability; formal convergence + dispersion analysis. Low new code, high insight.

### Phase B — Shallow-Water Shock Benchmarks (6–12 months)

MUSCL/TVD, dam-break suite, Riemann comparisons. Builds on the Lax baseline and existing ICs.

### Phase C — BC + Standing-Wave / Resonance Program

Mathematically rigorous reflective BCs; standing-wave experiments from `Experiments/experiments.txt`.

### Phase D — Coupled PDEs or 2D Extension

Pick based on whether the priority is *physics richness* (coupled) or *visualization/impact* (2D).

---

## What Makes a Trajectory "Strong" for This Project

The best directions share three properties the codebase already supports:

1. **Analytical or benchmark truth exists** (or can be manufactured) — validation is not hand-wavy
2. **Multiple schemes can be swapped via config** — systematic comparison, not one-off runs
3. **Error decomposition is possible** — dissipation, dispersion, BC artifacts, temporal drift

### Weaker Directions (Avoid Unless Validation Path Exists)

- Adding exotic physics with no validation path
- Chasing visual polish without measurable convergence/stability claims

---

## Priority Recommendations

1. **Formalize the dissipation–dispersion–accuracy triangle** — already rooted in observations and `docs/advection_validation_results.md`
2. **Shallow-water shock capturing with dam-break benchmarks** — builds on Lax–Friedrichs, existing ICs, and `Experiments/experiments.txt` goals
