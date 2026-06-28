# Solidifying Idea #1: From Validation Studies to a Methods Paper

This document outlines how to branch from planned validation and convergence studies (diffusion, advection, wave) toward a methods paper on dissipation, dispersion, and diagnostic ambiguity in 1D finite-difference solvers.

---

## The Thesis You're Building Toward

Idea #1 is not three separate validation reports. The paper claim is something like:

> **For 1D FD method-of-lines solvers, energy conservation and L2 error against analytical solutions measure different things. Dissipative schemes can beat dispersive ones on L2 while losing energy; the "best" operator depends on whether you care about shape fidelity, phase accuracy, or conserved quantities.**

The advection evidence already exists in `docs/advection_validation_results.md`. Diffusion and wave complete the picture across three PDE classes:

| Equation | Dominant Numerical Pathology | Physical Dissipation? |
|---|---|---|
| Advection (1st order hyperbolic) | Dissipation *or* dispersion (operator choice) | No |
| Wave (2nd order hyperbolic) | Dispersion + phase error | No (energy should be conserved) |
| Diffusion (2nd order parabolic) | Numerical diffusion *on top of* real diffusion | Yes |

Diffusion is the **control case**: error should shrink cleanly under refinement; energy *should* decay. That contrast makes the advection/wave tradeoffs sharper.

---

## Phase 1: Finish Baseline Studies (Current Plan)

Treat each equation like `configs/advec_conv.py` + `docs/advection_convergence_study.md`. Same template every time.

### Minimum Viable Dataset Per Equation

For each of advection, wave, diffusion:

1. **≥4 grid refinements** with fixed `dt/dx` (CFL), domain length, `FINAL_TIME`, IC
2. **Convergence plot** + empirical order `p` from log–log regression (pipeline already does this)
3. **Transient error plot** (`validation_transient_errors.png`)
4. **One written doc** mirroring `docs/advection_convergence_study.md` (methodology, tables, interpretation)

### Config Gaps to Close

Right now only advection has a proper convergence sweep in `configs/advec_conv.py`:

```python
grid_configs = [
    {"N": 250, "dx": 1.0, "dt": 0.1},
    {"N": 500, "dx": 0.5, "dt": 0.05},
    ...
]
```

`configs/wave.py` and `configs/diffusion.py` each have a **single grid**. Duplicate the `advec_conv.py` refinement pattern for both.

### Expected Orders (Sanity Checks)

| Equation | Operator | Expected Spatial Order |
|---|---|---|
| Advection + upwind | 1st | `p ≈ 1` (measured ~1.07) |
| Advection + gradient | 2nd | `p ≈ 2` (run in Phase 2) |
| Wave + laplacian + RK4 | 2nd | `p ≈ 2` spatial; watch temporal error separately |
| Diffusion + laplacian + RK4 | 2nd | `p ≈ 2` |

If measured `p` doesn't match theory, that's either a paper subsection (coupled spatial–temporal error) or a bug — both are useful.

### What to Hold Constant (Document Explicitly)

Replicate for wave and diffusion what the advection doc already does:

- Domain length `L = N·dx` fixed across refinements
- `dt/dx` fixed
- Same IC shape (Gaussian for all three: `advec_gauss`, `wave_gauss`, `diff_gauss`)
- BC documented and matched in analytical solution (periodic for advection; reflect or periodic for wave depending on study)

**Phase 1 deliverable:** three convergence docs + `master_metrics.csv` rows you can cite. That's the foundation, not the paper yet.

---

## Phase 2: The Branch — Cross-Scheme Sweeps (Where Idea #1 Starts)

Baseline convergence confirms the pipeline. The **methods contribution** needs **paired comparisons on the same problem**.

### Advection (Core of the Paper)

Run the **same** `advec_conv.py` grids twice:

| Sweep | Operator | BC | Question |
|---|---|---|---|
| A | `upwind` | periodic | dissipative baseline |
| B | `gradient` | periodic | dispersive baseline |

Same IC, same `FINAL_TIME`, same RK4. Make the existing upwind-vs-gradient observation **quantitative across 4 grids**:

- Does the L2 gap **grow or shrink** under refinement?
- Does energy loss scale like `O(dx)` (consistent with numerical viscosity)?

### Wave (Dispersion Without Artificial Dissipation)

Single operator (`laplacian`), but vary:

- **Integrator:** RK4 vs `leapfrog` (temporal structure preservation)
- **BC:** `reflect` vs `periodic` (observations note edge ripples)
- **Duration:** short run vs `t > 1000` (long-time drift from `Experiments/observations.txt`)

Wave doesn't have an upwind/gradient fork, but it gives the **dispersion + phase error** side of the story. Key extra metrics (see Phase 3): peak location error, zero-crossing count, energy drift.

### Diffusion (Control)

Single operator is fine. Optionally add **Crank–Nicolson** later for stiff-`dt` contrast. For idea #1, diffusion mainly shows: when physics is dissipative, L2 and energy decay **align** — unlike advection.

### Phase 2 Deliverable

A comparison table, not just per-equation convergence:

| Scheme | Eq | Final L2 | Mean L2 | Energy loss | Observed p |
|---|---|---|---|---|---|
| upwind | advection | ... | ... | ... | ~1 |
| gradient | advection | ... | ... | ~0 | ~2 |
| RK4 | wave | ... | ... | small drift | ~2 |
| RK4 | diffusion | ... | ... | physical decay | ~2 |

That table is the seed of the Results section.

---

## Phase 3: Metrics Beyond L2 (Turns Data Into a Methods Argument)

The pipeline already logs L1, L2, mean/median, peak max error, and energy. For idea #1, add **derived metrics** when analyzing `time_history.csv` and `spatial_data.npz`:

| Metric | What It Captures | Why It Matters for the Paper |
|---|---|---|
| **Energy drift** `ΔE/E₀` | Dissipation | Shows gradient "wins" energy but can lose on L2 |
| **Peak location error** | Phase / transport | Advection: upwind smears but tracks reasonably; gradient disperses |
| **Peak amplitude error** | Dissipation | Separates shape smearing from phase slip |
| **Spectral content** (FFT of `u`) | Dispersion | Quantifies ripples from central differences |
| **Zero-crossing count** | Oscillation proxy | Wave + gradient-like behaviour at boundaries |

Most of this needs no solver changes — a small post-processing script on saved outputs is enough.

### The Killer Figure for Idea #1

A 2×2 or 3-panel plot:

1. **L2 error vs time** — upwind vs gradient (advection)
2. **Total energy vs time** — same runs
3. **Snapshot at fixed t** — numerical vs analytical overlay showing dispersive ripples vs dissipative smearing

Panels 1–2 come from `TempestPlotter.plot_validation`. Panel 3 comes from `spatial_data.npz`.

**The punchline figure:** L2 and energy **anti-correlate** for advection. For diffusion they **correlate**. Wave sits in between (energy nearly conserved, L2 grows from phase/dispersion).

---

## Phase 4: Connect Data to Theory (Lightweight, High Credibility)

Short analytical sections that **predict** what was measured:

### Modified Equation Analysis (Advection)

Show that upwind on `u_t + c u_x = 0` acts like:

```
u_t + c u_x = ν_num u_xx
```

with `ν_num ∝ c·dx`. Then check: does measured energy decay scale with `dx` the way `ν_num` predicts?

### Dispersion Relation (Wave + Central Laplacian)

For mode `e^{i(kx - ωt)}`, derive numerical `ω(k)` vs exact `ω = ck`. Plot **phase speed error** vs `k·dx`. Explains boundary ripples and long-time phase drift without hand-waving.

### Link to Observed Convergence Order

Measured `p ≈ 1.07` for upwind isn't just "pipeline works" — it's evidence the **dominant error is dissipative (1st order)**, not dispersion. For gradient advection, if you get `p ≈ 2` but higher L2, that's the paper's central tension.

**Phase 4 deliverable:** 2–3 short derivations in an appendix that connect slopes and energy curves to named numerical effects.

---

## Phase 5: Shape the Methods Paper

### Working Title Options

- *"Dissipation, Dispersion, and Diagnostic Ambiguity in 1D Finite-Difference PDE Solvers"*
- *"When Energy Conservation Misleads: Accuracy Metrics for Method-of-Lines Advection and Wave Propagation"*

### Outline Mapped to Data

1. **Introduction** — scheme selection often uses stability or conservation; less often L2 vs shape fidelity
2. **Method** — TEMPEST architecture; analytical validation; convergence protocol (cite three convergence docs)
3. **Results §1 — Advection** — upwind vs gradient; convergence orders; L2 vs energy anti-correlation
4. **Results §2 — Wave** — dispersion-dominated errors; BC sensitivity; long-time drift
5. **Results §3 — Diffusion** — control case where L2 and dissipation align
6. **Discussion** — decision guide: pick upwind for smooth-pulse L2; gradient if energy matters; neither is universally "better"
7. **Appendix** — modified equation; dispersion relation; full parameter tables from `master_metrics.csv`

### What Makes It a Methods Paper, Not a Code Report

| Code Report | Methods Paper |
|---|---|
| "Tempest achieves 1st-order convergence" | "First-order dissipation can reduce L2 below 2nd-order dispersive schemes on smooth ICs" |
| Per-equation validation | Cross-metric comparison with theoretical explanation |
| Single operator per equation | Paired operator sweeps where physics allows |
| Pipeline description | Reproducible protocol others can follow |

### Venue Realism

- **Undergraduate/honours thesis chapter** — very achievable with Phase 1–3
- **Journal (e.g. *Am. J. Phys.*, *Eur. J. Phys.*, *SIAM URU*)** — needs Phase 4 theory + cleaner figures + broader literature review
- **arXiv technical note / preprint** — good target after Phase 2–3

---

## Concrete Workflow

```
Phase 1: Convergence sweeps (advection ✓, wave, diffusion)
    ↓
Phase 2: Paired advection sweep (upwind vs gradient)
         Wave long-time + BC sweep
         Diffusion control
    ↓
Phase 3: Post-process (energy, peak, FFT metrics)
    ↓
Phase 4: Modified equation + dispersion relation
    ↓
Phase 5: Paper draft
```

### This Week

1. Create `configs/wave_conv.py` and `configs/diffusion_conv.py` mirroring `advec_conv.py`
2. Run pipeline; write `docs/wave_convergence_study.md` and `docs/diffusion_convergence_study.md`
3. Create `configs/advec_operator_compare.py` — same grids, `[upwind, gradient]` in `operators_list`

### Next

4. Post-process `master_metrics.csv` into the comparison table
5. Build the L2-vs-energy figure for advection
6. Draft the 1-page "claim sheet" — one sentence thesis + 3 figures + 1 table

---

## One Sentence to Keep in Mind

**Phase 1 proves your instrument works. Phase 2 asks what the instrument reveals about numerical physics. The paper lives in Phase 2–4.**
