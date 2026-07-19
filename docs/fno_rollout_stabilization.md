# FNO Long-Horizon Rollout Stabilization

This document describes the fixes implemented to stabilize autoregressive
inference of Tempest's 1D Fourier Neural Operator (FNO). It covers the
original failure, the evidence-based plan, and exactly what was changed in
code.

Related investigation write-up: [`fno_rollout_instability.md`](fno_rollout_instability.md).

---

## 1. Initial issue

### Observed behavior

The FNO learned an accurate **one-step** map \(u(t)\mapsto u(t+\Delta t)\), but
**long closed-loop rollouts** diverged:

| Horizon | Behavior |
|---------|----------|
| Step 1 | Relative L2 ≈ 0.02–0.03 (usable) |
| ~Step 50–100 | Error grows steadily |
| ~Step 150 | Noticeable phase drift / oscillations |
| ~Step 240 | Severe divergence; high-frequency noise in the field |

Teacher forcing (always feed ground truth) stayed accurate for the full
trajectory (mean Rel L2 ≈ 0.008 over ~250 steps). Autoregressive rollout on
the **same** model reached Rel L2 > 1 and eventually O(1)–O(10).

### What this ruled out

Controlled tests showed the failure was **not**:

- Broken FFT / `irfft` / complex einsum
- Incorrect mode truncation
- Train/inference normalization mismatch (there is no normalization)
- A rollout indexing bug (AR step 1 ≡ teacher-forced step 1)

### Diagnosed root causes

Ranked from the investigation:

1. **Exposure bias / horizon mismatch** — training supervised only short
   unrolls (initially 1-step, then 4-step); inference ran ~250 steps.
2. **Non-conservative iterated map** — linear advection conserves mass;
   AR mass drifted strongly while teacher forcing did not.
3. **Phase / shape error in retained modes** — late AR error energy lived
   mostly in \(k < n_\mathrm{modes}\), not unresolved high modes.
4. **Late high-frequency blow-up** — secondary symptom after the state left
   the training distribution (GELU + local branch regenerating high modes).

---

## 2. Planned actionable steps

| Priority | Action | Targets |
|----------|--------|---------|
| 1 | Curriculum unrolling (increasing \(S\)) | Exposure bias |
| 2 | Scheduled sampling (teacher-forcing probability decay) | Exposure bias |
| 3 | Soft mass penalty + optional hard mass projection | Conservation drift |
| 4 | Residual \(\Delta u\) prediction head | Phase / near-identity transport |
| 5 | High-mode energy penalty | Late spectral blow-up |
| 6 | Multi-IC training + temporal val split | Distribution shift |

Constraints:

- Do not redesign the FNO spectral layer.
- Do not treat “add more modes” as the first fix (TF was already accurate).
- Preserve the one-step operator structure; change **training objectives and
  time-step parameterization**.

---

## 3. What was implemented

### 3.1 Paper-aligned FNO block (prerequisite)

**File:** [`ml/layers/fno_block.py`](../ml/layers/fno_block.py)

The block was corrected from an incorrect outer residual

```text
σ(K(x) + W(x)) + x
```

to the original FNO update

```text
σ(K(x) + W(x))
```

This removes a full-bandwidth identity bypass that carried prediction errors
through every layer unchanged.

### 3.2 Spectral weight initialization

**File:** [`ml/layers/spectral_conv.py`](../ml/layers/spectral_conv.py)

Replaced `1/(C_in·C_out)` Gaussian scaling with complex Xavier

\[
\sigma = 1/\sqrt{2 C_\mathrm{in}}
\]

so the spectral branch starts at a similar scale to the local \(1\times1\)
convolution (previously ~100× weaker at init).

### 3.3 Unrolled multi-step loss

**Files:** [`ml/core/train.py`](../ml/core/train.py), [`ml/core/data.py`](../ml/core/data.py)

- `load_unrolled_training_data(..., unroll_steps=S)` builds windows  
  `input = u(t)`, `target = [u(t+1),…,u(t+S)]` with shape `(N, S, nx)`.
- `unrolled_loss` runs the model autoregressively for \(S\) steps and averages
  MSE against the teacher targets.

This directly trains the closed-loop map used at inference, instead of only
\(u_t\to u_{t+1}\).

### 3.4 Curriculum unrolling

**Files:** [`ml/core/train.py`](../ml/core/train.py) (`train_curriculum`),  
[`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py)

Stages are `(n_epochs, unroll_steps)` pairs. Loaders are rebuilt each stage
so targets match the current horizon. Example progression:

```text
4 → 8 → 16 → 32 → (optionally 64, 128)
```

**How it helps:** the optimizer first learns a stable short closed loop, then
extends the horizon so the model sees the error states that appear in long
rollouts.

### 3.5 Scheduled sampling

**File:** [`ml/core/train.py`](../ml/core/train.py)

During training unrolls, with probability \(p\) the next input is the
**ground-truth** target; otherwise it is the **prediction**.

\(p\) decays linearly from `TEACHER_FORCING_START` (default 1.0) to
`TEACHER_FORCING_END` (default 0.0) over the full curriculum.

Validation always uses \(p=0\) (fully autoregressive loss).

**How it helps:** early in training the model is not forced to recover from
its own large errors; later it must, matching inference.

### 3.6 Mass conservation terms

**File:** [`ml/core/train.py`](../ml/core/train.py)

1. **Soft penalty** — for each predicted step,

   \[
   \mathcal{L}_\mathrm{mass} = \lambda_m \,\mathrm{MSE}\!\left(\Delta x\sum_i \hat u_i,\;
   \Delta x\sum_i u_i\right)
   \]

   with default \(\lambda_m = 10^{-2}\).

2. **Hard projection** (optional, default on) — after each model step, add a
   spatially uniform correction so \(\int \hat u\,\mathrm{d}x\) matches the
   mass of the step’s input state (`project_mass`).

Also applied in [`ml/core/eval.py`](../ml/core/eval.py) `rollout(..., mass_project=True)`
so inference respects the same invariant.

**How it helps:** prevents the AR drift of \(\int u\,\mathrm{d}x\) that
compounded under iteration for advection.

### 3.7 Residual time-step head

**File:** [`ml/models/fno.py`](../ml/models/fno.py)

With `residual=True` (default in the FNO runner):

```text
û(t+Δt) = u(t) + FNO_body(u(t))
```

**How it helps:** biases the learned map toward a small transport increment
rather than an arbitrary full-field rewrite, reducing phase/amplitude
wandering under long iteration.

### 3.8 High-mode energy penalty

**File:** [`ml/core/train.py`](../ml/core/train.py)

Adds \(\lambda_h \,\mathrm{mean}|û(k)|^2\) for \(k \ge n_\mathrm{modes}\)
(default \(\lambda_h = 10^{-4}\)).

**How it helps:** discourages the late GELU-driven high-frequency growth seen
after the trajectory left distribution.

### 3.9 Data / evaluation hygiene

| Change | Location | Role |
|--------|----------|------|
| Temporal train/val split | `build_train_val_loaders(..., split="temporal")` | Avoid leaking adjacent times into val |
| Multi-IC training | `TRAIN_ICS = [gaussian, square, sine_wave]` | Broader support for shapes |
| Fixed-horizon Rel L2 logging | `evaluate_rollout_horizons` | Catch AR degradation during sweeps |

---

## 4. How these actions solve the issue

```text
Short train horizon ──► Curriculum + longer unrolls
        │
        ▼
Never sees own errors ──► Scheduled sampling (TF → AR)
        │
        ▼
Mass drifts under AR ──► Mass MSE + mass projection
        │
        ▼
Phase/shape wander   ──► Residual Δu head + paper block
        │
        ▼
Late high-freq noise ──► High-mode penalty (secondary)
```

Together, training now optimizes a **closed-loop**, **mass-aware**,
**near-transport** operator over horizons that approach inference length,
instead of a one-step map that only looks good under teacher forcing.

---

## 5. Configuration (FNO runner)

Primary knobs in [`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py):

| Symbol | Purpose |
|--------|---------|
| `UNROLL_CURRICULUM` | `(epochs, unroll_steps)` stages |
| `TEACHER_FORCING_START` / `END` | Scheduled sampling schedule |
| `MASS_WEIGHT` / `MASS_PROJECT` | Soft / hard mass constraints |
| `HIGH_MODE_WEIGHT` | Unresolved-mode penalty |
| `RESIDUAL` | \(\Delta u\) head |
| `TRAIN_ICS` / `TEST_ICS` | Initial conditions |

CLI: `--no-residual`, `--no-mass-project`.

---

## 6. What was intentionally not changed

- Spectral convolution mathematics (`rfft` → truncated complex multiply → `irfft`)
- Loss family (still MSE in physical space, plus small auxiliary terms)
- Detaching rollout states or truncating BPTT (would change the objective)

Memory/OOM work for long unrolls is documented separately in
[`fno_curriculum_memory.md`](fno_curriculum_memory.md).
