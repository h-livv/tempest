# FNO Curriculum Memory (OOM) Mitigations

This document describes the memory optimizations added so Tempest’s FNO
**curriculum unrolling** can train long autoregressive horizons on modest GPUs
(e.g. 6 GB RTX 4050 Laptop) without changing the learning objective.

Stabilization of rollout *accuracy* is covered in
[`fno_rollout_stabilization.md`](fno_rollout_stabilization.md). This file covers
only **VRAM / OOM**.

---

## 1. Initial issue

### Symptom

Curriculum training progressed through short horizons, then failed with
**CUDA out-of-memory** once the unroll length reached **64 steps**.

Typical failing progression before the fix:

```text
4 → 16 → 64 → 128
```

OOM occurred during **backpropagation through the 64-step unrolled graph**,
not during a single forward step.

### Why memory scales so hard

Unrolled training builds a computational graph of length \(S\) (rollout
steps). Peak activation memory scales roughly as

\[
\mathrm{VRAM} \;\propto\; B \times S \times N \times C \times L
\]

where \(B\) = batch size, \(S\) = unroll steps, \(N\) = spatial points
(\(\sim 2000\)), \(C\) = channel width (32), \(L\) = FNO depth (4).

Holding \(B\) fixed while increasing \(S\) therefore increases activation
storage **linearly in \(S\)**. At \(S=64\) with a large fixed batch, the
backward graph exceeds a 6 GB device.

### Constraints

Required:

- Preserve the mathematical training loss and gradients (no detach, no truncated BPTT).
- Do not shrink width, mode count, or redesign the architecture to “fix” OOM.
- Keep scheduled sampling and auxiliary losses intact.

---

## 2. Planned actionable steps

| Priority | Action | Intent |
|----------|--------|--------|
| 1 | Adaptive batch size vs unroll | Keep \(B\cdot S\) roughly constant |
| 2 | Automatic mixed precision (AMP) | Shrink real-valued activation footprints |
| 3 | Gradient checkpointing on FNO blocks | Trade compute for activation storage |
| 4 | Smoother curriculum (avoid jump 16→64) | Reach long horizons with smaller steps |
| 5 | Per-stage memory diagnostics | Verify each stage fits before continuing |

Explicitly **not** planned: detaching states between steps, gradient
truncation, reducing Fourier modes, or changing the loss.

---

## 3. What was implemented

### 3.1 Adaptive batch size

**Files:** [`ml/core/memory.py`](../ml/core/memory.py),  
[`ml/core/train.py`](../ml/core/train.py) (`train_curriculum`),  
[`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py)

`batch_size_for_unroll(unroll_steps, base_batch_size, schedule=..., ref_unroll=4)`:

1. If `schedule` has an exact key → use it.
2. Else use the nearest **lower** schedule key.
3. Else inverse scaling: \(\mathrm{batch} \approx \mathrm{base}\cdot\mathrm{ref}/S\), floored at 1.

Default schedule in the FNO runner:

```python
BATCH_SIZE_SCHEDULE = {
    4: 16,
    8: 8,
    16: 8,
    32: 4,
    64: 2,
    128: 1,
}
BASE_BATCH_SIZE = 16
BATCH_REF_UNROLL = 4
```

Each curriculum stage rebuilds dataloaders with the stage’s batch size via
`build_loaders(unroll_steps, batch_size)`.

**How it solves OOM:** activation memory tracks \(B\cdot S\). Dropping \(B\)
as \(S\) grows is the largest single lever and does not change the per-sample
loss.

### 3.2 Automatic mixed precision (AMP)

**File:** [`ml/core/train.py`](../ml/core/train.py)

Training and validation wrap the loss in:

```python
with torch.amp.autocast("cuda", enabled=amp_enabled):
    loss = unrolled_loss(...)
```

AMP is **disabled on CPU**. Toggle with `use_amp` / `--no-amp`.

**Spectral path exception** ([`ml/layers/spectral_conv.py`](../ml/layers/spectral_conv.py)):

Complex FFT / einsum is not implemented for `ComplexHalf`. The spectral
convolution therefore:

1. Disables autocast locally.
2. Casts inputs to float32.
3. Runs `rfft` → complex multiply → `irfft` in float32.
4. Casts the real output back to the incoming dtype.

**GradScaler is not used.** CUDA GradScaler cannot unscale `ComplexFloat`
gradients on the Fourier weights. Using scaler raised
`NotImplementedError` on complex parameters. Autocast alone still reduces
memory for real-valued lift / local conv / GELU / project activations.

**How it solves OOM:** stores many intermediate real activations in FP16
during the long unroll, without changing the spectral operator’s FP32 math.

### 3.3 Gradient checkpointing (FNO blocks only)

**File:** [`ml/models/fno.py`](../ml/models/fno.py)

```python
if self.checkpoint_blocks and self.training:
    h = checkpoint(block, h, use_reentrant=False)
else:
    h = block(h)
```

- Applies only to **FNOBlock** modules (spectral + local + GELU), not lift/project.
- Active only when `model.training` is True.
- `use_reentrant=False` is required for complex spectral weights.
- `train_curriculum(..., enable_checkpointing=True)` temporarily sets
  `model.checkpoint_blocks = True` for the curriculum, then restores the prior flag.

**How it solves OOM:** discards block activations after the forward pass and
recomputes them during backward, cutting stored activation depth through the
network at the cost of extra forward compute.

### 3.4 Smoother curriculum

**File:** [`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py)

Replaced abrupt `4 → 16 → 64 → 128` with a denser ladder, e.g.:

```text
4 → 8 → 16 → 32 → 64
```

(Exact `(epochs, unroll)` pairs are configurable; longer horizons including
128 remain supported via schedule entry `128: 1`.)

**How it helps OOM:** avoids jumping straight into a memory cliff; each stage
can be sized with a known batch. It does not by itself reduce peak VRAM at a
given \((B,S)\), but makes long horizons reachable incrementally.

### 3.5 Memory diagnostics

**Files:** [`ml/core/memory.py`](../ml/core/memory.py),  
[`ml/core/train.py`](../ml/core/train.py)

After each curriculum stage (when `report_memory=True`):

- Reset peak stats at stage start (`reset_peak_memory_stats`)
- Report allocated / reserved / peak allocated MiB
- Log batch size and unroll length

Disable with `--no-memory-report`.

**How it helps:** makes OOM regressions visible stage-by-stage instead of
failing opaquely mid-epoch.

---

## 4. Measured effect

Controlled micro-benchmark on CUDA (unroll \(S=64\), batch \(B=2\),
\(n_x=2000\), width 32, 4 layers, residual FNO):

| Configuration | Peak allocated |
|---------------|---------------:|
| Eager FP32 | ~429 MiB |
| Checkpoint FP32 | ~180 MiB (~2.4× less) |
| Checkpoint + AMP | ~103 MiB (~4.2× less vs eager) |

The former OOM stage fits with large margin on a 6 GB device under the default
schedule (`S=64`, `B=2`, checkpoint + AMP).

Checkpointing recomputes FNO blocks in backward; expect slower steps
(often tens of percent to ~2× depending on whether the run was memory-bound).

---

## 5. How the pieces interact

```text
Fixed B + growing S  ──► OOM at S≈64
        │
        ├─ Adaptive B(S)     → cut B·S product
        ├─ Checkpoint blocks → cut depth of stored activations
        ├─ AMP autocast      → cut bytes per real activation
        └─ Smoother curriculum → approach S=64 with known-safe (B,S)
```

None of these alter the unrolled MSE (+ mass / high-mode auxiliaries) or
detach the BPTT graph. Gradients remain full-horizon within each stage’s \(S\).

---

## 6. Configuration reference

| Symbol / flag | Role |
|---------------|------|
| `BATCH_SIZE_SCHEDULE` | Explicit `unroll → batch` map |
| `BASE_BATCH_SIZE` / `BATCH_REF_UNROLL` | Fallback inverse scaling |
| `USE_AMP` / `--no-amp` | Autocast on/off |
| `CHECKPOINT_BLOCKS` / `--no-checkpoint` | Block checkpointing on/off |
| `REPORT_MEMORY` / `--no-memory-report` | Stage VRAM logs |
| `UNROLL_CURRICULUM` | `(epochs, unroll_steps)` list |

### Recommended defaults by GPU size

| VRAM | Curriculum | Batch notes |
|------|------------|-------------|
| ~6 GB | Through 64 with schedule above | Peak ~100 MiB at S=64,B=2 (ckpt+AMP) |
| ~8 GB | May add unroll 128 at `batch=1` | Watch fragmentation |
| ≥12 GB | Include 128; optionally raise batches | e.g. 64→4, 128→2 |

---

## 7. Remaining bottlenecks

Even with these mitigations:

1. The unrolled graph still stores **per-step** field tensors \(B\times S\times N\).
2. Spectral weights and FFT math stay **complex FP32**.
3. Beyond ~128 steps, `batch=1` + checkpointing is typically required on 6–8 GB cards.
4. Fragmentation can still OOM even when peak-allocated looks safe — empty cache between stages if needed.

Further VRAM cuts that **would** change the objective (truncated BPTT, detach,
smaller models) are out of scope for this work.
