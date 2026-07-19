# FNO Long-Horizon Rollout Instability

Evidence-based diagnosis of autoregressive divergence in Tempest's 1D Fourier
Neural Operator. No architecture changes were made during the investigation;
recommendations below follow from measured failure modes.

**Experiment setup.** Advection equation, Gaussian IC. Representative model:
64 Fourier modes, width 32, 4 FNO blocks, 4-step unrolled MSE, 40 epochs,
temporal train/val split. Artifacts from the controlled run live under
`/tmp/tempest_fno_rollout_investigation/` (`results.json` and diagnostic PNGs).

---

## 1. Pipeline summary

### Training

| Piece | Behavior |
|-------|----------|
| Data | Contiguous windows from validation NPZ series (`load_unrolled_training_data`) |
| Normalization | **None** — raw `float32` fields |
| Objective | \(S\)-step autoregressive unrolled MSE (\(S=4\) in the diagnosed run) |
| Split | Temporal 80/20 within each IC trajectory |
| Loss | Physical-space MSE only |

For each window the trainer feeds the model its own predictions for \(S\) steps
and averages MSE against \(\{u(t+1),\ldots,u(t+S)\}\).

### Inference

| Piece | Behavior |
|-------|----------|
| Rollout | Fully autoregressive from `series[0]` for the full trajectory (~250 steps) |
| Code path | `ml.core.eval.rollout` |
| Normalization | None (matches training) |

**Critical mismatch:** train horizon \(S=4\) ≪ inference horizon \(T\approx 250\).

### Spectral layer (verified correct)

```
rfft(dim=-1) → retain n_modes → complex einsum (bix,iox→box) → irfft(n=N)
```

Inactive modes are zeroed inside `SpectralConv1d`. The block update is
\(\sigma(K(x)+W(x))\) (paper-aligned). High frequencies can still reappear
via the local \(1\times1\) branch and GELU nonlinearity.

---

## 2. Experimental results

### Teacher forcing vs autoregressive

Always-feed-GT (teacher forcing) vs closed-loop AR on the same trained model:

| Step | TF Rel L2 | AR Rel L2 | AR / TF |
|-----:|----------:|----------:|--------:|
| 1 | 0.0324 | 0.0324 | 1.0 |
| 20 | 0.0103 | 0.0242 | 2.3 |
| 50 | 0.0048 | 0.0628 | 13 |
| 100 | 0.0037 | 0.484 | 130 |
| 150 | 0.0067 | **4.79** | 710 |
| 240 | 0.0127 | **7.27** | 570 |

- Mean TF Rel L2 over the full horizon: **0.0078**
- Mean AR Rel L2: **3.02**
- Step 1 TF ≡ AR exactly → no rollout indexing bug

### Error growth

AR Rel L2 stays \(O(0.02\)–\(0.06)\) until ~step 50, then accelerates through
\(O(0.5)\) by step 100 and \(O(5)\) by step 150, then plateaus near 7. This is
**compounding closed-loop blow-up**, not a gentle linear drift of a stable map.

### Conservation (linear advection)

Mass \(M=\int u\,dx\) is conserved by the PDE (GT mass drift = 0).

| Quantity | Value |
|----------|------:|
| GT mass drift | 0.000 |
| TF mass drift | +0.010 |
| AR mass drift | **−0.569** |

GT \(\|u\|_2\) slowly decays (numerical dissipation). AR \(\|u\|_2\) tracks
briefly, then grows past 2 after divergence — the learned map is not
norm-stable under iteration.

### Spectral analysis

| Quantity | Early | Step 150 | Step 240 |
|----------|------:|---------:|---------:|
| GT energy fraction above 64 modes | 0.47% | ~0 | ~0 |
| AR energy fraction above 64 modes | — | \(5\times10^{-7}\) | 0.53% |
| AR **error** energy in retained modes | 52% | **≈100%** | 99.4% |
| AR **error** energy in high modes | 48% | ~0 | 0.56% |

High-mode energy in the AR prediction **collapses** through mid-rollout, then
explodes only **after** the field has already diverged (~step 140+).
Catastrophic error at step 150 lives in retained modes (phase/shape), not in
unresolved high frequencies.

### Error localization

| Step | Error near Gaussian peak | Error away from peak |
|-----:|-------------------------:|---------------------:|
| 1 | 64% | 36% |
| 50 | 27% | 73% |
| 150 | 0.4% | **99.6%** |
| 240 | 5% | 95% |

Early errors are peak-localized (phase/amplitude mismatch). Late errors fill
the domain — spurious oscillations away from the true pulse.

### Dataset characterization

- Equation: advection; train IC: gaussian only (diagnosed run)
- Series length: 251; \(n_x=2000\); \(\Delta t=0.02\); \(\Delta x=0.005\)
- Amplitude range: \([0,\approx 2.0]\)
- Normalization: none
- Inference horizon ≈ 62× longer than the unrolled training loss

### Fourier layer inspection

No implementation defects found: correct FFT axis, complex weights, mode
truncation, and einsum contraction. Prior unit tests cover round-trip,
identity weights, imaginary phase, and nonzero spectral gradients.

---

## 3. Root cause analysis

### 1. Exposure bias / horizon mismatch — confidence: **high**

**Evidence.** TF stays ≤ 0.013 Rel L2 for 250 steps; AR reaches 7.27. Training
only supervises 4-step rollouts. By step 20 AR is already 2× worse than TF;
by step 50 it is 13× worse.

**Mechanism.** The model never trains on the closed-loop states it sees at
long inference horizons, so small per-step errors compound until the input
leaves the training distribution.

### 2. Non-conservative iterated map — confidence: **high**

**Evidence.** GT mass is flat; AR mass drifts by −0.57. AR \(\|u\|_2\) grows
after ~step 100 while TF mass drift stays ~0.01.

**Mechanism.** Soft conservation errors of an otherwise accurate one-step map
accumulate under autoregression and amplify instability.

### 3. Phase / transport error in retained modes — confidence: **medium–high**

**Evidence.** At step 150, ≈100% of AR error energy is in \(k < 64\). Early
error is peak-localized; late error is domain-filling.

**Mechanism.** Progressive phase lag/lead and shape distortion in the learned
transport operator, not an initial high-frequency instability.

### 4. High-frequency spectral leakage — confidence: **low as root cause**

**Evidence.** High-mode energy collapses mid-rollout and explodes only after
divergence. GT has &lt; 0.5% energy above 64 modes.

**Mechanism.** Secondary GELU / local-branch regeneration once the state is
OOD — a *symptom*, not the initiator.

### Ruled out

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| FFT / irfft / einsum bug | Ruled out | Unit tests; TF accurate for 250 steps |
| Train/infer normalization mismatch | Ruled out | No normalization in either path |
| Rollout indexing bug | Ruled out | TF step 1 ≡ AR step 1 |
| Spectral truncation as primary cause | Low confidence | Retained-mode error dominates at blow-up |

---

## 4. Recommended fixes

Each recommendation targets a diagnosed failure mode. **Do not redesign the
FNO block or chase more Fourier modes as the first fix.**

| Fix | Targets | Why |
|-----|---------|-----|
| Curriculum unrolling (\(4\to16\to64\to128\)) | Exposure bias | Trains on closed-loop states seen at inference |
| Scheduled sampling | Exposure bias | Mixes GT and predicted inputs so the model learns to correct itself |
| Soft mass penalty (and optional mass projection) | Conservation drift | Directly constrains the invariant AR violates |
| Residual time-step head for advection | Phase accumulation | Predict \(\Delta u\) so the map stays near transport |
| High-mode energy penalty | Late high-freq blow-up | Suppresses secondary GELU noise after primary drift is controlled |
| Multi-IC training | Distribution shift | Reduces OOD amplitude/shape once errors accumulate |

### Implementation status

The following fixes are implemented in the FNO training stack
([`ml/core/train.py`](../ml/core/train.py), [`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py),
[`ml/models/fno.py`](../ml/models/fno.py)):

1. **Curriculum unrolling** — default `4 → 8 → 16 → 32 → 64` (smoother; 128 still configurable)
2. **Scheduled sampling** — teacher-forcing probability decays \(1\to 0\) over the curriculum
3. **Soft mass penalty** + **hard mass projection** after each step (`MASS_WEIGHT`, `MASS_PROJECT`)
4. **Residual \(\Delta u\) head** — `FNO1d(..., residual=True)`
5. **High-mode energy penalty** — `HIGH_MODE_WEIGHT` on \(k \ge n_\mathrm{modes}\)
6. **Multi-IC training** — `TRAIN_ICS = [gaussian, square, sine_wave]`
7. **Adaptive batch size** — `BATCH_SIZE_SCHEDULE` / inverse scaling (`ml/core/memory.py`)
8. **AMP autocast** — FP16 real-valued paths; spectral conv forced FP32 (no GradScaler: complex weights)
9. **Gradient checkpointing** — FNO blocks only, training-time, configurable
10. **Per-stage memory diagnostics** — allocated / reserved / peak MiB

Re-run: `python -m ml.experiments.fno.run`

### Memory-efficient curriculum (VRAM)

Long unrolls store activations for every step. On a 6 GB laptop GPU this OOMs
around unroll 64 with a fixed batch. Measured peak for `S=64`, `B=2`,
`nx=2000`, width 32, 4 layers:

| Config | Peak allocated |
|--------|---------------:|
| Eager FP32 | ~429 MiB |
| Checkpoint FP32 | ~180 MiB (~2.4×) |
| Checkpoint + AMP | ~103 MiB (~4.2× vs eager) |

**AMP note.** `GradScaler` is unused — CUDA cannot unscale `ComplexFloat`
Fourier gradients. Spectral convolution runs under disabled autocast (FP32);
lift / local / GELU / project may use FP16 activations.

| GPU VRAM | Recommended default | Notes |
|----------|---------------------|-------|
| ~6 GB | Curriculum through 64; batches `{4:16,…,64:2}` | Fits with large margin |
| ~8 GB | Add `(20, 128)` with `batch=1` | Watch fragmentation |
| ≥12 GB | Include 128; raise batches (e.g. 64→4, 128→2) | Faster steps |

Flags: `--no-amp`, `--no-checkpoint`, `--no-memory-report`, `--batch-size`.

### Code references

- Training loop: [`ml/core/train.py`](../ml/core/train.py)
- Memory helpers: [`ml/core/memory.py`](../ml/core/memory.py)
- Rollout: [`ml/core/eval.py`](../ml/core/eval.py)
- Unrolled windows: [`ml/core/data.py`](../ml/core/data.py)
- FNO runner: [`ml/experiments/fno/run.py`](../ml/experiments/fno/run.py)
- Spectral conv / block: [`ml/layers/spectral_conv.py`](../ml/layers/spectral_conv.py), [`ml/layers/fno_block.py`](../ml/layers/fno_block.py)
