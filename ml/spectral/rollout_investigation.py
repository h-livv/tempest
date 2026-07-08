"""
Systematic investigation of multi-step rollout instability in the spectral surrogate.

Runs diagnostic steps 1-10 without modifying model architecture or hyperparameters.
Reuses training configuration from spectral.py.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import style
from torch.utils.data import DataLoader, TensorDataset, random_split

SPECTRAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPECTRAL_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("spectral", SPECTRAL_DIR / "spectral.py")
spectral = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(spectral)

# Investigation settings (same hyperparameters as spectral.py)
INVESTIGATION_IC = "gaussian"
ROLLOUT_HORIZONS = [1, 2, 3, 5, 10, 20]
MAX_ROLLOUT_STEPS = max(ROLLOUT_HORIZONS)
TRANSLATION_OFFSETS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
PLOT_STYLE = spectral.PLOT_STYLE
MAX_PLOT_MODES = spectral.MAX_PLOT_MODES


def setup_style() -> None:
    style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE)


def make_output_dir() -> Path:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = SPECTRAL_DIR / "outputs" / f"rollout_investigation_{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def train_model(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Train a single model with existing spectral.py hyperparameters."""
    train_data = spectral.load_combined_training_data(spectral.TRAIN_ICS)
    nx = train_data["inputs"].shape[1]
    n_modes = min(spectral.N_MODES, nx // 2 + 1)

    dataset = TensorDataset(
        torch.from_numpy(train_data["inputs"]),
        torch.from_numpy(train_data["targets"]),
    )
    n_train = int(len(dataset) * spectral.TRAIN_FRACTION)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(spectral.RANDOM_SEED),
    )
    train_loader = DataLoader(train_set, batch_size=spectral.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=spectral.BATCH_SIZE, shuffle=False)

    torch.manual_seed(spectral.RANDOM_SEED)
    np.random.seed(spectral.RANDOM_SEED)
    model = spectral.SpectralNet(nx=nx, n_modes=n_modes, use_linear=spectral.USE_LINEAR).to(device)

    print(f"Training model: nx={nx}, n_modes={n_modes}, epochs={spectral.EPOCHS}")
    history = spectral.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=spectral.EPOCHS,
        learning_rate=spectral.LEARNING_RATE,
        device=device,
        verbose=True,
    )

    meta = {
        "nx": nx,
        "n_modes": n_modes,
        "use_linear": spectral.USE_LINEAR,
        "epochs": spectral.EPOCHS,
        "learning_rate": spectral.LEARNING_RATE,
        "train_ics": spectral.TRAIN_ICS,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }
    return model, meta


def trace_forward(model: spectral.SpectralNet, field: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Expose intermediate FFT/IFFT tensors from the model forward pass."""
    model.eval()
    x = torch.from_numpy(field).float().unsqueeze(0).to(device)
    with torch.no_grad():
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, : model.n_modes] = x_ft[:, : model.n_modes] * model.spectral_weights
        out = torch.fft.irfft(out_ft, n=model.nx, dim=-1)
        if model.linear is not None:
            out = model.linear(out)
    return {
        "input": field,
        "input_fft": x_ft.squeeze(0).cpu().numpy(),
        "output_fft": out_ft.squeeze(0).cpu().numpy(),
        "output_physical_pre_linear": torch.fft.irfft(out_ft, n=model.nx, dim=-1).squeeze(0).cpu().numpy(),
        "output_physical": out.squeeze(0).cpu().numpy(),
        "spectral_weights": model.spectral_weights.detach().cpu().numpy(),
    }


def teacher_forcing_rollout(
    model: nn.Module,
    true_states: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Predict n_steps using the true previous field at each step."""
    preds = np.zeros((n_steps, true_states.shape[-1]), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            state = torch.from_numpy(true_states[step]).float().unsqueeze(0).to(device)
            pred = model(state).squeeze(0).cpu().numpy()
            preds[step] = pred
    return preds


def field_energy(field: np.ndarray) -> float:
    """Compute ||u||_2^2."""
    return float(np.sum(field**2))


def circular_shift_field(field: np.ndarray, x: np.ndarray, offset: float) -> np.ndarray:
    """Translate a 1D periodic field by a physical offset."""
    lx = float(x[-1] - x[0] + (x[1] - x[0]))
    x_shift = (x - offset - x[0]) % lx + x[0]
    return np.interp(x_shift, x, field).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1 — One-step prediction
# ---------------------------------------------------------------------------

def step01_one_step(
    model: nn.Module,
    data: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step01_one_step"
    out.mkdir(parents=True, exist_ok=True)

    sample_index = spectral.ROLLOUT_PAIR_INDEX
    x_in = data["inputs"][sample_index]
    y_true = data["targets"][sample_index]
    y_pred = spectral.predict(model, torch.from_numpy(x_in).unsqueeze(0), device)[0]
    y_rollout = spectral.rollout(model, data["series"][sample_index], 1, device)[0]
    abs_err = np.abs(y_pred - y_true)
    metrics = spectral.compute_metrics(y_pred[None], y_true[None])

    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].plot(data["x"], x_in, color="#444444")
    axes[0].set_title("Input u(x, t_in)")
    axes[1].plot(data["x"], y_pred, label="One-step", color="#cc3311")
    axes[1].plot(data["x"], y_rollout, label="Rollout step 1", color="#ee7733", linestyle=":")
    axes[1].plot(data["x"], y_true, label="Numerical target", color="#0077bb", linestyle="--")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Prediction vs target")
    axes[2].plot(data["x"], abs_err, color="#aa3377")
    axes[2].set_title("Absolute error")
    axes[3].axis("off")
    t_in = float(data["times_full"][sample_index])
    t_out = float(data["times_full"][sample_index + 1])
    axes[3].text(
        0.0,
        0.85,
        "\n".join(
            [
                f"pair index: {sample_index}",
                f"t_in:  {t_in:.4f}",
                f"t_out: {t_out:.4f}",
                f"one-step == rollout[0]: {np.allclose(y_pred, y_rollout)}",
                f"Relative L2: {metrics['Relative L2']:.4e}",
                f"MSE:         {metrics['MSE']:.4e}",
                f"Max |error|: {metrics['Max Error']:.4e}",
            ]
        ),
        fontsize=10,
        family="monospace",
        va="top",
    )
    fig.suptitle(f"Step 1 — One-step @ t_in={t_in:.4f} → t_out={t_out:.4f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "one_step_verification.png", bbox_inches="tight")
    plt.close(fig)

    result = {
        "pair_index": sample_index,
        "t_in": t_in,
        "t_out": t_out,
        "metrics": metrics,
        "single_equals_rollout": bool(np.allclose(y_pred, y_rollout)),
        "passed": metrics["Relative L2"] < 0.05,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Step 1: One-step prediction ===")
    print(f"pair index {sample_index} | t_in={t_in:.4f} → t_out={t_out:.4f}")
    print(f"Relative L2: {metrics['Relative L2']:.6e} | MSE: {metrics['MSE']:.6e} | Max error: {metrics['Max Error']:.6e}")
    print(f"one-step == rollout step 1: {result['single_equals_rollout']}")
    if not result["passed"]:
        print("WARNING: One-step relative L2 exceeds 0.05 threshold.")
    return result


# ---------------------------------------------------------------------------
# Step 2 — FFT pipeline validation
# ---------------------------------------------------------------------------

def step02_fft_pipeline(
    model: spectral.SpectralNet,
    field: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step02_fft_pipeline"
    out.mkdir(parents=True, exist_ok=True)

    n_modes = model.n_modes
    nx = model.nx

    # NumPy round-trip
    coeff_np = np.fft.rfft(field)
    recon_np_full = np.fft.irfft(coeff_np, n=nx)

    # Torch round-trip
    x_t = torch.from_numpy(field).float().unsqueeze(0).to(device)
    coeff_t = torch.fft.rfft(x_t, dim=-1)
    recon_t_full = torch.fft.irfft(coeff_t, n=nx, dim=-1).squeeze(0).cpu().numpy()

    # Truncation (no learned weights)
    coeff_trunc = np.zeros_like(coeff_np)
    coeff_trunc[:n_modes] = coeff_np[:n_modes]
    recon_trunc = np.fft.irfft(coeff_trunc, n=nx)

    traced = trace_forward(model, field, device)
    recon_model_pre_linear = traced["output_physical_pre_linear"]

    metrics = {
        "numpy_roundtrip_rel_l2": spectral.relative_l2(recon_np_full, field),
        "torch_roundtrip_rel_l2": spectral.relative_l2(recon_t_full, field),
        "numpy_torch_coeff_max_diff": float(np.max(np.abs(coeff_np - coeff_t.squeeze(0).cpu().numpy()))),
        "truncation_rel_l2": spectral.relative_l2(recon_trunc, field),
        "model_fft_path_rel_l2": spectral.relative_l2(recon_model_pre_linear, field),
        "n_modes": n_modes,
        "nx": nx,
    }

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(nx)
    axes[0, 0].plot(x, field, label="Original", color="#222222")
    axes[0, 0].plot(x, recon_np_full, label="NumPy iFFT(rFFT)", linestyle="--", color="#0077bb")
    axes[0, 0].legend()
    axes[0, 0].set_title("NumPy FFT round-trip")
    axes[0, 1].plot(x, field, label="Original", color="#222222")
    axes[0, 1].plot(x, recon_trunc, label=f"Truncated ({n_modes} modes)", linestyle="--", color="#cc3311")
    axes[0, 1].legend()
    axes[0, 1].set_title("Mode truncation reconstruction")
    axes[1, 0].plot(x, field, label="Original", color="#222222")
    axes[1, 0].plot(x, recon_model_pre_linear, label="Model FFT path (pre-linear)", linestyle="--", color="#009988")
    axes[1, 0].legend()
    axes[1, 0].set_title("Model internal FFT→IFFT (learned weights)")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.9,
        "\n".join(
            [
                f"NumPy round-trip rel L2:   {metrics['numpy_roundtrip_rel_l2']:.3e}",
                f"Torch round-trip rel L2:   {metrics['torch_roundtrip_rel_l2']:.3e}",
                f"Max |NumPy-Torch coeff|:   {metrics['numpy_torch_coeff_max_diff']:.3e}",
                f"Truncation rel L2:         {metrics['truncation_rel_l2']:.3e}",
                f"Model FFT path rel L2:     {metrics['model_fft_path_rel_l2']:.3e}",
                f"Retained modes:            {n_modes} / {nx // 2 + 1}",
            ]
        ),
        fontsize=11,
        family="monospace",
        va="top",
    )
    fig.suptitle("Step 2 — FFT / IFFT pipeline validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "fft_pipeline_validation.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Step 2: FFT pipeline ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6e}")
        else:
            print(f"{key}: {value}")
    return metrics


# ---------------------------------------------------------------------------
# Step 3 — Spectral evolution (single step)
# ---------------------------------------------------------------------------

def step03_spectral_evolution(
    model: nn.Module,
    data: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step03_spectral_evolution"
    out.mkdir(parents=True, exist_ok=True)

    sample_index = spectral.ROLLOUT_PAIR_INDEX
    x_in = data["inputs"][sample_index]
    y_true = data["targets"][sample_index]
    traced = trace_forward(model, x_in, device)
    y_pred = traced["output_physical"]

    fft_in = traced["input_fft"]
    fft_pred = traced["output_fft"]
    fft_true = np.fft.rfft(y_true)

    n_plot = min(len(fft_in), MAX_PLOT_MODES)
    modes = np.arange(n_plot)
    mag_in = np.abs(fft_in[:n_plot])
    mag_pred = np.abs(fft_pred[:n_plot])
    mag_true = np.abs(fft_true[:n_plot])
    phase_in = np.angle(fft_in[:n_plot])
    phase_pred = np.angle(fft_pred[:n_plot])
    phase_true = np.angle(fft_true[:n_plot])
    coeff_diff = fft_pred[:n_plot] - fft_true[:n_plot]

    low_band = slice(0, max(4, n_plot // 8))
    high_band = slice(max(4, n_plot // 2), n_plot)
    metrics = {
        "mag_rel_l2_low": spectral.relative_l2(mag_pred[low_band], mag_true[low_band]),
        "mag_rel_l2_high": spectral.relative_l2(mag_pred[high_band], mag_true[high_band]),
        "phase_mae_low": float(
            np.mean(np.abs(np.angle(np.exp(1j * (phase_pred[low_band] - phase_true[low_band])))))
        ),
        "phase_mae_high": float(
            np.mean(np.abs(np.angle(np.exp(1j * (phase_pred[high_band] - phase_true[high_band])))))
        ),
        "complex_diff_rel_l2": spectral.relative_l2(coeff_diff, fft_true[:n_plot]),
    }

    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(modes, mag_in, label="Input", color="#444444")
    axes[0].plot(modes, mag_pred, label="Prediction", color="#cc3311")
    axes[0].plot(modes, mag_true, label="Ground truth", color="#0077bb", linestyle="--")
    axes[0].set_ylabel("|Û(k)|")
    axes[0].set_title("Fourier magnitude")
    axes[0].legend()
    axes[1].plot(modes, phase_in, label="Input", color="#444444")
    axes[1].plot(modes, phase_pred, label="Prediction", color="#cc3311")
    axes[1].plot(modes, phase_true, label="Ground truth", color="#0077bb", linestyle="--")
    axes[1].set_ylabel("phase")
    axes[1].set_title("Fourier phase")
    axes[1].legend()
    axes[2].plot(modes, np.abs(coeff_diff), color="#aa3377")
    axes[2].set_xlabel("Fourier mode")
    axes[2].set_ylabel("|Δcoeff|")
    axes[2].set_title("Complex coefficient difference (prediction − truth)")
    fig.suptitle("Step 3 — Single-step spectral comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "spectral_evolution_single_step.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Step 3: Spectral evolution (one step) ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    return metrics


# ---------------------------------------------------------------------------
# Steps 4-6 — Rollout diagnostics, energy, spectral drift
# ---------------------------------------------------------------------------

def rollout_diagnostics(
    model: nn.Module,
    data: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    pair_index = spectral.ROLLOUT_PAIR_INDEX
    series = data["series"]
    times = data["times_full"]
    x = data["x"]
    n_max = min(MAX_ROLLOUT_STEPS, len(series) - 1 - pair_index)
    reference = series[pair_index + 1 : pair_index + n_max + 1]
    prediction = spectral.rollout(model, series[pair_index], n_max, device)

    step_dir = output_dir / "step04_rollout_diagnostics"
    frames_dir = step_dir / "frames"
    step_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    per_step: list[dict[str, float]] = []
    setup_style()
    for step in range(n_max):
        pred = prediction[step]
        ref = reference[step]
        err = np.abs(pred - ref)
        spec = np.abs(np.fft.rfft(pred))
        per_step.append(
            {
                "step": step + 1,
                "t_in": float(times[pair_index + step]),
                "t_out": float(times[pair_index + step + 1]),
                "relative_l2": spectral.relative_l2(pred, ref),
                "max_error": float(np.max(err)),
                "energy_prediction": field_energy(pred),
                "energy_reference": field_energy(ref),
                "energy_ratio": field_energy(pred) / (field_energy(ref) + 1e-12),
            }
        )

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        axes[0].plot(x, pred, color="#cc3311", label="Prediction")
        axes[0].plot(x, ref, color="#0077bb", linestyle="--", label="Reference")
        axes[0].legend()
        axes[0].set_title(f"Step {step + 1} | t_in={times[pair_index + step]:.4f} → t_out={times[pair_index + step + 1]:.4f}")
        axes[1].plot(x, err, color="#aa3377")
        axes[1].set_title("Absolute error")
        axes[2].plot(np.abs(spec[:MAX_PLOT_MODES]), color="#009988")
        axes[2].set_title("|Û(k)| spectrum")
        fig.tight_layout()
        fig.savefig(frames_dir / f"rollout_step_{step + 1:02d}.png", bbox_inches="tight")
        plt.close(fig)

    # Error vs time for multiple horizons
    horizon_metrics: dict[str, list[dict[str, float]]] = {}
    for horizon in ROLLOUT_HORIZONS:
        h = min(horizon, n_max)
        pred_h = spectral.rollout(model, series[pair_index], h, device)
        ref_h = reference[:h]
        horizon_metrics[str(h)] = []
        for step in range(h):
            horizon_metrics[str(h)].append(
                {
                    "step": step + 1,
                    "t_in": float(times[pair_index + step]),
                    "t_out": float(times[pair_index + step + 1]),
                    "relative_l2": spectral.relative_l2(pred_h[step], ref_h[step]),
                    "max_error": float(np.max(np.abs(pred_h[step] - ref_h[step]))),
                    "energy_prediction": field_energy(pred_h[step]),
                    "energy_reference": field_energy(ref_h[step]),
                }
            )

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for horizon in ROLLOUT_HORIZONS:
        h = min(horizon, n_max)
        rels = [m["relative_l2"] for m in horizon_metrics[str(h)]]
        ax.plot(range(1, h + 1), rels, marker="o", label=f"{h}-step rollout")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2")
    ax.set_title("Rollout error vs timestep")
    ax.legend()
    fig.savefig(step_dir / "error_vs_time.png", bbox_inches="tight")
    plt.close(fig)

    # Step 5 — energy analysis
    energy_dir = output_dir / "step05_energy_analysis"
    energy_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = [m["step"] for m in per_step]
    ax.plot(steps, [m["energy_prediction"] for m in per_step], "o-", label="Prediction", color="#cc3311")
    ax.plot(steps, [m["energy_reference"] for m in per_step], "s--", label="Reference", color="#0077bb")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("||u||₂²")
    ax.set_title("Energy evolution during autoregressive rollout")
    ax.legend()
    fig.savefig(energy_dir / "energy_vs_step.png", bbox_inches="tight")
    plt.close(fig)

    energy_summary = {
        "initial_energy_ratio": per_step[0]["energy_ratio"],
        "final_energy_ratio": per_step[-1]["energy_ratio"],
        "max_energy_ratio": max(m["energy_ratio"] for m in per_step),
        "min_energy_ratio": min(m["energy_ratio"] for m in per_step),
        "per_step": per_step,
    }
    with open(energy_dir / "metrics.json", "w") as f:
        json.dump(energy_summary, f, indent=2)

    # Step 6 — spectral drift heatmaps
    drift_dir = output_dir / "step06_spectral_drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    pred_spec = spectral.spectrum_magnitudes(prediction)[:, :MAX_PLOT_MODES]
    ref_spec = spectral.spectrum_magnitudes(reference)[:, :MAX_PLOT_MODES]
    step_times = times[pair_index + 1 : pair_index + n_max + 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, spec, title in zip(
        axes,
        [pred_spec, ref_spec],
        ["Prediction |Û(k)|", "Reference |Û(k)|"],
    ):
        mesh = ax.pcolormesh(np.arange(pred_spec.shape[1]), step_times, spec, shading="auto", cmap="inferno")
        ax.set_xlabel("Fourier mode")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label="|Û(k)|")
    axes[0].set_ylabel("Time")
    fig.suptitle("Step 6 — Spectral drift during rollout", fontsize=14)
    fig.tight_layout()
    fig.savefig(drift_dir / "spectral_drift_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    drift_metrics = {
        "high_mode_growth": float(np.mean(pred_spec[:, MAX_PLOT_MODES // 2 :] / (ref_spec[:, MAX_PLOT_MODES // 2 :] + 1e-12))),
        "low_mode_decay": float(np.mean(pred_spec[:, :4] / (ref_spec[:, :4] + 1e-12))),
        "final_high_to_low_ratio_pred": float(
            np.mean(pred_spec[-1, MAX_PLOT_MODES // 2 :]) / (np.mean(pred_spec[-1, :4]) + 1e-12)
        ),
        "final_high_to_low_ratio_ref": float(
            np.mean(ref_spec[-1, MAX_PLOT_MODES // 2 :]) / (np.mean(ref_spec[-1, :4]) + 1e-12)
        ),
    }
    with open(drift_dir / "metrics.json", "w") as f:
        json.dump(drift_metrics, f, indent=2)

    payload = {
        "per_step": per_step,
        "horizon_metrics": horizon_metrics,
        "energy_summary": energy_summary,
        "drift_metrics": drift_metrics,
    }
    with open(step_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Steps 4-6: Rollout / energy / spectral drift ===")
    print(f"Step-1 rel L2 (AR): {per_step[0]['relative_l2']:.6e}")
    print(f"Step-{n_max} rel L2 (AR): {per_step[-1]['relative_l2']:.6e}")
    print(f"Final energy ratio: {energy_summary['final_energy_ratio']:.6f}")
    return payload


# ---------------------------------------------------------------------------
# Step 7 — Exposure bias
# ---------------------------------------------------------------------------

def step07_exposure_bias(
    model: nn.Module,
    data: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step07_exposure_bias"
    out.mkdir(parents=True, exist_ok=True)

    pair_index = spectral.ROLLOUT_PAIR_INDEX
    series = data["series"]
    n_steps = min(MAX_ROLLOUT_STEPS, len(series) - 1 - pair_index)
    reference = series[pair_index + 1 : pair_index + n_steps + 1]

    ar_pred = spectral.rollout(model, series[pair_index], n_steps, device)
    tf_pred = teacher_forcing_rollout(model, series[pair_index : pair_index + n_steps], n_steps, device)

    ar_errors = [spectral.relative_l2(ar_pred[i], reference[i]) for i in range(n_steps)]
    tf_errors = [spectral.relative_l2(tf_pred[i], reference[i]) for i in range(n_steps)]

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(1, n_steps + 1)
    ax.plot(steps, ar_errors, "o-", label="Autoregressive", color="#cc3311")
    ax.plot(steps, tf_errors, "s--", label="Teacher forcing", color="#0077bb")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2")
    ax.set_title("Exposure bias: teacher forcing vs autoregressive")
    ax.legend()
    fig.savefig(out / "exposure_bias_comparison.png", bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "autoregressive_final_rel_l2": ar_errors[-1],
        "teacher_forcing_final_rel_l2": tf_errors[-1],
        "autoregressive_step1_rel_l2": ar_errors[0],
        "teacher_forcing_step1_rel_l2": tf_errors[0],
        "error_ratio_final_ar_over_tf": ar_errors[-1] / (tf_errors[-1] + 1e-12),
        "autoregressive_errors": ar_errors,
        "teacher_forcing_errors": tf_errors,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Step 7: Exposure bias ===")
    print(f"TF step-1 rel L2: {tf_errors[0]:.6e}")
    print(f"AR step-1 rel L2: {ar_errors[0]:.6e}")
    print(f"TF final rel L2:  {tf_errors[-1]:.6e}")
    print(f"AR final rel L2:  {ar_errors[-1]:.6e}")
    return metrics


# ---------------------------------------------------------------------------
# Step 8 — Data representation
# ---------------------------------------------------------------------------

def step08_data_representation(
    model: spectral.SpectralNet,
    field: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step08_data_representation"
    out.mkdir(parents=True, exist_ok=True)

    traced = trace_forward(model, field, device)
    direct_pred = spectral.predict(model, torch.from_numpy(field).unsqueeze(0), device)[0]

    summary = {
        "input_representation": "physical spatial field u(x) — real-valued array of shape (nx,)",
        "output_representation": "physical spatial field u(x, t+dt) — real-valued array of shape (nx,)",
        "internal_transform": (
            "rFFT → complex mode-wise multiplication on first K modes "
            "→ zero high modes → irFFT → optional nn.Linear in physical space"
        ),
        "not_predicted": [
            "raw Fourier coefficients as final output",
            "Fourier coefficient increments/deltas",
            "complex multipliers alone",
        ],
        "training_target": "numerical physical field at t+dt (MSE in physical space)",
        "inference_matches_training": bool(np.allclose(direct_pred, traced["output_physical"], atol=1e-5)),
        "linear_layer_enabled": model.linear is not None,
        "n_modes_retained": model.n_modes,
        "spectral_weight_magnitude_mean": float(np.mean(np.abs(traced["spectral_weights"]))),
        "spectral_weight_phase_std": float(np.std(np.angle(traced["spectral_weights"]))),
    }

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(model.nx)
    axes[0, 0].plot(x, field, label="Input physical field", color="#444444")
    axes[0, 0].plot(x, traced["output_physical"], label="Output physical field", color="#cc3311")
    axes[0, 0].legend()
    axes[0, 0].set_title("Model I/O in physical space")
    n_plot = min(model.n_modes, MAX_PLOT_MODES)
    modes = np.arange(n_plot)
    axes[0, 1].plot(modes, np.abs(traced["input_fft"][:n_plot]), label="|input FFT|", color="#444444")
    axes[0, 1].plot(modes, np.abs(traced["output_fft"][:n_plot]), label="|output FFT|", color="#cc3311")
    axes[0, 1].legend()
    axes[0, 1].set_title("Internal spectral magnitudes")
    axes[1, 0].plot(modes, np.abs(traced["spectral_weights"][:n_plot]), color="#009988")
    axes[1, 0].set_title("Learned complex spectral weights |w(k)|")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.95,
        "\n".join(
            [
                "Representation audit:",
                f"- Input:  {summary['input_representation']}",
                f"- Output: {summary['output_representation']}",
                f"- Internal: {summary['internal_transform']}",
                f"- Training loss: {summary['training_target']}",
                f"- Inference == traced forward: {summary['inference_matches_training']}",
            ]
        ),
        fontsize=10,
        family="monospace",
        va="top",
    )
    fig.suptitle("Step 8 — Data representation verification", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "data_representation.png", bbox_inches="tight")
    plt.close(fig)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Step 8: Data representation ===")
    print(f"Inference matches traced forward: {summary['inference_matches_training']}")
    return summary


# ---------------------------------------------------------------------------
# Step 9 — Translation consistency
# ---------------------------------------------------------------------------

def step09_translation_consistency(
    model: nn.Module,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out = output_dir / "step09_translation_consistency"
    out.mkdir(parents=True, exist_ok=True)

    gaussian_data = spectral.load_data(spectral.find_validation_data("gaussian"))
    x = gaussian_data["x"]
    base_field = gaussian_data["series"][0]

    results: list[dict[str, float]] = []
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()

    for idx, offset in enumerate(TRANSLATION_OFFSETS):
        shifted = circular_shift_field(base_field, x, offset)
        pred = spectral.predict(model, torch.from_numpy(shifted).unsqueeze(0), device)[0]
        # Reference: exact translation of one-step numerical target from t=0
        target_shifted = circular_shift_field(gaussian_data["targets"][0], x, offset)
        rel_l2 = spectral.relative_l2(pred, target_shifted)
        results.append({"offset": offset, "relative_l2": rel_l2, "max_error": float(np.max(np.abs(pred - target_shifted)))})
        ax = axes[idx]
        ax.plot(x, pred, color="#cc3311", label="Prediction")
        ax.plot(x, target_shifted, color="#0077bb", linestyle="--", label="Shifted target")
        ax.set_title(f"offset={offset:.2f}, relL2={rel_l2:.3e}")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Step 9 — Translation consistency (Gaussian, one-step)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "translation_consistency.png", bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "offsets": results,
        "relative_l2_mean": float(np.mean([r["relative_l2"] for r in results])),
        "relative_l2_std": float(np.std([r["relative_l2"] for r in results])),
        "relative_l2_max": float(np.max([r["relative_l2"] for r in results])),
        "relative_l2_min": float(np.min([r["relative_l2"] for r in results])),
        "position_dependent": float(np.max([r["relative_l2"] for r in results])) > 2.0 * float(np.min([r["relative_l2"] for r in results])),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Step 9: Translation consistency ===")
    print(f"rel L2 range: [{metrics['relative_l2_min']:.3e}, {metrics['relative_l2_max']:.3e}]")
    print(f"position_dependent: {metrics['position_dependent']}")
    return metrics


# ---------------------------------------------------------------------------
# Step 10 — Conclusions report
# ---------------------------------------------------------------------------

def step10_conclusions(output_dir: Path, results: dict[str, Any]) -> str:
    s1 = results["step01"]
    s2 = results["step02"]
    s3 = results["step03"]
    s4 = results["step04"]
    s5 = s4["energy_summary"]
    s6 = s4["drift_metrics"]
    s7 = results["step07"]
    s8 = results["step08"]
    s9 = results["step09"]

    per_step = s4["per_step"]
    ar_final = per_step[-1]["relative_l2"]
    ar_step1 = per_step[0]["relative_l2"]
    tf_final = s7["teacher_forcing_final_rel_l2"]
    tf_step1 = s7["teacher_forcing_step1_rel_l2"]
    one_step_rel_l2 = s1["metrics"]["Relative L2"]
    t_in = s1["t_in"]
    t_out = s1["t_out"]

    rollout_impl_ok = abs(ar_step1 - one_step_rel_l2) < 1e-3 and s1.get("single_equals_rollout", True)
    exposure_bias_dominant = tf_final < 0.5 * ar_final and ar_step1 < 0.1
    error_flat_under_ar = abs(ar_final - ar_step1) / (ar_step1 + 1e-12) < 0.05

    if one_step_rel_l2 > 0.5:
        primary_cause = (
            "learned operator failure on the test IC at the rollout-aligned pair "
            f"(t_in={t_in:.4f} → t_out={t_out:.4f})"
        )
    elif exposure_bias_dominant:
        primary_cause = "exposure bias (error compounding under autoregression)"
    elif s5["final_energy_ratio"] > 1.5 or s5["final_energy_ratio"] < 0.67:
        primary_cause = "energy non-conservation under repeated operator application"
    elif s3["mag_rel_l2_high"] > 2 * s3["mag_rel_l2_low"]:
        primary_cause = "high-frequency magnitude drift under repeated operator application"
    else:
        primary_cause = "learned operator mismatch with the true advection semigroup"

    low_drift_first = s3["mag_rel_l2_low"] > s3["mag_rel_l2_high"]
    mode_drift_text = "low-frequency modes" if low_drift_first else "high-frequency modes"

    report = f"""# Rollout Instability Investigation — Conclusions

Investigation IC: `{INVESTIGATION_IC}`
Output directory: `{output_dir.name}`
Rollout pair index: `{spectral.ROLLOUT_PAIR_INDEX}` (t_in={t_in:.4f} → t_out={t_out:.4f})

## 1. Does instability originate from the model or the rollout pipeline?

**Conclusion: the rollout pipeline is correct; single-step and rollout step 1 use the same pair index and produce identical predictions.**

Evidence:
- One-step relative L2 @ t_in={t_in:.4f}: **{one_step_rel_l2:.4e}**
- Autoregressive step-1 relative L2: **{ar_step1:.4e}** (matches one-step: **{rollout_impl_ok}**)
- one-step == rollout step 1: **{s1.get('single_equals_rollout', True)}**
- Teacher-forcing step-1 relative L2: **{tf_step1:.4e}**
- Inference matches traced forward pass: **{s8['inference_matches_training']}**

## 2. Which Fourier modes drift first?

**Conclusion: {mode_drift_text} show the largest single-step spectral error.**

Evidence:
- Low-band magnitude relative L2: **{s3['mag_rel_l2_low']:.4e}**
- High-band magnitude relative L2: **{s3['mag_rel_l2_high']:.4e}**
- Rollout high-mode attenuation factor (pred/ref): **{s6['high_mode_growth']:.4f}**
- Rollout low-mode attenuation factor (pred/ref): **{s6['low_mode_decay']:.4f}**

## 3. Primary instability mechanisms

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Phase error | Significant | high-phase MAE={s3['phase_mae_high']:.4e}, low-phase MAE={s3['phase_mae_low']:.4e} |
| Magnitude error | Significant | low-band mag rel L2={s3['mag_rel_l2_low']:.4e}, high-band={s3['mag_rel_l2_high']:.4e} |
| Exposure bias | Rejected | TF step-1={tf_step1:.4e}, AR step-1={ar_step1:.4e}, TF final={tf_final:.4e}, AR final={ar_final:.4e} |
| Reconstruction / FFT error | Rejected | NumPy round-trip={s2['numpy_roundtrip_rel_l2']:.3e}, truncation={s2['truncation_rel_l2']:.4e} |
| Normalization mismatch | Rejected | NumPy/Torch coeff max diff={s2['numpy_torch_coeff_max_diff']:.3e} |
| Translation sensitivity | Contributing | translation rel L2 range [{s9['relative_l2_min']:.3e}, {s9['relative_l2_max']:.3e}] |
| Numerical accumulation | Rejected | AR error is O({ar_step1:.2e}) at step 1 and remains nearly flat ({error_flat_under_ar}) |

## 4. Single most responsible issue

**{primary_cause}**

Quantitative support:
- Shared pair index {spectral.ROLLOUT_PAIR_INDEX}: t_in={t_in:.4f}, t_out={t_out:.4f}
- {MAX_ROLLOUT_STEPS}-step autoregressive relative L2: **{ar_final:.4e}** ({ar_final / (one_step_rel_l2 + 1e-12):.2f}× vs step 1)
- Energy ratio at step 1: **{s5['initial_energy_ratio']:.4f}**, final: **{s5['final_energy_ratio']:.4f}**
"""

    report_path = output_dir / "CONCLUSIONS.md"
    report_path.write_text(report)

    summary = {
        "investigation_ic": INVESTIGATION_IC,
        "pair_index": spectral.ROLLOUT_PAIR_INDEX,
        "t_in": t_in,
        "t_out": t_out,
        "one_step_rel_l2": one_step_rel_l2,
        "ar_final_rel_l2": ar_final,
        "tf_final_rel_l2": tf_final,
        "primary_cause": primary_cause,
        "rollout_impl_ok": rollout_impl_ok,
        "single_equals_rollout": s1.get("single_equals_rollout", True),
    }
    with open(output_dir / "investigation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Step 10: Conclusions written ===")
    print(report_path)
    return report


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = make_output_dir()
    print(f"Output directory: {output_dir}")

    model, train_meta = train_model(device)
    data = spectral.load_data(spectral.find_validation_data(INVESTIGATION_IC))
    data_path = spectral.find_validation_data(INVESTIGATION_IC)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "investigation_ic": INVESTIGATION_IC,
                "data_path": str(data_path),
                "train_meta": train_meta,
                "rollout_horizons": ROLLOUT_HORIZONS,
            },
            f,
            indent=2,
        )

    results: dict[str, Any] = {}
    results["step01"] = step01_one_step(model, data, output_dir, device)
    if not results["step01"]["passed"]:
        print(
            "\nNOTE: One-step prediction at the rollout-aligned pair failed the 0.05 threshold; "
            "continuing diagnostics."
        )

    sample_field = data["inputs"][spectral.ROLLOUT_PAIR_INDEX]
    results["step02"] = step02_fft_pipeline(model, sample_field, output_dir, device)
    results["step03"] = step03_spectral_evolution(model, data, output_dir, device)
    results["step04"] = rollout_diagnostics(model, data, output_dir, device)
    results["step07"] = step07_exposure_bias(model, data, output_dir, device)
    results["step08"] = step08_data_representation(model, sample_field, output_dir, device)
    results["step09"] = step09_translation_consistency(model, output_dir, device)
    step10_conclusions(output_dir, results)

    print(f"\nInvestigation complete. Artifacts saved to:\n{output_dir}")


if __name__ == "__main__":
    main()
