from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
}

def _setup_plot_style() -> None:
    style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE)


def _mode_axis_limit(n_freq: int, max_modes: int = 32) -> int:
    """Limit Fourier mode axes to the low-frequency band used in training."""
    return min(n_freq, max_modes)


def plot_overlay(
    x: np.ndarray,
    prediction: np.ndarray,
    numerical: np.ndarray,
    analytical: np.ndarray,
    output_dir: Path,
    sample_index: int = 0,
    times_full: np.ndarray | None = None,
    output_name: str = "01_overlay.png",
) -> None:
    """1. Prediction vs Numerical overlay at the rollout-aligned one-step pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    if times_full is not None:
        t_in = float(times_full[sample_index])
        t_out = float(times_full[sample_index + 1])
        title = f"One-step prediction (index {sample_index}, t_in={t_in:.4f} → t_out={t_out:.4f})"
    else:
        title = f"One-step prediction (pair index {sample_index})"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, prediction[sample_index], label="Prediction", color="#cc3311")
    ax.plot(x, numerical[sample_index], label="Numerical", color="#0077bb", linestyle="--")
    #ax.plot(x, analytical[sample_index], label="Analytical", color="#009988", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(frameon=True)
    fig.savefig(output_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def spectrum_magnitudes(series: np.ndarray) -> np.ndarray:
    """Return |rFFT| for each snapshot. Shape: (T, n_freq)."""
    return np.abs(np.fft.rfft(series, axis=-1))


def plot_fourier_spectrum_evolution(
    series: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
) -> None:
    """2. |Û(k)| heatmap: Fourier mode (x) vs timestep (y) for all timesteps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    magnitudes = spectrum_magnitudes(series)
    n_plot = _mode_axis_limit(magnitudes.shape[1])
    mode_indices = np.arange(n_plot)
    magnitudes = magnitudes[:, :n_plot]

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(
        mode_indices,
        times,
        magnitudes,
        shading="auto",
        cmap="inferno",
    )
    ax.set_xlim(0, n_plot - 1)
    ax.set_xlabel("Fourier Mode")
    ax.set_ylabel("Time")
    ax.set_title("Fourier Spectrum Evolution  |Û(k)|")
    fig.colorbar(mesh, ax=ax, label="|Û(k)|")
    fig.savefig(output_dir / "02_fourier_spectrum_evolution.png", bbox_inches="tight")
    plt.close(fig)


def reconstruct_field(field: np.ndarray, n_modes: int) -> np.ndarray:
    """Reconstruct a field keeping only the lowest n_modes Fourier coefficients."""
    spectrum = np.fft.rfft(field)
    truncated = np.zeros_like(spectrum)
    truncated[:n_modes] = spectrum[:n_modes]
    return np.fft.irfft(truncated, n=field.shape[-1])


def plot_fourier_reconstruction(
    x: np.ndarray,
    field: np.ndarray,
    recon_modes: list[int],
    output_dir: Path,
) -> None:
    """3. Same field reconstructed with increasing mode counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, field, label="Original", color="#222222", linewidth=2.5)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(recon_modes)))

    for n_modes, color in zip(recon_modes, colors):
        reconstructed = reconstruct_field(field, n_modes)
        ax.plot(x, reconstructed, label=f"{n_modes} modes", color=color)

    ax.set_title("Fourier Reconstruction")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(frameon=True, title="Retained modes")
    fig.savefig(output_dir / "03_fourier_reconstruction.png", bbox_inches="tight")
    plt.close(fig)


def plot_modes_vs_relative_error(
    modes: list[int],
    rel_errors_numerical: list[float],
    rel_errors_analytical: list[float],
    output_dir: Path,
) -> None:
    """4. Relative L2 error as a function of retained Fourier modes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(modes, rel_errors_numerical, "o-", label="vs Numerical", color="#0077bb")
    ax.plot(modes, rel_errors_analytical, "s--", label="vs Analytical", color="#009988")
    ax.set_xscale("log", base=2)
    ax.set_xticks(modes)
    ax.set_xticklabels([str(m) for m in modes])
    ax.set_xlabel("Number of Fourier Modes")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Number of Modes vs Relative L2 Error")
    ax.legend(frameon=True)
    fig.savefig(output_dir / "04_modes_vs_relative_error.png", bbox_inches="tight")
    plt.close(fig)


def plot_spectral_coefficients(
    prediction: np.ndarray,
    target: np.ndarray,
    output_dir: Path,
) -> None:
    """5. |Û(k)| for prediction and target on one plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    pred_spec = np.abs(np.fft.rfft(prediction))
    target_spec = np.abs(np.fft.rfft(target))
    n_plot = _mode_axis_limit(len(pred_spec))
    mode_indices = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mode_indices, pred_spec[:n_plot], label="Prediction", color="#cc3311")
    ax.plot(mode_indices, target_spec[:n_plot], label="Target", color="#0077bb", linestyle="--")
    ax.set_xlim(0, n_plot - 1)
    ax.set_xlabel("Fourier Mode")
    ax.set_ylabel("|Û(k)|")
    ax.set_title("Spectral Coefficients")
    ax.legend(frameon=True)
    fig.savefig(output_dir / "05_spectral_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def plot_rollout_comparison(
    x: np.ndarray,
    step_times: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_dir: Path,
) -> None:
    """6. Multi-step rollout: prediction vs ground truth at every timestep."""
    from ml.core.eval import relative_l2
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    n_steps = ground_truth.shape[0]
    rel_l2_errors = [relative_l2(prediction[i], ground_truth[i]) for i in range(n_steps)]
    mae_errors = [float(np.mean(np.abs(prediction[i] - ground_truth[i]))) for i in range(n_steps)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    vmin = min(ground_truth.min(), prediction.min())
    vmax = max(ground_truth.max(), prediction.max())

    im0 = axes[0].pcolormesh(x, step_times, ground_truth, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Time")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x, step_times, prediction, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Rollout Prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Time")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(
        x,
        step_times,
        np.abs(prediction - ground_truth),
        shading="auto",
        cmap="magma",
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Time")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("Multi-Step Rollout", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "06_rollout_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_times, rel_l2_errors, "o-", label="Relative L2", color="#cc3311")
    ax.plot(step_times, mae_errors, "s--", label="Mean Absolute Error", color="#0077bb")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title("Rollout Error vs Timestep")
    ax.legend(frameon=True)
    fig.savefig(output_dir / "06_rollout_error_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def save_rollout_frames(
    x: np.ndarray,
    prediction: np.ndarray,
    numerical: np.ndarray,
    analytical: np.ndarray,
    times_full: np.ndarray,
    output_dir: Path,
    pair_index: int = 0,
) -> Path:
    """Save per-timestep rollout comparison images (analytical, numerical, prediction)."""
    from ml.core.eval import relative_l2
    frames_dir = output_dir / "rollout_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    y_min = min(numerical.min(), prediction.min())   #, analytical.min())
    y_max = max(numerical.max(), prediction.max()) #, analytical.max())
    y_pad = 0.05 * (y_max - y_min + 1e-12)

    for i in range(prediction.shape[0]):
        t_in = float(times_full[pair_index + i])
        t_out = float(times_full[pair_index + i + 1])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        #ax.plot(x, analytical[i], label="Analytical", color="#009988", linestyle=":")
        ax.plot(x, numerical[i], label="Numerical", color="#0077bb", linestyle="--")
        ax.plot(x, prediction[i], label="Prediction", color="#cc3311")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        l2_err = relative_l2(prediction[i], numerical[i])
        ax.set_title(
            f"Rollout step {i + 1} | t_in={t_in:.4f} → t_out={t_out:.4f} | Relative L2 = {l2_err:.5f}"
        )
        ax.legend(frameon=True)
        fig.savefig(frames_dir / f"frame_{i:04d}.png", bbox_inches="tight")
        plt.close(fig)

    return frames_dir


def generate_rollout_video(frames_dir: Path, output_video: Path) -> None:
    """Build an MP4 from rollout frame images using ml/animate.py."""
    animate_script = PROJECT_ROOT / "ml" / "animate.py"
    subprocess.run(
        [
            sys.executable,
            str(animate_script),
            str(frames_dir),
            "--output",
            str(output_video),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
