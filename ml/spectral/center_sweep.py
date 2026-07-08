"""
Sweep Gaussian center position and measure one-step error with a frozen surrogate.

Trains once (same hyperparameters as spectral.py), then evaluates u(t) -> u(t+dt)
for center_ratio in {0.2, 0.3, ..., 0.8} without retraining.

Usage:
    python ml/spectral/center_sweep.py
    python ml/spectral/center_sweep.py --weights path/to/model.pt
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
import uuid
from pathlib import Path

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

CENTER_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GAUSSIAN_SIGMA = 0.05
GAUSSIAN_AMPLITUDE = 2.0
ADVECTION_VELOCITY = 1.0


def make_output_dir() -> Path:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = SPECTRAL_DIR / "outputs" / f"center_sweep_{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def gaussian_field(
    x: np.ndarray,
    center_ratio: float,
    sigma: float = GAUSSIAN_SIGMA,
    amplitude: float = GAUSSIAN_AMPLITUDE,
) -> np.ndarray:
    """Gaussian profile matching init_conditions.GaussianIC on a 1D periodic grid."""
    center = center_ratio * float(x.max())
    return (amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))).astype(np.float32)


def advect_periodic_1d(
    field: np.ndarray,
    x: np.ndarray,
    velocity: float,
    dt: float,
) -> np.ndarray:
    """One-step periodic advection target (exact for linear advection)."""
    lx = float(x[-1] - x[0] + (x[1] - x[0]))
    x_shift = (x - velocity * dt - x[0]) % lx + x[0]
    return np.interp(x_shift, x, field).astype(np.float32)


def load_grid_and_dt() -> tuple[np.ndarray, float]:
    """Use the same grid and effective timestep as bundled validation data."""
    data = spectral.load_data(spectral.find_validation_data("gaussian"))
    return data["x"], float(data["dt"])


def train_model(device: torch.device) -> tuple[nn.Module, dict]:
    """Train a single surrogate once using spectral.py hyperparameters."""
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

    print(f"Training once: nx={nx}, n_modes={n_modes}, epochs={spectral.EPOCHS}")
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
        "epochs": spectral.EPOCHS,
        "train_ics": spectral.TRAIN_ICS,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }
    return model, meta


def save_model(model: nn.Module, meta: dict, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["meta"]
    model = spectral.SpectralNet(
        nx=meta["nx"],
        n_modes=meta["n_modes"],
        use_linear=spectral.USE_LINEAR,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def measure_one_step_errors(
    model: nn.Module,
    x: np.ndarray,
    dt: float,
    device: torch.device,
    center_ratios: list[float],
) -> list[dict]:
    """Evaluate one-step error for each Gaussian center without retraining."""
    results: list[dict] = []
    for center_ratio in center_ratios:
        u_in = gaussian_field(x, center_ratio)
        u_target = advect_periodic_1d(u_in, x, ADVECTION_VELOCITY, dt)
        u_pred = spectral.predict(model, torch.from_numpy(u_in).unsqueeze(0), device)[0]
        metrics = spectral.compute_metrics(u_pred[None], u_target[None])
        results.append(
            {
                "center_ratio": center_ratio,
                "center": center_ratio * float(x.max()),
                "relative_l2": metrics["Relative L2"],
                "mse": metrics["MSE"],
                "mae": metrics["MAE"],
                "max_error": metrics["Max Error"],
            }
        )
        print(
            f"  center={center_ratio:.1f} | rel L2={metrics['Relative L2']:.6e}"
            f" | MSE={metrics['MSE']:.6e}"
        )
    return results


def plot_center_vs_error(results: list[dict], output_dir: Path) -> None:
    style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(spectral.PLOT_STYLE)

    centers = [r["center_ratio"] for r in results]
    rel_l2 = [r["relative_l2"] for r in results]
    mse = [r["mse"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(centers, rel_l2, "o-", color="#cc3311", linewidth=2, markersize=7)
    axes[0].axvline(0.5, color="#888888", linestyle=":", label="train IC center (0.5)")
    axes[0].set_xlabel("Gaussian center ratio")
    axes[0].set_ylabel("Relative L2")
    axes[0].set_title("One-step error vs center")
    axes[0].set_xticks(centers)
    axes[0].legend(frameon=True)

    axes[1].semilogy(centers, mse, "s--", color="#0077bb", linewidth=2, markersize=7)
    axes[1].axvline(0.5, color="#888888", linestyle=":", label="train IC center (0.5)")
    axes[1].set_xlabel("Gaussian center ratio")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("One-step MSE vs center")
    axes[1].set_xticks(centers)
    axes[1].legend(frameon=True)

    fig.suptitle("Frozen surrogate — one-step error across Gaussian centers", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "center_vs_one_step_error.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Gaussian center vs one-step error.")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Load a pre-trained model checkpoint (skips training)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = make_output_dir()
    x, dt = load_grid_and_dt()
    print(f"Grid nx={len(x)}, effective dt={dt}")

    if args.weights:
        print(f"Loading weights from {args.weights}")
        model, meta = load_model(args.weights, device)
    else:
        model, meta = train_model(device)
        save_model(model, meta, output_dir / "model.pt")

    print(f"\nMeasuring one-step error for centers: {CENTER_RATIOS}")
    results = measure_one_step_errors(model, x, dt, device, CENTER_RATIOS)

    payload = {
        "center_ratios": CENTER_RATIOS,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "gaussian_amplitude": GAUSSIAN_AMPLITUDE,
        "advection_velocity": ADVECTION_VELOCITY,
        "effective_dt": dt,
        "train_meta": meta,
        "results": results,
    }
    with open(output_dir / "center_sweep.json", "w") as f:
        json.dump(payload, f, indent=2)

    plot_center_vs_error(results, output_dir)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
