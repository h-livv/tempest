"""
Sweep Gaussian center position and measure one-step error with a frozen surrogate.

Trains once (same hyperparameters as the spectral/FNO experiments), then evaluates
u(t) -> u(t+dt) for center_ratio in {0.2, 0.3, ..., 0.8} without retraining.

Usage:
    python -m ml.tests.center_sweep
    python -m ml.tests.center_sweep --model fno
    python -m ml.tests.center_sweep --equation advection
    python -m ml.tests.center_sweep --weights path/to/model.pt
"""

from __future__ import annotations

import argparse
import datetime
import json
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import style
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.core.data import find_validation_data, load_combined_training_data, load_data
from ml.core.device import setup_seed_and_device
from ml.core.equations import add_equation_cli_args, resolve_experiment_ics
from ml.core.eval import compute_metrics, predict
from ml.core.train import train
from ml.diagnostics.plots import PLOT_STYLE
from ml.models.registry import build_model

# ---------------------------------------------------------------------------
# Configuration — edit these lists to choose equation and ICs
# ---------------------------------------------------------------------------

EQUATION = "wave"
TRAIN_ICS = ["gaussian", "square", "sine_wave"]

CENTER_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GAUSSIAN_SIGMA = 0.05
GAUSSIAN_AMPLITUDE = 2.0
ADVECTION_VELOCITY = 1.0

# Shared experiment hyperparameters (match fno/spectral runners).
FNO_WIDTH = 32
FNO_N_LAYERS = 4
FNO_N_MODES = 32
SPECTRAL_N_MODES = 256
USE_LINEAR = True
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


def make_output_dir(model_name: str) -> Path:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / f"center_sweep_{model_name}_{timestamp}_{run_id}"
    )
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


def load_grid_and_dt(equation: str = "wave") -> tuple[np.ndarray, float]:
    """Use the same grid and effective timestep as bundled validation data."""
    data = load_data(find_validation_data("gaussian", equation=equation))
    return data["x"], float(data["dt"])


def _build_model(model_name: str, nx: int, n_modes: int) -> nn.Module:
    if model_name == "spectral":
        return build_model("spectral", nx=nx, n_modes=n_modes, use_linear=USE_LINEAR)
    if model_name == "fno":
        return build_model(
            "fno",
            n_modes=n_modes,
            width=FNO_WIDTH,
            n_layers=FNO_N_LAYERS,
        )
    raise ValueError(f"Unknown model '{model_name}'")


def train_model(
    model_name: str,
    device: torch.device,
    train_ics: list[str],
    equation: str,
) -> tuple[nn.Module, dict]:
    """Train a single surrogate once using experiment hyperparameters."""
    n_modes_cfg = SPECTRAL_N_MODES if model_name == "spectral" else FNO_N_MODES

    train_data = load_combined_training_data(train_ics, equation=equation)
    nx = train_data["inputs"].shape[1]
    n_modes = min(n_modes_cfg, nx // 2 + 1)

    dataset = TensorDataset(
        torch.from_numpy(train_data["inputs"]),
        torch.from_numpy(train_data["targets"]),
    )
    n_train = int(len(dataset) * TRAIN_FRACTION)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=use_cuda,
    )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    model = _build_model(model_name, nx, n_modes).to(device)

    print(
        f"Training once ({model_name}, equation={equation}): "
        f"nx={nx}, n_modes={n_modes}, epochs={EPOCHS}"
    )
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        verbose=True,
    )

    meta: dict = {
        "model": model_name,
        "equation": equation,
        "nx": nx,
        "n_modes": n_modes,
        "epochs": EPOCHS,
        "train_ics": train_ics,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
    }
    if model_name == "fno":
        meta["width"] = FNO_WIDTH
        meta["n_layers"] = FNO_N_LAYERS
    return model, meta


def save_model(model: nn.Module, meta: dict, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["meta"]
    model_name = meta.get("model", "spectral")
    model = _build_model(model_name, meta["nx"], meta["n_modes"]).to(device)
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
        u_pred = predict(model, torch.from_numpy(u_in).unsqueeze(0), device)[0]
        metrics = compute_metrics(u_pred[None], u_target[None])
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


def plot_center_vs_error(results: list[dict], output_dir: Path, model_name: str) -> None:
    style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE)

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

    fig.suptitle(
        f"Frozen {model_name} surrogate — one-step error across Gaussian centers",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "center_vs_one_step_error.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Gaussian center vs one-step error.")
    parser.add_argument(
        "--model",
        choices=["spectral", "fno"],
        default="spectral",
        help="Surrogate architecture to train/evaluate (default: spectral)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Load a pre-trained model checkpoint (skips training)",
    )
    add_equation_cli_args(parser, default_equation=EQUATION)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available",
    )
    args = parser.parse_args()

    equation, train_ics, _ = resolve_experiment_ics(
        args.equation,
        args.train_ics,
        None,
        default_train_ics=TRAIN_ICS,
    )
    device = setup_seed_and_device(RANDOM_SEED, prefer_cuda=not args.cpu)

    output_dir = make_output_dir(args.model)
    x, dt = load_grid_and_dt(equation=equation)
    print(f"Equation: {equation}")
    print(f"Grid nx={len(x)}, effective dt={dt}")

    if args.weights:
        print(f"Loading weights from {args.weights}")
        model, meta = load_model(args.weights, device)
    else:
        model, meta = train_model(args.model, device, train_ics, equation)
        save_model(model, meta, output_dir / "model.pt")

    print(f"\nMeasuring one-step error for centers: {CENTER_RATIOS}")
    results = measure_one_step_errors(model, x, dt, device, CENTER_RATIOS)

    payload = {
        "model": args.model,
        "equation": equation,
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

    plot_center_vs_error(results, output_dir, args.model)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
