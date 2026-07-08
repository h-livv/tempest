"""
Minimal Fourier Neural Operator demo for one-step PDE field prediction.

This script trains a small spectral network to map u(x, t) -> u(x, t + dt)
and compares predictions against numerical and analytical ground truth.

Educational focus: FFT -> learn in Fourier space -> inverse FFT.
Not a full FNO implementation from the paper.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import style
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALIDATION_DIR = PROJECT_ROOT / "outputs/wave/1D/validation"
TRAIN_ICS = ["gaussian", "square", "sine_wave"]
TEST_ICS = ["gaussian", "square", "double_gaussian", "shifted_gaussian", "sine_wave"]


def find_validation_data(initial_condition: str) -> Path:
    """Return the latest validation spatial_data.npz for an exact initial condition name."""
    matches: list[Path] = []
    for npz_path in sorted(VALIDATION_DIR.glob("*/data/spatial_data.npz")):
        config_path = npz_path.parent / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            ic_name = json.load(f).get("initial_condition")
        if ic_name == initial_condition:
            matches.append(npz_path)
    if not matches:
        raise FileNotFoundError(
            f"No validation data found for '{initial_condition}' under {VALIDATION_DIR}"
        )
    return matches[-1]


def load_combined_training_data(initial_conditions: list[str]) -> dict[str, Any]:
    """Load and concatenate one-step training pairs from multiple initial conditions."""
    datasets = [load_data(find_validation_data(ic)) for ic in initial_conditions]
    nx = datasets[0]["inputs"].shape[1]
    for data in datasets[1:]:
        if data["inputs"].shape[1] != nx:
            raise ValueError("All training datasets must share the same spatial resolution.")

    return {
        "inputs": np.concatenate([d["inputs"] for d in datasets], axis=0),
        "targets": np.concatenate([d["targets"] for d in datasets], axis=0),
        "analytical": np.concatenate([d["analytical"] for d in datasets], axis=0),
        "x": datasets[0]["x"],
        "sources": initial_conditions,
    }

# Training hyperparameters
N_MODES = 256          # Default modes for sample field plots
MODE_SWEEP = [2, 4, 8, 16, 32, 64]  # Modes scanned for convergence plot
USE_LINEAR = True     # Optional pointwise linear layer after inverse FFT
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

# Plotting
RECON_MODES = [2, 4, 8, 16, 32, 64]  # Mode counts for Fourier reconstruction plot
MAX_PLOT_MODES = 32
#max(max(MODE_SWEEP), max(RECON_MODES), N_MODES)  # x-axis limit for spectrum plots
# Index shared by one-step plots and rollout (inputs[i] -> targets[i], rollout from series[i])
ROLLOUT_PAIR_INDEX = 0
PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _periodic_interp_2d(
    field: np.ndarray,
    x_shift: np.ndarray,
    y_shift: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Sample a 2D periodic field at shifted coordinates."""
    x0, x1 = float(x_coords[0]), float(x_coords[-1])
    y0, y1 = float(y_coords[0]), float(y_coords[-1])
    lx = x1 - x0 + (x_coords[1] - x_coords[0])
    ly = y1 - y0 + (y_coords[1] - y_coords[0])

    x_wrapped = (x_shift - x0) % lx + x0
    y_wrapped = (y_shift - y0) % ly + y0

    # Regular grid: use bilinear-style indexing via searchsorted
    ix = np.searchsorted(x_coords, x_wrapped, side="right") - 1
    iy = np.searchsorted(y_coords, y_wrapped, side="right") - 1
    ix = np.clip(ix, 0, len(x_coords) - 2)
    iy = np.clip(iy, 0, len(y_coords) - 2)

    tx = (x_wrapped - x_coords[ix]) / (x_coords[ix + 1] - x_coords[ix] + 1e-12)
    ty = (y_wrapped - y_coords[iy]) / (y_coords[iy + 1] - y_coords[iy] + 1e-12)

    v00 = field[iy, ix]
    v10 = field[iy, ix + 1]
    v01 = field[iy + 1, ix]
    v11 = field[iy + 1, ix + 1]

    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def _analytical_wave_1d(
    initial_field: np.ndarray,
    x_coords: np.ndarray,
    times: np.ndarray,
    wave_speed: float,
) -> np.ndarray:
    """D'Alembert solution for 1D wave equation with periodic boundaries."""
    u0 = initial_field
    dx = float(x_coords[1] - x_coords[0])
    domain_length = float(x_coords.max() + dx)
    analytical = np.zeros((len(times), len(x_coords)), dtype=np.float64)

    for i, t in enumerate(times):
        x_minus = x_coords - wave_speed * t
        x_plus = x_coords + wave_speed * t
        x_right = (x_minus - x_coords[0]) % domain_length + x_coords[0]
        x_left = (x_plus - x_coords[0]) % domain_length + x_coords[0]
        u_right = np.interp(x_right, x_coords, u0)
        u_left = np.interp(x_left, x_coords, u0)
        analytical[i] = 0.5 * (u_right + u_left)

    return analytical


def _analytical_advection_slice(
    initial_field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    y_index: int,
    velocity: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute analytical 1D slices u(x, y=y_index, t) for 2D linear advection.

    Uses the exact solution u(x, y, t) = u0(x - vx*t, y - vy*t) with periodic BCs.
    """
    vx = float(velocity[0]) if len(velocity) > 0 else 0.0
    vy = float(velocity[1]) if len(velocity) > 1 else 0.0

    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")
    analytical = np.zeros((len(times), len(x_coords)), dtype=np.float64)

    for i, t in enumerate(times):
        x_shift = x_grid - vx * t
        y_shift = y_grid - vy * t
        full_field = _periodic_interp_2d(initial_field, x_shift, y_shift, x_coords, y_coords)
        analytical[i] = full_field[y_index]

    return analytical


def load_data(
    data_path: Path | str,
    slice_index: int | None = None,
) -> dict[str, Any]:
    """
    Load PDE snapshots and build (input, numerical target, analytical target) pairs.

    Returns arrays with shapes (N, nx):
        inputs[i]     = u(x, t_i)
        targets[i]    = numerical u(x, t_i + dt)
        analytical[i] = analytical u(x, t_i + dt)

    For 2D simulation data, a fixed spatial row is extracted to form a 1D problem.
    """
    data_path = Path(data_path)
    raw = np.load(data_path)
    ml_tensor = raw["ml_tensor_data"]  # (T, ...) time series of numerical snapshots

    # Load simulation times from companion CSV when available
    time_csv = data_path.parent / "time_history.csv"
    if time_csv.exists():
        import pandas as pd

        times = pd.read_csv(time_csv)["time"].to_numpy(dtype=np.float64)
    else:
        times = np.arange(ml_tensor.shape[0], dtype=np.float64)

    if times.shape[0] != ml_tensor.shape[0]:
        times = np.linspace(0.0, 1.0, ml_tensor.shape[0])

    # Reduce 2D fields to a 1D slice along x (fixed y index)
    y_coords: np.ndarray | None = None
    if ml_tensor.ndim == 3:
        if slice_index is None:
            slice_index = ml_tensor.shape[1] // 2
        series = ml_tensor[:, slice_index, :]  # (T, nx)
        if raw["x"].ndim == 3:
            # Tempest stores meshgrid coordinates: x[1] varies along columns, x[0] along rows
            x_coords = raw["x"][1, slice_index, :]
            y_coords = raw["x"][0, :, 0]
        else:
            x_coords = raw["x"][0]
    elif ml_tensor.ndim == 2:
        series = ml_tensor
        x_coords = raw["x"][0] if raw["x"].ndim > 1 else raw["x"]
    else:
        raise ValueError(f"Unsupported ml_tensor_data shape: {ml_tensor.shape}")

    # One-step pairs: input at t_i, target at t_{i+1}
    inputs = series[:-1]
    targets = series[1:]
    target_times = times[1:]

    # Analytical solution at each target time
    config_path = data_path.parent / "config.json"
    equation_name = "advection"
    wave_speed = 1.0
    velocity = np.array([1.0, 1.0], dtype=np.float64)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        dt_step = float(config.get("dt", 1.0))
        record_interval = int(config.get("record_interval", 1))
        effective_dt = dt_step * record_interval
        equation_name = config.get("equation", "advection")
        wave_speed = float(config.get("wave_speed", 1.0))
    else:
        effective_dt = float(times[1] - times[0]) if len(times) > 1 else 1.0

    initial_field = ml_tensor[0] if ml_tensor.ndim == 3 else series[0]

    if equation_name == "wave":
        if ml_tensor.ndim == 3 and y_coords is not None:
            analytical = _analytical_wave_1d(
                initial_field=initial_field[slice_index, :],
                x_coords=x_coords,
                times=target_times,
                wave_speed=wave_speed,
            )
        else:
            analytical = _analytical_wave_1d(
                initial_field=initial_field,
                x_coords=x_coords,
                times=target_times,
                wave_speed=wave_speed,
            )
    elif initial_field is not None and ml_tensor.ndim == 3 and y_coords is not None:
        analytical = _analytical_advection_slice(
            initial_field=initial_field,
            x_coords=x_coords,
            y_coords=y_coords,
            y_index=slice_index,
            velocity=velocity,
            times=target_times,
        )
    else:
        # Fallback: shift the initial 1D profile (pure x-advection)
        u0 = series[0]
        lx = float(x_coords[-1] - x_coords[0] + (x_coords[1] - x_coords[0]))
        analytical = np.zeros_like(targets)
        for i, t in enumerate(target_times):
            x_shift = (x_coords - velocity[0] * t - x_coords[0]) % lx + x_coords[0]
            analytical[i] = np.interp(x_shift, x_coords, u0)

    return {
        "inputs": inputs.astype(np.float32),
        "targets": targets.astype(np.float32),
        "analytical": analytical.astype(np.float32),
        "series": series.astype(np.float32),
        "x": x_coords.astype(np.float32),
        "times": target_times.astype(np.float32),
        "times_full": times.astype(np.float32),
        "dt": np.float32(effective_dt),
        "slice_index": slice_index,
        "data_path": str(data_path),
    }


def benchmark_numerical_solver(data_path: Path | str) -> float:
    """Re-run the Tempest numerical solver for a validation case and return wall time."""
    from src.core.config import SimulationConfig
    from src.core.simulation import Simulation
    from src.mesh import boundaries
    from src.numerics import operators, integrators
    from src.physics import equations, init_conditions

    data_path = Path(data_path)
    with open(data_path.parent / "config.json") as f:
        cfg = json.load(f)

    ic_map = {
        "gaussian": init_conditions.GaussianIC(),
        "square": init_conditions.SquareIC(),
        "shifted_gaussian": init_conditions.ShiftedGaussianIC(),
        "sine_wave": init_conditions.SineWaveIC(),
        "double_gaussian": init_conditions.DoubleGaussianIC(),
        "spike": init_conditions.SpikeIC(),
    }
    integrator_map = {
        "rk4": integrators.rk4,
        "euler": integrators.euler,
        "leapfrog": integrators.leapfrog,
    }
    operator_map = {
        "upwind": operators.upwind,
        "gradient": operators.gradient,
        "laplacian": operators.laplacian,
    }
    equation_map = {
        "advection": lambda cfg: equations.AdvectionEquation(velocity=float(cfg.get("velocity", 1.0))),
        "wave": lambda cfg: equations.WaveEquation(wave_speed=float(cfg.get("wave_speed", 1.0))),
    }
    boundary_map = {
        "periodic": boundaries.periodic,
        "dirichlet": boundaries.Dirichlet(0.0, 0.0),
    }

    n_val = cfg["N"]
    dx_val = cfg["dx"]
    shape = (n_val,) if isinstance(n_val, int) else tuple(n_val)
    spacing = (dx_val,) if isinstance(dx_val, (int, float)) else tuple(dx_val)

    equation_name = cfg.get("equation", "advection")
    if equation_name not in equation_map:
        raise ValueError(f"Unsupported equation '{equation_name}' in benchmark_numerical_solver")

    wave_fields = 2 if equation_name == "wave" else 1

    def make_ic(name: str):
        if name == "gaussian":
            return init_conditions.GaussianIC(num_fields=wave_fields)
        if name == "square":
            return init_conditions.SquareIC(num_fields=wave_fields)
        if name == "shifted_gaussian":
            return init_conditions.ShiftedGaussianIC(num_fields=wave_fields)
        if name == "sine_wave":
            return init_conditions.SineWaveIC(num_fields=wave_fields)
        if name == "double_gaussian":
            return init_conditions.DoubleGaussianIC(num_fields=wave_fields, speed=0.0)
        if name == "spike":
            return init_conditions.SpikeIC(num_fields=wave_fields)
        raise KeyError(name)

    ic_name = cfg["initial_condition"]
    if ic_name not in ic_map and ic_name not in {
        "gaussian", "square", "shifted_gaussian", "sine_wave", "double_gaussian", "spike"
    }:
        raise KeyError(f"Unknown initial condition '{ic_name}'")

    sim_config = SimulationConfig(
        shape=shape,
        spacing=spacing,
        dt=float(cfg["dt"]),
        final_time=float(cfg["final_time"]),
        steps_per_frame=int(cfg["steps_per_frame"]),
        record_interval=int(cfg["record_interval"]),
        equation=equation_map[equation_name](cfg),
        operator=operator_map[cfg["operator"]],
        boundary=boundary_map[cfg["boundary_function"]],
        integrator=integrator_map[cfg["integrator"]],
        initial_condition=make_ic(ic_name),
    )

    start = time.perf_counter()
    Simulation(sim_config).run()
    return time.perf_counter() - start


def print_timing_comparison(timing: dict[str, Any]) -> None:
    """Print numerical solver vs surrogate wall-clock comparison."""
    print("\n" + "=" * 26)
    print("Timing Comparison")
    print("=" * 26)
    print(f"\nSurrogate training ({', '.join(TRAIN_ICS)}): {timing['surrogate_training']:.3f} s")
    print(f"Surrogate inference (all test ICs):          {timing['surrogate_inference']:.3f} s")
    print(f"Surrogate total:                             {timing['surrogate_total']:.3f} s")
    print("\nNumerical solver (per test IC):")
    for ic_name, elapsed in timing["numerical_solver_by_ic"].items():
        print(f"  {ic_name:<18} {elapsed:.3f} s")
    print(f"Numerical solver total:                      {timing['numerical_solver_total']:.3f} s")
    print(f"Speedup (numerical total / surrogate total): {timing['speedup']:.2f}x")


def save_timing(timing: dict[str, Any], output_dir: Path) -> None:
    """Write timing summary to JSON in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **timing,
        "train_ics": TRAIN_ICS,
        "test_ics": TEST_ICS,
        "train_data_paths": {ic: str(find_validation_data(ic)) for ic in TRAIN_ICS},
        "test_data_paths": {ic: str(find_validation_data(ic)) for ic in TEST_ICS},
    }
    with open(output_dir / "timings.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Model: minimal spectral network
# ---------------------------------------------------------------------------

class SpectralNet(nn.Module):
    """
    Minimal spectral neural network.

    Input -> FFT -> keep K modes -> complex multiply -> zero high modes
         -> inverse FFT -> (optional linear) -> output
    """

    def __init__(self, nx: int, n_modes: int, use_linear: bool = True) -> None:
        super().__init__()
        self.nx = nx
        self.n_modes = min(n_modes, nx // 2 + 1)
        self.n_freq = nx // 2 + 1

        # Learnable complex weights for the lowest K Fourier modes
        weight = torch.randn(self.n_modes, dtype=torch.cfloat) * 0.02
        self.spectral_weights = nn.Parameter(weight)

        self.linear: nn.Linear | None
        if use_linear:
            self.linear = nn.Linear(nx, nx)
        else:
            self.linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, nx) real spatial field
        x_ft = torch.fft.rfft(x, dim=-1)

        # Keep only the first K modes; zero out the rest
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, : self.n_modes] = x_ft[:, : self.n_modes] * self.spectral_weights

        out = torch.fft.irfft(out_ft, n=self.nx, dim=-1)

        if self.linear is not None:
            out = self.linear(out)

        return out


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Train with Adam + MSE. Returns per-epoch loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            n_train += x_batch.size(0)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * x_batch.size(0)
                n_val += x_batch.size(0)

        train_loss /= max(n_train, 1)
        val_loss /= max(n_val, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if verbose:
            print(f"Epoch {epoch:4d} | Training Loss: {train_loss:.6e} | Validation Loss: {val_loss:.6e}")

    return history


def predict(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run the trained model on a batch of input fields."""
    model.eval()
    with torch.no_grad():
        return model(inputs.to(device)).cpu().numpy()


def compute_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics between prediction and reference."""
    diff = prediction - reference
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ref_norm = float(np.linalg.norm(reference))
    rel_l2 = float(np.linalg.norm(diff) / (ref_norm + 1e-12))
    max_error = float(np.max(np.abs(diff)))

    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((reference - np.mean(reference)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Relative L2": rel_l2,
        "Max Error": max_error,
        "R²": r2,
    }


def evaluate(
    model: nn.Module,
    inputs: np.ndarray,
    numerical: np.ndarray,
    analytical: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Predict on all samples and compute metrics vs numerical and analytical truth."""
    pred = predict(model, torch.from_numpy(inputs), device)

    metrics_numerical = compute_metrics(pred, numerical)
    metrics_analytical = compute_metrics(pred, analytical)

    return {"numerical": metrics_numerical, "analytical": metrics_analytical}, pred


def rollout(
    model: nn.Module,
    initial_state: np.ndarray,
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Autoregressively predict the next n_steps fields from an initial state."""
    model.eval()
    trajectory = np.zeros((n_steps, initial_state.shape[-1]), dtype=np.float32)
    state = torch.from_numpy(initial_state).float().unsqueeze(0).to(device)

    with torch.no_grad():
        for step in range(n_steps):
            state = model(state)
            trajectory[step] = state.squeeze(0).cpu().numpy()

    return trajectory


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Global relative L2 error ||pred - ref|| / ||ref||."""
    diff = prediction - reference
    return float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-12))


def verify_rollout_data_alignment(
    model: nn.Module,
    test_data: dict[str, Any],
    device: torch.device,
    pair_index: int = ROLLOUT_PAIR_INDEX,
) -> dict[str, Any]:
    """
    Verify that single-step and rollout paths use identical data at tᵢ→tᵢ₊₁.

    Checks:
      - inputs[i] == series[i]       (one-step input matches rollout initial state)
      - targets[i] == series[i + 1]  (one-step target matches rollout frame-1 reference)
      - predict(inputs[i]) == rollout(series[i])[0] (same model output)
    """
    inputs = test_data["inputs"]
    targets = test_data["targets"]
    series = test_data["series"]
    times_full = test_data["times_full"]

    pred_single = predict(model, torch.from_numpy(inputs[pair_index : pair_index + 1]), device)[0]
    pred_rollout = rollout(model, series[pair_index], 1, device)[0]

    report = {
        "pair_index": pair_index,
        "t_in": float(times_full[pair_index]),
        "t_out": float(times_full[pair_index + 1]),
        "inputs_equal_series": bool(np.allclose(inputs[pair_index], series[pair_index])),
        "targets_equal_series_next": bool(np.allclose(targets[pair_index], series[pair_index + 1])),
        "single_step_equals_rollout_step0": bool(np.allclose(pred_single, pred_rollout)),
        "single_rollout_max_abs_diff": float(np.max(np.abs(pred_single - pred_rollout))),
        "relative_l2": relative_l2(pred_single, targets[pair_index]),
        "alignment_ok": True,
    }
    report["alignment_ok"] = (
        report["inputs_equal_series"]
        and report["targets_equal_series_next"]
        and report["single_step_equals_rollout_step0"]
    )
    return report


def save_rollout_alignment_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write rollout alignment verification to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "rollout_alignment.json", "w") as f:
        json.dump(report, f, indent=2)


def print_rollout_alignment_report(report: dict[str, Any], ic_name: str) -> None:
    """Print rollout vs single-step data path verification."""
    print(f"\n  Rollout alignment ({ic_name}) @ pair index {report['pair_index']}"
          f" (t_in={report['t_in']:.4f} → t_out={report['t_out']:.4f}):")
    print(f"    inputs[i] == series[i]:            {report['inputs_equal_series']}")
    print(f"    targets[i] == series[i+1]:        {report['targets_equal_series_next']}")
    print(f"    predict(inputs[i]) == rollout[0]: {report['single_step_equals_rollout_step0']}"
          f" (max |diff|={report['single_rollout_max_abs_diff']:.3e})")
    print(f"    one-step / rollout-step-1 rel L2:  {report['relative_l2']:.6e}")


def reconstruct_field(field: np.ndarray, n_modes: int) -> np.ndarray:
    """Reconstruct a field keeping only the lowest n_modes Fourier coefficients."""
    spectrum = np.fft.rfft(field)
    truncated = np.zeros_like(spectrum)
    truncated[:n_modes] = spectrum[:n_modes]
    return np.fft.irfft(truncated, n=field.shape[-1])


def spectrum_magnitudes(series: np.ndarray) -> np.ndarray:
    """Return |rFFT| for each snapshot. Shape: (T, n_freq)."""
    return np.abs(np.fft.rfft(series, axis=-1))


def print_metrics(metrics: dict[str, dict[str, float]], split_label: str = "Test") -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 26)
    print(f"Evaluation Results ({split_label})")
    print("=" * 26)

    for label, title in [("numerical", "Prediction vs Numerical"), ("analytical", "Prediction vs Analytical")]:
        print(f"\n{title}\n")
        for name, value in metrics[label].items():
            print(f"{name}: {value:.6e}" if name != "R²" else f"{name}: {value:.6f}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _setup_plot_style() -> None:
    style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE)


def _mode_axis_limit(n_freq: int) -> int:
    """Limit Fourier mode axes to the low-frequency band used in training."""
    return min(n_freq, MAX_PLOT_MODES)


def plot_overlay(
    x: np.ndarray,
    prediction: np.ndarray,
    numerical: np.ndarray,
    analytical: np.ndarray,
    output_dir: Path,
    sample_index: int = ROLLOUT_PAIR_INDEX,
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
    pair_index: int = ROLLOUT_PAIR_INDEX,
) -> Path:
    """Save per-timestep rollout comparison images (analytical, numerical, prediction)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nTrain ICs: {', '.join(TRAIN_ICS)}")
    for ic_name in TRAIN_ICS:
        print(f"  {ic_name}: {find_validation_data(ic_name)}")
    print(f"Test ICs:  {', '.join(TEST_ICS)}")
    for ic_name in TEST_ICS:
        print(f"  {ic_name}: {find_validation_data(ic_name)}")

    train_data = load_combined_training_data(TRAIN_ICS)
    test_datasets = {ic_name: load_data(find_validation_data(ic_name)) for ic_name in TEST_ICS}

    train_inputs = train_data["inputs"]
    train_targets = train_data["targets"]

    print(f"\nCombined train pairs: {train_inputs.shape}")
    for ic_name, test_data in test_datasets.items():
        print(f"  {ic_name} test pairs: {test_data['inputs'].shape}")

    nx = train_inputs.shape[1]
    for ic_name, test_data in test_datasets.items():
        if test_data["inputs"].shape[1] != nx:
            raise ValueError(f"Grid mismatch for test IC '{ic_name}'")

    plot_modes = N_MODES if N_MODES <= nx // 2 + 1 else nx // 2 + 1
    dataset = TensorDataset(
        torch.from_numpy(train_inputs),
        torch.from_numpy(train_targets),
    )

    n_train = int(len(dataset) * TRAIN_FRACTION)
    n_val = len(dataset) - n_train
    train_set, _val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(_val_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    timing: dict[str, Any] = {}
    train_start = time.perf_counter()

    print(f"\nMode sweep: {MODE_SWEEP}")
    rel_errors_numerical: list[float] = []
    rel_errors_analytical: list[float] = []
    trained_modes: list[int] = []
    metrics_by_mode: dict[int, dict[str, dict[str, float]]] = {}
    rollout_model: nn.Module | None = None

    for n_modes in MODE_SWEEP:
        effective_modes = min(n_modes, nx // 2 + 1)
        if effective_modes != n_modes:
            print(f"  Skipping {n_modes} modes (grid supports at most {effective_modes})")
            continue

        print(f"\n--- Training with {n_modes} modes ---")
        torch.manual_seed(RANDOM_SEED)
        model = SpectralNet(nx=nx, n_modes=n_modes, use_linear=USE_LINEAR).to(device)

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            verbose=False,
        )

        test_rel_num: list[float] = []
        test_rel_ana: list[float] = []
        for ic_name, test_data in test_datasets.items():
            metrics, _ = evaluate(
                model,
                test_data["inputs"],
                test_data["targets"],
                test_data["analytical"],
                device,
            )
            test_rel_num.append(metrics["numerical"]["Relative L2"])
            test_rel_ana.append(metrics["analytical"]["Relative L2"])
            print(f"  [{ic_name}] relative L2 vs numerical:  {metrics['numerical']['Relative L2']:.6e}")
            print(f"  [{ic_name}] relative L2 vs analytical: {metrics['analytical']['Relative L2']:.6e}")

        rel_num = float(np.mean(test_rel_num))
        rel_ana = float(np.mean(test_rel_ana))
        trained_modes.append(n_modes)
        rel_errors_numerical.append(rel_num)
        rel_errors_analytical.append(rel_ana)
        metrics_by_mode[n_modes] = {
            "numerical": {"Relative L2": rel_num},
            "analytical": {"Relative L2": rel_ana},
        }
        if n_modes == plot_modes:
            rollout_model = model

        print(f"  Mean test relative L2 vs numerical:  {rel_num:.6e}")
        print(f"  Mean test relative L2 vs analytical: {rel_ana:.6e}")

    timing["surrogate_training"] = time.perf_counter() - train_start

    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = Path(__file__).resolve().parent / "outputs" / f"fno_demo_{timestamp}_{run_id}"

    plot_modes_vs_relative_error(
        modes=trained_modes,
        rel_errors_numerical=rel_errors_numerical,
        rel_errors_analytical=rel_errors_analytical,
        output_dir=output_dir,
    )

    if plot_modes not in metrics_by_mode:
        plot_modes = trained_modes[-1]
        rollout_model = SpectralNet(nx=nx, n_modes=plot_modes, use_linear=USE_LINEAR).to(device)
        extra_train_start = time.perf_counter()
        train(
            model=rollout_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            verbose=False,
        )
        timing["surrogate_training"] += time.perf_counter() - extra_train_start

    assert rollout_model is not None
    print(f"\nGenerating plots at {plot_modes} modes")

    inference_start = time.perf_counter()
    timing["numerical_solver_by_ic"] = {}

    for ic_name, test_data in test_datasets.items():
        test_inputs = test_data["inputs"]
        test_targets = test_data["targets"]
        test_analytical = test_data["analytical"]
        test_series = test_data["series"]
        x = test_data["x"]
        times_full = test_data["times_full"]

        metrics, sample_predictions = evaluate(
            rollout_model, test_inputs, test_targets, test_analytical, device
        )
        pair_index = ROLLOUT_PAIR_INDEX
        ic_output_dir = output_dir / ic_name
        alignment = verify_rollout_data_alignment(rollout_model, test_data, device, pair_index)
        print_rollout_alignment_report(alignment, ic_name)
        save_rollout_alignment_report(alignment, ic_output_dir)

        n_rollout_steps = len(test_series) - 1 - pair_index
        rollout_prediction = rollout(rollout_model, test_series[pair_index], n_rollout_steps, device)
        rollout_ground_truth = test_series[pair_index + 1 : pair_index + n_rollout_steps + 1]
        rollout_times_out = times_full[pair_index + 1 : pair_index + n_rollout_steps + 1]

        print(f"\n--- Test IC: {ic_name} ---")
        print_metrics(metrics, split_label=ic_name)
        one_step_metrics = compute_metrics(
            sample_predictions[pair_index : pair_index + 1],
            test_targets[pair_index : pair_index + 1],
        )
        print(f"  One-step @ pair {pair_index} (t_in={times_full[pair_index]:.4f}):"
              f" Relative L2 = {one_step_metrics['Relative L2']:.6e}")
        print(f"  Rollout step 1 rel L2:              {alignment['relative_l2']:.6e}")

        plot_overlay(
            x=x,
            prediction=sample_predictions,
            numerical=test_targets,
            analytical=test_analytical,
            output_dir=ic_output_dir,
            sample_index=pair_index,
            times_full=times_full,
        )
        plot_fourier_spectrum_evolution(
            series=test_series,
            times=times_full,
            output_dir=ic_output_dir,
        )
        plot_fourier_reconstruction(
            x=x,
            field=test_targets[pair_index],
            recon_modes=RECON_MODES,
            output_dir=ic_output_dir,
        )
        plot_spectral_coefficients(
            prediction=sample_predictions[pair_index],
            target=test_targets[pair_index],
            output_dir=ic_output_dir,
        )
        plot_rollout_comparison(
            x=x,
            step_times=rollout_times_out,
            ground_truth=rollout_ground_truth,
            prediction=rollout_prediction,
            output_dir=ic_output_dir,
        )
        frames_dir = save_rollout_frames(
            x=x,
            prediction=rollout_prediction,
            numerical=rollout_ground_truth,
            analytical=test_analytical[pair_index : pair_index + n_rollout_steps],
            times_full=times_full,
            output_dir=ic_output_dir,
            pair_index=pair_index,
        )
        video_path = ic_output_dir / f"rollout_{ic_name}.mp4"
        print(f"Generating rollout video for {ic_name}...")
        generate_rollout_video(frames_dir, video_path)
        print(f"Rollout video saved to: {video_path}")

    timing["surrogate_inference"] = time.perf_counter() - inference_start

    for ic_name in TEST_ICS:
        print(f"Benchmarking numerical solver for {ic_name}...")
        timing["numerical_solver_by_ic"][ic_name] = benchmark_numerical_solver(
            find_validation_data(ic_name)
        )
    timing["numerical_solver_total"] = sum(timing["numerical_solver_by_ic"].values())
    timing["surrogate_total"] = timing["surrogate_training"] + timing["surrogate_inference"]
    timing["speedup"] = timing["numerical_solver_total"] / timing["surrogate_total"]

    print_timing_comparison(timing)
    save_timing(timing, output_dir)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
