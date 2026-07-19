from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.core.equations import get_validation_dir

# Backward-compatible default (wave equation).
VALIDATION_DIR = get_validation_dir("wave")


def list_available_ics(equation: str = "wave") -> list[str]:
    """Discover initial conditions present under an equation's validation directory."""
    validation_dir = get_validation_dir(equation)
    if not validation_dir.exists():
        return []

    ics: set[str] = set()
    for config_path in sorted(validation_dir.glob("*/data/config.json")):
        with open(config_path) as f:
            ic_name = json.load(f).get("initial_condition")
        if ic_name:
            ics.add(ic_name)
    return sorted(ics)


def find_validation_data(
    initial_condition: str,
    equation: str = "wave",
) -> Path:
    """Return the latest validation spatial_data.npz for an exact initial condition name."""
    validation_dir = get_validation_dir(equation)
    matches: list[Path] = []
    for npz_path in sorted(validation_dir.glob("*/data/spatial_data.npz")):
        config_path = npz_path.parent / "config.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            ic_name = json.load(f).get("initial_condition")
        if ic_name == initial_condition:
            matches.append(npz_path)
    if not matches:
        available = ", ".join(list_available_ics(equation)) or "(none found)"
        raise FileNotFoundError(
            f"No validation data found for '{initial_condition}' under {validation_dir}. "
            f"Available ICs: {available}"
        )
    return matches[-1]


def load_combined_training_data(
    initial_conditions: list[str],
    equation: str = "wave",
) -> dict[str, Any]:
    """Load and concatenate one-step training pairs from multiple initial conditions."""
    datasets = [load_data(find_validation_data(ic, equation=equation)) for ic in initial_conditions]
    nx = datasets[0]["inputs"].shape[1]
    for data in datasets[1:]:
        if data["inputs"].shape[1] != nx:
            raise ValueError("All training datasets must share the same spatial resolution.")

    segment_lengths = [int(d["inputs"].shape[0]) for d in datasets]
    return {
        "inputs": np.concatenate([d["inputs"] for d in datasets], axis=0),
        "targets": np.concatenate([d["targets"] for d in datasets], axis=0),
        "analytical": np.concatenate([d["analytical"] for d in datasets], axis=0),
        "x": datasets[0]["x"],
        "sources": initial_conditions,
        "equation": equation,
        "segment_lengths": segment_lengths,
    }


def load_unrolled_training_data(
    initial_conditions: list[str],
    equation: str = "wave",
    unroll_steps: int = 1,
) -> dict[str, Any]:
    """
    Build contiguous trajectory windows for multi-step (unrolled) training.

    For each IC series of length T, creates windows:
        input  = u(t_i)
        target = [u(t_i+1), ..., u(t_i+unroll_steps)]

    Returns targets with shape (N, unroll_steps, nx).
    """
    if unroll_steps < 1:
        raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")

    datasets = [
        load_data(find_validation_data(ic, equation=equation))
        for ic in initial_conditions
    ]
    nx = datasets[0]["series"].shape[1]
    for data in datasets[1:]:
        if data["series"].shape[1] != nx:
            raise ValueError("All training datasets must share the same spatial resolution.")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    segment_lengths: list[int] = []

    for data in datasets:
        series = data["series"]
        n_times = series.shape[0]
        n_windows = n_times - unroll_steps
        if n_windows <= 0:
            raise ValueError(
                f"Series length {n_times} is too short for unroll_steps={unroll_steps}"
            )
        for i in range(n_windows):
            inputs.append(series[i])
            targets.append(series[i + 1 : i + 1 + unroll_steps])
        segment_lengths.append(n_windows)

    return {
        "inputs": np.stack(inputs, axis=0).astype(np.float32),
        "targets": np.stack(targets, axis=0).astype(np.float32),
        "x": datasets[0]["x"],
        "sources": initial_conditions,
        "equation": equation,
        "segment_lengths": segment_lengths,
        "unroll_steps": unroll_steps,
    }


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


def _estimate_burgers_shock_params(
    u0: np.ndarray,
    x_coords: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Estimate (u_L, u_R, shock_speed, x0, viscosity) from a 1D Burgers shock profile.

    Viscosity is inferred from the peak gradient of the tanh traveling-wave profile:
        max|du/dx| = (u_L - u_R)^2 / (8 ν)
    """
    u_L = float(u0[0])
    u_R = float(u0[-1])
    amp = u_L - u_R
    c_speed = 0.5 * (u_L + u_R)

    idx_left = np.where(u0 >= c_speed)[0]
    if len(idx_left) > 0 and len(idx_left) < len(u0):
        idx = int(idx_left[-1])
        idx_next = min(idx + 1, len(u0) - 1)
        denom = float(u0[idx_next] - u0[idx]) + 1e-12
        x0 = float(
            x_coords[idx]
            + (c_speed - float(u0[idx])) * float(x_coords[idx_next] - x_coords[idx]) / denom
        )
    else:
        x0 = 0.5 * float(x_coords[0] + x_coords[-1])

    dx = float(x_coords[1] - x_coords[0])
    max_slope = float(np.max(np.abs(np.gradient(u0, dx))))
    nu = (amp * amp) / (8.0 * max_slope + 1e-12)
    nu = max(nu, 1e-8)
    return u_L, u_R, c_speed, x0, nu


def _analytical_burgers_1d(
    u0: np.ndarray,
    x_coords: np.ndarray,
    times: np.ndarray,
    *,
    ic_name: str = "",
    boundary: str = "dirichlet",
    viscosity: float | None = None,
) -> np.ndarray:
    """
    Exact traveling / stationary viscous Burgers shock (tanh profile).

    Matches the construction in src/validation/validation.py.
    """
    u_L, u_R, c_speed, x0, nu_est = _estimate_burgers_shock_params(u0, x_coords)
    nu = float(viscosity) if viscosity is not None else nu_est
    dx = float(x_coords[1] - x_coords[0])
    domain_length = float(x_coords.max() + dx)
    amp = u_L - u_R
    analytical = np.zeros((len(times), len(x_coords)), dtype=np.float64)

    stationary = "stationary" in ic_name
    for i, t in enumerate(times):
        if stationary:
            # Stationary shock: u = -U tanh(U (x-x0) / (2ν)), U ≈ |u|_max
            u_amp = float(np.max(np.abs(u0)))
            analytical[i] = -u_amp * np.tanh(u_amp * (x_coords - x0) / (2.0 * nu))
            continue

        rel_x = x_coords - (x0 + c_speed * float(t))
        if boundary == "dirichlet":
            wrapped = rel_x
        else:
            wrapped = (rel_x + 0.5 * domain_length) % domain_length - 0.5 * domain_length
        analytical[i] = c_speed - 0.5 * amp * np.tanh((amp / (4.0 * nu)) * wrapped)

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
    ic_name = ""
    boundary = "periodic"
    viscosity: float | None = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        dt_step = float(config.get("dt", 1.0))
        record_interval = int(config.get("record_interval", 1))
        effective_dt = dt_step * record_interval
        equation_name = config.get("equation", "advection")
        wave_speed = float(config.get("wave_speed", 1.0))
        ic_name = str(config.get("initial_condition", ""))
        boundary = str(config.get("boundary_function", "periodic")).lower()
        if "viscosity" in config:
            viscosity = float(config["viscosity"])
    else:
        effective_dt = float(times[1] - times[0]) if len(times) > 1 else 1.0

    initial_field = ml_tensor[0] if ml_tensor.ndim == 3 else series[0]
    u0_1d = (
        initial_field[slice_index, :]
        if ml_tensor.ndim == 3 and slice_index is not None
        else np.asarray(initial_field)
    )

    if equation_name == "wave":
        analytical = _analytical_wave_1d(
            initial_field=u0_1d,
            x_coords=x_coords,
            times=target_times,
            wave_speed=wave_speed,
        )
    elif equation_name == "burgers":
        if "shock" in ic_name or "smooth" in ic_name or ic_name == "":
            analytical = _analytical_burgers_1d(
                u0=u0_1d,
                x_coords=x_coords,
                times=target_times,
                ic_name=ic_name,
                boundary=boundary,
                viscosity=viscosity,
            )
        else:
            # Generic Burgers IC (e.g. gaussian): no closed form — blank analytical.
            analytical = np.zeros_like(targets, dtype=np.float64)
    elif ml_tensor.ndim == 3 and y_coords is not None:
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
        "equation": equation_name,
        "data_path": str(data_path),
    }
