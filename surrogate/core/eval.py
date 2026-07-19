from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROLLOUT_PAIR_INDEX = 0

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
    *,
    dx: float | None = None,
    mass_project: bool = False,
) -> np.ndarray:
    """Autoregressively predict the next n_steps fields from an initial state."""
    from ml.core.train import project_mass

    model.eval()
    trajectory = np.zeros((n_steps, initial_state.shape[-1]), dtype=np.float32)
    state = torch.from_numpy(initial_state).float().unsqueeze(0).to(device)
    ref_mass = None
    if mass_project and dx is not None:
        ref_mass = state.sum(dim=-1) * dx

    with torch.no_grad():
        for step in range(n_steps):
            state = model(state)
            if mass_project and dx is not None and ref_mass is not None:
                state = project_mass(state, ref_mass, dx)
            trajectory[step] = state.squeeze(0).cpu().numpy()

    return trajectory


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Global relative L2 error ||pred - ref|| / ||ref||."""
    diff = prediction - reference
    return float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-12))


def evaluate_rollout_horizons(
    model: nn.Module,
    series: np.ndarray,
    device: torch.device,
    horizons: list[int] | tuple[int, ...] = (1, 5, 10, 20),
    start_index: int = 0,
    *,
    dx: float | None = None,
    mass_project: bool = False,
) -> dict[int, float]:
    """
    Autoregressive Relative L2 at fixed horizons from series[start_index].

    Returns {horizon: relative_l2} for each requested horizon that fits in series.
    """
    max_available = len(series) - 1 - start_index
    usable = [h for h in horizons if 1 <= h <= max_available]
    if not usable:
        return {}

    max_h = max(usable)
    prediction = rollout(
        model,
        series[start_index],
        max_h,
        device,
        dx=dx,
        mass_project=mass_project,
    )
    ground_truth = series[start_index + 1 : start_index + 1 + max_h]
    return {h: relative_l2(prediction[:h], ground_truth[:h]) for h in usable}


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


def print_metrics(metrics: dict[str, dict[str, float]], split_label: str = "Test") -> None:
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 26)
    print(f"Evaluation Results ({split_label})")
    print("=" * 26)

    for label, title in [("numerical", "Prediction vs Numerical"), ("analytical", "Prediction vs Analytical")]:
        print(f"\n{title}\n")
        for name, value in metrics[label].items():
            print(f"{name}: {value:.6e}" if name != "R²" else f"{name}: {value:.6f}")
