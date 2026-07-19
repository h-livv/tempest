from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from ml.core.benchmarks import benchmark_numerical_solver, print_timing_comparison, save_timing
from ml.core.data import find_validation_data, load_combined_training_data, load_data
from ml.core.device import setup_seed_and_device
from ml.core.equations import add_equation_cli_args, resolve_experiment_ics
from ml.core.eval import (
    ROLLOUT_PAIR_INDEX,
    compute_metrics,
    evaluate,
    evaluate_rollout_horizons,
    print_metrics,
    rollout,
    verify_rollout_data_alignment,
)
from ml.core.train import train, train_curriculum
from ml.diagnostics.plots import (
    generate_rollout_video,
    plot_fourier_reconstruction,
    plot_fourier_spectrum_evolution,
    plot_modes_vs_relative_error,
    plot_overlay,
    plot_rollout_comparison,
    plot_spectral_coefficients,
    save_rollout_frames,
)



def print_data_paths(
    train_ics: list[str],
    test_ics: list[str],
    equation: str = "wave",
) -> None:
    print(f"\nEquation: {equation}")
    print(f"Train ICs: {', '.join(train_ics)}")
    for ic_name in train_ics:
        print(f"  {ic_name}: {find_validation_data(ic_name, equation=equation)}")
    print(f"Test ICs:  {', '.join(test_ics)}")
    for ic_name in test_ics:
        print(f"  {ic_name}: {find_validation_data(ic_name, equation=equation)}")


def load_experiment_datasets(
    train_ics: list[str],
    test_ics: list[str],
    equation: str = "wave",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    train_data = load_combined_training_data(train_ics, equation=equation)
    test_datasets = {
        ic_name: load_data(find_validation_data(ic_name, equation=equation))
        for ic_name in test_ics
    }

    nx = train_data["inputs"].shape[1]
    for ic_name, test_data in test_datasets.items():
        if test_data["inputs"].shape[1] != nx:
            raise ValueError(f"Grid mismatch for test IC '{ic_name}'")

    print(f"\nCombined train pairs: {train_data['inputs'].shape}")
    for ic_name, test_data in test_datasets.items():
        print(f"  {ic_name} test pairs: {test_data['inputs'].shape}")

    return train_data, test_datasets, nx


def build_train_val_loaders(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    batch_size: int,
    train_fraction: float,
    seed: int,
    device: torch.device | None = None,
    split: str = "random",
    segment_lengths: list[int] | None = None,
) -> tuple[DataLoader, DataLoader, int, int]:
    """
    Build train/val loaders.

    split:
      "random"   — shuffled random_split (legacy; can leak adjacent times)
      "temporal" — contiguous prefix/suffix per trajectory segment
    """
    dataset = TensorDataset(
        torch.from_numpy(train_inputs),
        torch.from_numpy(train_targets),
    )
    if split == "temporal":
        train_set, val_set, n_train, n_val = _temporal_split(
            dataset,
            train_fraction=train_fraction,
            segment_lengths=segment_lengths,
        )
    elif split == "random":
        n_train = int(len(dataset) * train_fraction)
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'random' or 'temporal'.")

    use_cuda = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
    )
    print(f"Train samples: {n_train}, Validation samples: {n_val} (split={split})")
    return train_loader, val_loader, n_train, n_val


def _temporal_split(
    dataset: TensorDataset,
    train_fraction: float,
    segment_lengths: list[int] | None,
) -> tuple[Subset, Subset, int, int]:
    """Contiguous train/val split within each trajectory segment."""
    n_total = len(dataset)
    if segment_lengths is None:
        segment_lengths = [n_total]
    if sum(segment_lengths) != n_total:
        raise ValueError(
            f"segment_lengths sum {sum(segment_lengths)} != dataset length {n_total}"
        )

    train_indices: list[int] = []
    val_indices: list[int] = []
    offset = 0
    for length in segment_lengths:
        n_train_seg = int(length * train_fraction)
        if length >= 2:
            n_train_seg = min(max(n_train_seg, 1), length - 1)
        else:
            n_train_seg = length
        train_indices.extend(range(offset, offset + n_train_seg))
        val_indices.extend(range(offset + n_train_seg, offset + length))
        offset += length

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        len(train_indices),
        len(val_indices),
    )


def create_output_dir(experiment_dir: Path, prefix: str) -> Path:
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    output_dir = experiment_dir / "outputs" / f"{prefix}_{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate_test_ics(
    model: nn.Module,
    test_datasets: dict[str, dict[str, Any]],
    device: torch.device,
    rollout_horizons: list[int] | tuple[int, ...] | None = None,
) -> tuple[float, float]:
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
        if rollout_horizons:
            horizon_scores = evaluate_rollout_horizons(
                model,
                test_data["series"],
                device,
                horizons=rollout_horizons,
                start_index=ROLLOUT_PAIR_INDEX,
            )
            formatted = ", ".join(
                f"h={h}: {score:.6e}" for h, score in sorted(horizon_scores.items())
            )
            print(f"  [{ic_name}] rollout Relative L2 → {formatted}")
    return float(np.mean(test_rel_num)), float(np.mean(test_rel_ana))


def save_rollout_alignment_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "rollout_alignment.json", "w") as f:
        json.dump(report, f, indent=2)


def print_rollout_alignment_report(report: dict[str, Any], ic_name: str) -> None:
    print(
        f"\n  Rollout alignment ({ic_name}) @ pair index {report['pair_index']}"
        f" (t_in={report['t_in']:.4f} → t_out={report['t_out']:.4f}):"
    )
    print(f"    inputs[i] == series[i]:            {report['inputs_equal_series']}")
    print(f"    targets[i] == series[i+1]:        {report['targets_equal_series_next']}")
    print(
        f"    predict(inputs[i]) == rollout[0]: {report['single_step_equals_rollout_step0']}"
        f" (max |diff|={report['single_rollout_max_abs_diff']:.3e})"
    )
    print(f"    one-step / rollout-step-1 rel L2:  {report['relative_l2']:.6e}")


def run_ic_diagnostics(
    model: nn.Module,
    test_datasets: dict[str, dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    recon_modes: list[int],
    *,
    dx: float | None = None,
    mass_project: bool = False,
) -> None:
    pair_index = ROLLOUT_PAIR_INDEX
    for ic_name, test_data in test_datasets.items():
        test_inputs = test_data["inputs"]
        test_targets = test_data["targets"]
        test_analytical = test_data["analytical"]
        test_series = test_data["series"]
        x = test_data["x"]
        times_full = test_data["times_full"]
        ic_dx = dx if dx is not None else float(x[1] - x[0])

        metrics, sample_predictions = evaluate(
            model, test_inputs, test_targets, test_analytical, device
        )
        ic_output_dir = output_dir / ic_name
        alignment = verify_rollout_data_alignment(model, test_data, device, pair_index)
        print_rollout_alignment_report(alignment, ic_name)
        save_rollout_alignment_report(alignment, ic_output_dir)

        n_rollout_steps = len(test_series) - 1 - pair_index
        rollout_prediction = rollout(
            model,
            test_series[pair_index],
            n_rollout_steps,
            device,
            dx=ic_dx,
            mass_project=mass_project,
        )
        rollout_ground_truth = test_series[pair_index + 1 : pair_index + n_rollout_steps + 1]
        rollout_times_out = times_full[pair_index + 1 : pair_index + n_rollout_steps + 1]

        print(f"\n--- Test IC: {ic_name} ---")
        print_metrics(metrics, split_label=ic_name)
        one_step_metrics = compute_metrics(
            sample_predictions[pair_index : pair_index + 1],
            test_targets[pair_index : pair_index + 1],
        )
        print(
            f"  One-step @ pair {pair_index} (t_in={times_full[pair_index]:.4f}):"
            f" Relative L2 = {one_step_metrics['Relative L2']:.6e}"
        )
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
            recon_modes=recon_modes,
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


def benchmark_all_ics(
    test_ics: list[str],
    timing: dict[str, Any],
    train_ics: list[str],
    output_dir: Path,
    equation: str = "wave",
) -> None:
    timing["numerical_solver_by_ic"] = {}
    for ic_name in test_ics:
        print(f"Benchmarking numerical solver for {ic_name}...")
        timing["numerical_solver_by_ic"][ic_name] = benchmark_numerical_solver(
            find_validation_data(ic_name, equation=equation)
        )
    timing["numerical_solver_total"] = sum(timing["numerical_solver_by_ic"].values())
    timing["surrogate_total"] = timing["surrogate_training"] + timing["surrogate_inference"]
    timing["speedup"] = timing["numerical_solver_total"] / timing["surrogate_total"]
    print_timing_comparison(timing, train_ics)
    save_timing(timing, output_dir, train_ics, test_ics, equation=equation)


def run_mode_sweep(
    mode_sweep: list[int],
    nx: int,
    build_model_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_datasets: dict[str, dict[str, Any]],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    seed: int,
    plot_modes: int,
    unroll_steps: int = 1,
    rollout_horizons: list[int] | tuple[int, ...] | None = None,
    *,
    curriculum: list[tuple[int, int]] | None = None,
    build_loaders=None,
    train_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[int],
    list[float],
    list[float],
    dict[int, dict[str, dict[str, float]]],
    nn.Module | None,
]:
    """
    Train one model per mode count.

    If `curriculum` and `build_loaders` are provided, uses curriculum unrolling
    (ignoring the fixed train_loader/val_loader for training). Otherwise trains
    with the provided loaders for `epochs` at fixed `unroll_steps`.
    """
    rel_errors_numerical: list[float] = []
    rel_errors_analytical: list[float] = []
    trained_modes: list[int] = []
    metrics_by_mode: dict[int, dict[str, dict[str, float]]] = {}
    rollout_model: nn.Module | None = None
    extra = dict(train_kwargs or {})

    print(f"\nMode sweep: {mode_sweep}")
    if curriculum:
        print(f"Curriculum stages (epochs, unroll): {curriculum}")
    elif unroll_steps > 1:
        print(f"Unrolled training steps: {unroll_steps}")

    for n_modes in mode_sweep:
        effective_modes = min(n_modes, nx // 2 + 1)
        if effective_modes != n_modes:
            print(f"  Skipping {n_modes} modes (grid supports at most {effective_modes})")
            continue

        print(f"\n--- Training with {n_modes} modes ---")
        torch.manual_seed(seed)
        model = build_model_fn(n_modes).to(device)

        stage_kwargs = dict(extra)
        # Prefer per-model high-mode cutoff matching this sweep entry.
        if "high_mode_cutoff" not in stage_kwargs:
            stage_kwargs["high_mode_cutoff"] = n_modes

        if curriculum is not None:
            if build_loaders is None:
                raise ValueError("curriculum requires build_loaders(unroll_steps)")
            train_curriculum(
                model=model,
                stages=curriculum,
                build_loaders=build_loaders,
                learning_rate=learning_rate,
                device=device,
                verbose=False,
                **stage_kwargs,
            )
        else:
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                verbose=False,
                unroll_steps=unroll_steps,
                **stage_kwargs,
            )

        rel_num, rel_ana = evaluate_test_ics(
            model,
            test_datasets,
            device,
            rollout_horizons=rollout_horizons,
        )
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

    return (
        trained_modes,
        rel_errors_numerical,
        rel_errors_analytical,
        metrics_by_mode,
        rollout_model,
    )
