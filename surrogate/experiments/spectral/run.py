"""
Minimal spectral surrogate demo for one-step PDE field prediction.

This script trains a small spectral network to map u(x, t) -> u(x, t + dt)
and compares predictions against numerical and analytical ground truth.

Educational focus: FFT -> learn in Fourier space -> inverse FFT.
Not a full FNO implementation from the paper.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ml.core.experiment import (
    add_equation_cli_args,
    benchmark_all_ics,
    build_train_val_loaders,
    create_output_dir,
    load_experiment_datasets,
    print_data_paths,
    resolve_experiment_ics,
    run_ic_diagnostics,
    run_mode_sweep,
    setup_seed_and_device,
)
from ml.core.train import train
from ml.diagnostics.plots import plot_modes_vs_relative_error
from ml.models.registry import build_model

# ---------------------------------------------------------------------------
# Configuration — edit these lists to choose equation and ICs
# ---------------------------------------------------------------------------

EQUATION = "wave"

TRAIN_ICS = ["gaussian", "square", "sine_wave"]
TEST_ICS = [
    "gaussian",
    "square",
    "double_gaussian",
    "shifted_gaussian",
    "sine_wave",
]

N_MODES = 256
MODE_SWEEP = [2, 4, 8, 16, 32, 64]
USE_LINEAR = True
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
RECON_MODES = [2, 4, 8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a spectral surrogate for one-step PDE prediction.",
    )
    add_equation_cli_args(parser, default_equation=EQUATION)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    equation, train_ics, test_ics = resolve_experiment_ics(
        args.equation,
        args.train_ics,
        args.test_ics,
        default_train_ics=TRAIN_ICS,
        default_test_ics=TEST_ICS,
    )

    device = setup_seed_and_device(RANDOM_SEED, prefer_cuda=not args.cpu)
    print_data_paths(train_ics, test_ics, equation=equation)
    train_data, test_datasets, nx = load_experiment_datasets(
        train_ics,
        test_ics,
        equation=equation,
    )

    plot_modes = N_MODES if N_MODES <= nx // 2 + 1 else nx // 2 + 1
    train_loader, val_loader, _, _ = build_train_val_loaders(
        train_data["inputs"],
        train_data["targets"],
        args.batch_size,
        TRAIN_FRACTION,
        RANDOM_SEED,
        device=device,
    )

    timing: dict[str, Any] = {}
    train_start = time.perf_counter()

    def build_spectral(n_modes: int):
        return build_model(
            "spectral",
            nx=nx,
            n_modes=n_modes,
            use_linear=USE_LINEAR,
        )

    (
        trained_modes,
        rel_errors_numerical,
        rel_errors_analytical,
        metrics_by_mode,
        rollout_model,
    ) = run_mode_sweep(
        MODE_SWEEP,
        nx,
        build_spectral,
        train_loader,
        val_loader,
        test_datasets,
        device,
        args.epochs,
        LEARNING_RATE,
        RANDOM_SEED,
        plot_modes,
    )

    timing["surrogate_training"] = time.perf_counter() - train_start

    output_dir = create_output_dir(Path(__file__).resolve().parent, "spectral")
    plot_modes_vs_relative_error(
        modes=trained_modes,
        rel_errors_numerical=rel_errors_numerical,
        rel_errors_analytical=rel_errors_analytical,
        output_dir=output_dir,
    )

    if plot_modes not in metrics_by_mode:
        plot_modes = trained_modes[-1]
        rollout_model = build_spectral(plot_modes).to(device)
        extra_train_start = time.perf_counter()
        train(
            model=rollout_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=LEARNING_RATE,
            device=device,
            verbose=False,
        )
        timing["surrogate_training"] += time.perf_counter() - extra_train_start

    assert rollout_model is not None
    print(f"\nGenerating plots at {plot_modes} modes")

    inference_start = time.perf_counter()
    run_ic_diagnostics(
        rollout_model,
        test_datasets,
        device,
        output_dir,
        RECON_MODES,
    )
    timing["surrogate_inference"] = time.perf_counter() - inference_start

    benchmark_all_ics(test_ics, timing, train_ics, output_dir, equation=equation)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
