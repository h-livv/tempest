"""
End-to-end 1D Fourier Neural Operator experiment.

Trains FNO1d with curriculum unrolling, scheduled sampling, mass / high-mode
penalties, residual head, adaptive batch sizes, AMP, and gradient checkpointing.
See docs/fno_rollout_instability.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.core.data import load_unrolled_training_data
from ml.core.equations import get_equation_spec
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
from ml.core.memory import batch_size_for_unroll
from ml.core.train import train_curriculum
from ml.diagnostics.plots import plot_modes_vs_relative_error
from ml.models.registry import build_model

# ---------------------------------------------------------------------------
# Configuration — edit these lists to choose equation and ICs
# ---------------------------------------------------------------------------

EQUATION = "burgers"

TRAIN_ICS = ["burgers_traveling_shock"]
TEST_ICS = [
    "burgers_traveling_shock",
]

WIDTH = 32
N_LAYERS = 4
N_MODES = 64
MODE_SWEEP = [32]

# Smoother curriculum (epochs, unroll). Larger horizons still supported via config.
UNROLL_CURRICULUM = [
    (30, 4),
    (30, 8),
    (30, 16),
    (30, 32),
]

# Adaptive batch sizes by unroll length (keeps activation memory ~flat).
# Unlisted lengths fall back to inverse scaling from BASE_BATCH_SIZE @ ref=4.
BATCH_SIZE_SCHEDULE: dict[int, int] = {
    4: 16,
    8: 8,
    16: 8,
    32: 4,
    64: 2,
    128: 1,
}
BASE_BATCH_SIZE = 16
BATCH_REF_UNROLL = 4

# Memory / precision knobs (do not change the mathematical objective).
USE_AMP = True
CHECKPOINT_BLOCKS = True
REPORT_MEMORY = True

# Scheduled sampling: TF prob decays 1 → 0 over the full curriculum.
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.0
MASS_WEIGHT = 1e-2
MASS_PROJECT = True
HIGH_MODE_WEIGHT = 1e-4
RESIDUAL = True

TRAIN_VAL_SPLIT = "temporal"
ROLLOUT_VAL_HORIZONS = [1, 5, 10, 20, 50, 100]
EPOCHS = sum(n for n, _ in UNROLL_CURRICULUM)
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
RECON_MODES = [8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a 1D Fourier Neural Operator surrogate.",
    )
    add_equation_cli_args(parser, default_equation=EQUATION)
    parser.add_argument("--epochs", type=int, default=None, help="Unused when curriculum is set")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BASE_BATCH_SIZE,
        help="Base batch size for adaptive schedule (default: BASE_BATCH_SIZE)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual Δu head",
    )
    parser.add_argument(
        "--no-mass-project",
        action="store_true",
        help="Disable hard mass projection after each step",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable FNO-block gradient checkpointing",
    )
    parser.add_argument(
        "--no-memory-report",
        action="store_true",
        help="Disable per-stage GPU memory diagnostics",
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
    eq_spec = get_equation_spec(equation)
    use_residual = RESIDUAL and not args.no_residual
    # Dirichlet Burgers (and similar) do not conserve ∫u — skip hard mass projection.
    mass_project = (
        MASS_PROJECT and not args.no_mass_project and eq_spec.conserves_mass
    )
    mass_weight = MASS_WEIGHT if eq_spec.conserves_mass else 0.0
    use_amp = USE_AMP and not args.no_amp
    checkpoint_blocks = CHECKPOINT_BLOCKS and not args.no_checkpoint
    report_memory = REPORT_MEMORY and not args.no_memory_report
    base_batch = args.batch_size
    if MASS_PROJECT and not eq_spec.conserves_mass:
        print(
            f"Note: equation '{equation}' does not conserve mass "
            f"({eq_spec.description}); disabling mass projection / mass loss."
        )

    device = setup_seed_and_device(RANDOM_SEED, prefer_cuda=not args.cpu)
    print_data_paths(train_ics, test_ics, equation=equation)

    _, test_datasets, nx = load_experiment_datasets(
        train_ics,
        test_ics,
        equation=equation,
    )
    sample_x = next(iter(test_datasets.values()))["x"]
    dx = float(sample_x[1] - sample_x[0])

    def build_loaders(unroll_steps: int, batch_size: int | None = None):
        bs = batch_size if batch_size is not None else batch_size_for_unroll(
            unroll_steps,
            base_batch,
            schedule=BATCH_SIZE_SCHEDULE,
            ref_unroll=BATCH_REF_UNROLL,
        )
        windows = load_unrolled_training_data(
            train_ics,
            equation=equation,
            unroll_steps=unroll_steps,
        )
        print(
            f"  Windows @ unroll={unroll_steps}, batch={bs}: "
            f"{windows['inputs'].shape[0]} (segments={windows['segment_lengths']})"
        )
        return build_train_val_loaders(
            windows["inputs"],
            windows["targets"],
            bs,
            TRAIN_FRACTION,
            RANDOM_SEED,
            device=device,
            split=TRAIN_VAL_SPLIT,
            segment_lengths=windows["segment_lengths"],
        )[:2]

    train_loader, val_loader = build_loaders(UNROLL_CURRICULUM[0][1])

    plot_modes = N_MODES if N_MODES <= nx // 2 + 1 else nx // 2 + 1
    timing: dict[str, Any] = {}
    train_start = time.perf_counter()

    def build_fno(n_modes: int) -> nn.Module:
        return build_model(
            "fno",
            n_modes=n_modes,
            width=WIDTH,
            n_layers=N_LAYERS,
            residual=use_residual,
            checkpoint_blocks=False,  # enabled inside train_curriculum when requested
        )

    print(
        f"\nStability / memory: residual={use_residual}, mass_weight={mass_weight}, "
        f"mass_project={mass_project}, high_mode_weight={HIGH_MODE_WEIGHT}, "
        f"tf={TEACHER_FORCING_START}→{TEACHER_FORCING_END}, "
        f"amp={use_amp}, checkpoint={checkpoint_blocks}, "
        f"batch_schedule={BATCH_SIZE_SCHEDULE}"
    )

    trained_modes, rel_num, rel_ana, metrics_by_mode, rollout_model = run_mode_sweep(
        mode_sweep=MODE_SWEEP,
        nx=nx,
        build_model_fn=build_fno,
        train_loader=train_loader,
        val_loader=val_loader,
        test_datasets=test_datasets,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
        plot_modes=plot_modes,
        unroll_steps=UNROLL_CURRICULUM[0][1],
        rollout_horizons=ROLLOUT_VAL_HORIZONS,
        curriculum=UNROLL_CURRICULUM,
        build_loaders=build_loaders,
        train_kwargs={
            "teacher_forcing_start": TEACHER_FORCING_START,
            "teacher_forcing_end": TEACHER_FORCING_END,
            "dx": dx,
            "mass_weight": mass_weight,
            "high_mode_weight": HIGH_MODE_WEIGHT,
            "mass_project": mass_project,
            "base_batch_size": base_batch,
            "batch_size_schedule": BATCH_SIZE_SCHEDULE,
            "batch_ref_unroll": BATCH_REF_UNROLL,
            "use_amp": use_amp,
            "report_memory": report_memory,
            "enable_checkpointing": checkpoint_blocks,
        },
    )
    timing["surrogate_training"] = time.perf_counter() - train_start

    output_dir = create_output_dir(Path(__file__).resolve().parent, "fno")
    plot_modes_vs_relative_error(
        modes=trained_modes,
        rel_errors_numerical=rel_num,
        rel_errors_analytical=rel_ana,
        output_dir=output_dir,
    )

    if plot_modes not in metrics_by_mode:
        plot_modes = trained_modes[-1]
        torch.manual_seed(RANDOM_SEED)
        rollout_model = build_fno(plot_modes).to(device)
        extra_train_start = time.perf_counter()
        train_curriculum(
            model=rollout_model,
            stages=UNROLL_CURRICULUM,
            build_loaders=build_loaders,
            learning_rate=LEARNING_RATE,
            device=device,
            verbose=False,
            teacher_forcing_start=TEACHER_FORCING_START,
            teacher_forcing_end=TEACHER_FORCING_END,
            dx=dx,
            mass_weight=mass_weight,
            high_mode_weight=HIGH_MODE_WEIGHT,
            high_mode_cutoff=plot_modes,
            mass_project=mass_project,
            base_batch_size=base_batch,
            batch_size_schedule=BATCH_SIZE_SCHEDULE,
            batch_ref_unroll=BATCH_REF_UNROLL,
            use_amp=use_amp,
            report_memory=report_memory,
            enable_checkpointing=checkpoint_blocks,
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
        dx=dx,
        mass_project=mass_project,
    )
    timing["surrogate_inference"] = time.perf_counter() - inference_start

    benchmark_all_ics(test_ics, timing, train_ics, output_dir, equation=equation)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
