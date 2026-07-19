from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.core.memory import (
    batch_size_for_unroll,
    format_memory_report,
    gpu_memory_stats,
    reset_peak_memory_stats,
)


def mass_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """MSE between discrete masses ∫u dx for pred and target ([B, N])."""
    mass_pred = pred.sum(dim=-1) * dx
    mass_tgt = target.sum(dim=-1) * dx
    return torch.mean((mass_pred - mass_tgt) ** 2)


def high_mode_energy(pred: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Mean |û(k)|² for Fourier modes with k >= n_keep."""
    ft = torch.fft.rfft(pred, dim=-1)
    if ft.size(-1) <= n_keep:
        return pred.new_zeros(())
    return (ft[..., n_keep:].abs() ** 2).mean()


def project_mass(field: torch.Tensor, ref_mass: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Add a spatially uniform correction so ∫field dx matches ref_mass.

    field: [B, N], ref_mass: [B]
    """
    n = field.shape[-1]
    current = field.sum(dim=-1) * dx
    correction = ((ref_mass - current) / (n * dx)).unsqueeze(-1)
    return field + correction


def _step_aux_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dx: float | None,
    mass_weight: float,
    high_mode_weight: float,
    high_mode_cutoff: int | None,
) -> torch.Tensor:
    aux = pred.new_zeros(())
    if mass_weight > 0.0 and dx is not None:
        aux = aux + mass_weight * mass_mse(pred, target, dx)
    if high_mode_weight > 0.0 and high_mode_cutoff is not None:
        aux = aux + high_mode_weight * high_mode_energy(pred, high_mode_cutoff)
    return aux


def unrolled_loss(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,
    *,
    teacher_forcing_prob: float = 0.0,
    dx: float | None = None,
    mass_weight: float = 0.0,
    high_mode_weight: float = 0.0,
    high_mode_cutoff: int | None = None,
    mass_project: bool = False,
) -> torch.Tensor:
    """
    One-step or unrolled multi-step loss with optional scheduled sampling.

    y_batch shapes:
      [B, nx]     — one-step target
      [B, S, nx]  — S-step teacher targets

    During training, with probability teacher_forcing_prob the next input is the
    ground-truth target (scheduled sampling); otherwise the prediction is used.
    Validation should pass teacher_forcing_prob=0 for fully autoregressive loss.
    """
    if y_batch.ndim == 2:
        pred = model(x_batch)
        if mass_project and dx is not None:
            ref = x_batch.sum(dim=-1) * dx
            pred = project_mass(pred, ref, dx)
        loss = criterion(pred, y_batch)
        loss = loss + _step_aux_loss(
            pred,
            y_batch,
            dx=dx,
            mass_weight=mass_weight,
            high_mode_weight=high_mode_weight,
            high_mode_cutoff=high_mode_cutoff,
        )
        return loss

    if y_batch.ndim != 3:
        raise ValueError(f"Expected y_batch ndim 2 or 3, got shape {tuple(y_batch.shape)}")

    state = x_batch
    loss = x_batch.new_zeros(())
    steps = y_batch.size(1)
    use_tf = model.training and teacher_forcing_prob > 0.0

    for step in range(steps):
        pred = model(state)
        if mass_project and dx is not None:
            ref = state.sum(dim=-1) * dx
            pred = project_mass(pred, ref, dx)
        target = y_batch[:, step]
        loss = loss + criterion(pred, target)
        loss = loss + _step_aux_loss(
            pred,
            target,
            dx=dx,
            mass_weight=mass_weight,
            high_mode_weight=high_mode_weight,
            high_mode_cutoff=high_mode_cutoff,
        )
        if use_tf and torch.rand((), device=pred.device) < teacher_forcing_prob:
            state = target
        else:
            state = pred

    return loss / steps


# Backward-compatible alias used by older tests.
def _batch_loss(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return unrolled_loss(model, x_batch, y_batch, criterion)


def teacher_forcing_prob_for_epoch(
    epoch: int,
    total_epochs: int,
    *,
    start: float = 1.0,
    end: float = 0.0,
) -> float:
    """Linear decay of teacher-forcing probability over epochs."""
    if total_epochs <= 1:
        return end
    t = (epoch - 1) / (total_epochs - 1)
    return float(start + (end - start) * t)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    verbose: bool = True,
    criterion: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    unroll_steps: int = 1,
    teacher_forcing_start: float = 0.0,
    teacher_forcing_end: float = 0.0,
    dx: float | None = None,
    mass_weight: float = 0.0,
    high_mode_weight: float = 0.0,
    high_mode_cutoff: int | None = None,
    mass_project: bool = False,
    epoch_offset: int = 0,
    total_epochs_for_schedule: int | None = None,
    use_amp: bool = False,
) -> list[dict[str, float]]:
    """
    Train with Adam + MSE by default.

    Supports unrolled multi-step targets, scheduled sampling, mass / high-mode
    losses, optional mass projection, and CUDA automatic mixed precision.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if unroll_steps < 1:
        raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")

    amp_enabled = bool(use_amp and device.type == "cuda")
    # GradScaler cannot unscale ComplexFloat gradients from SpectralConv weights.
    # Autocast still reduces activation memory for real-valued lift/local/GELU paths.

    schedule_epochs = total_epochs_for_schedule or epochs
    history: list[dict[str, float]] = []
    non_blocking = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + epoch
        tf_prob = teacher_forcing_prob_for_epoch(
            global_epoch,
            schedule_epochs,
            start=teacher_forcing_start,
            end=teacher_forcing_end,
        )

        model.train()
        train_loss = 0.0
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = unrolled_loss(
                    model,
                    x_batch,
                    y_batch,
                    criterion,
                    teacher_forcing_prob=tf_prob,
                    dx=dx,
                    mass_weight=mass_weight,
                    high_mode_weight=high_mode_weight,
                    high_mode_cutoff=high_mode_cutoff,
                    mass_project=mass_project,
                )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            n_train += x_batch.size(0)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=non_blocking)
                y_batch = y_batch.to(device, non_blocking=non_blocking)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = unrolled_loss(
                        model,
                        x_batch,
                        y_batch,
                        criterion,
                        teacher_forcing_prob=0.0,
                        dx=dx,
                        mass_weight=mass_weight,
                        high_mode_weight=high_mode_weight,
                        high_mode_cutoff=high_mode_cutoff,
                        mass_project=mass_project,
                    )
                val_loss += loss.item() * x_batch.size(0)
                n_val += x_batch.size(0)

        train_loss /= max(n_train, 1)
        val_loss /= max(n_val, 1)
        history.append(
            {
                "epoch": global_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "teacher_forcing_prob": tf_prob,
                "unroll_steps": float(unroll_steps),
            }
        )

        if verbose:
            print(
                f"Epoch {global_epoch:4d} | unroll={unroll_steps:3d} | "
                f"tf={tf_prob:.2f} | Training Loss: {train_loss:.6e} | "
                f"Validation Loss: {val_loss:.6e}"
            )

    return history


def train_curriculum(
    model: nn.Module,
    stages: Sequence[tuple[int, int]],
    build_loaders: Callable[..., tuple[DataLoader, DataLoader]],
    learning_rate: float,
    device: torch.device,
    *,
    verbose: bool = True,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.0,
    dx: float | None = None,
    mass_weight: float = 0.0,
    high_mode_weight: float = 0.0,
    high_mode_cutoff: int | None = None,
    mass_project: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    base_batch_size: int = 16,
    batch_size_schedule: Mapping[int, int] | None = None,
    batch_ref_unroll: int = 4,
    use_amp: bool = True,
    report_memory: bool = True,
    enable_checkpointing: bool = True,
) -> list[dict[str, float]]:
    """
    Curriculum unrolling: stages are (n_epochs, unroll_steps) pairs.

    Rebuilds loaders at each stage with an adaptive batch size. Optional AMP,
    FNO-block gradient checkpointing, and per-stage GPU memory diagnostics.
    """
    if not stages:
        raise ValueError("Curriculum stages must be non-empty")

    total_epochs = sum(n for n, _ in stages)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Enable checkpointing only during curriculum training when requested.
    checkpoint_restore: bool | None = None
    if enable_checkpointing and hasattr(model, "checkpoint_blocks"):
        checkpoint_restore = bool(model.checkpoint_blocks)
        model.checkpoint_blocks = True

    history: list[dict[str, float]] = []
    memory_reports: list[dict[str, Any]] = []
    epoch_offset = 0

    try:
        for n_epochs, unroll_steps in stages:
            batch_size = batch_size_for_unroll(
                unroll_steps,
                base_batch_size,
                schedule=batch_size_schedule,
                ref_unroll=batch_ref_unroll,
            )
            train_loader, val_loader = _call_build_loaders(
                build_loaders, unroll_steps, batch_size
            )

            if report_memory:
                reset_peak_memory_stats(device)

            if verbose:
                print(
                    f"\n=== Curriculum stage: {n_epochs} epochs @ "
                    f"unroll={unroll_steps}, batch={batch_size}, "
                    f"amp={use_amp and device.type == 'cuda'}, "
                    f"checkpoint={getattr(model, 'checkpoint_blocks', False)} ==="
                )

            stage_hist = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=n_epochs,
                learning_rate=learning_rate,
                device=device,
                verbose=verbose,
                optimizer=optimizer,
                unroll_steps=unroll_steps,
                teacher_forcing_start=teacher_forcing_start,
                teacher_forcing_end=teacher_forcing_end,
                dx=dx,
                mass_weight=mass_weight,
                high_mode_weight=high_mode_weight,
                high_mode_cutoff=high_mode_cutoff,
                mass_project=mass_project,
                epoch_offset=epoch_offset,
                total_epochs_for_schedule=total_epochs,
                use_amp=use_amp,
            )
            history.extend(stage_hist)
            epoch_offset += n_epochs

            if report_memory:
                stats = gpu_memory_stats(device)
                report = {
                    "unroll_steps": unroll_steps,
                    "batch_size": batch_size,
                    **stats,
                }
                memory_reports.append(report)
                print(format_memory_report(
                    unroll_steps=unroll_steps,
                    batch_size=batch_size,
                    stats=stats,
                ))
    finally:
        if checkpoint_restore is not None:
            model.checkpoint_blocks = checkpoint_restore

    # Per-stage memory is printed above when report_memory=True.
    # Do not setattr on a plain list (fails on modern Python).
    return history


def _call_build_loaders(
    build_loaders: Callable[..., tuple[DataLoader, DataLoader]],
    unroll_steps: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Support build_loaders(unroll) or build_loaders(unroll, batch_size)."""
    try:
        return build_loaders(unroll_steps, batch_size)
    except TypeError:
        return build_loaders(unroll_steps)
