"""GPU memory helpers for curriculum / unrolled training."""

from __future__ import annotations

from typing import Mapping

import torch


def batch_size_for_unroll(
    unroll_steps: int,
    base_batch_size: int,
    *,
    schedule: Mapping[int, int] | None = None,
    ref_unroll: int = 4,
    min_batch_size: int = 1,
) -> int:
    """
    Choose a batch size that keeps activation memory roughly constant vs unroll.

    Priority:
      1. Exact match in `schedule` (e.g. {4: 16, 64: 2})
      2. Nearest lower key in `schedule` (for unroll between listed points)
      3. Inverse scaling: batch ≈ base * ref_unroll / unroll_steps
    """
    if unroll_steps < 1:
        raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")

    if schedule:
        if unroll_steps in schedule:
            return max(min_batch_size, int(schedule[unroll_steps]))
        lower = [k for k in schedule if k <= unroll_steps]
        if lower:
            return max(min_batch_size, int(schedule[max(lower)]))
        # Unroll shorter than any schedule key — use the smallest-unroll entry.
        return max(min_batch_size, int(schedule[min(schedule)]))

    scaled = int(round(base_batch_size * ref_unroll / float(unroll_steps)))
    return max(min_batch_size, scaled)


def gpu_memory_stats(device: torch.device) -> dict[str, float]:
    """Return allocated / reserved / peak memory in MiB (empty on CPU)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
    }


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def format_memory_report(
    *,
    unroll_steps: int,
    batch_size: int,
    stats: Mapping[str, float],
) -> str:
    if not stats:
        return (
            f"Memory @ unroll={unroll_steps}, batch={batch_size}: "
            "(CPU / CUDA stats unavailable)"
        )
    return (
        f"Memory @ unroll={unroll_steps}, batch={batch_size}: "
        f"alloc={stats['allocated_mb']:.1f} MiB, "
        f"reserved={stats['reserved_mb']:.1f} MiB, "
        f"peak={stats['peak_allocated_mb']:.1f} MiB"
    )
