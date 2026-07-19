"""Centralized PyTorch device selection and diagnostics."""

from __future__ import annotations

import warnings

import torch


def get_device(*, prefer_cuda: bool = True, verbose: bool = True) -> torch.device:
    """
    Return the best available compute device.

    Uses CUDA when available and preferred; otherwise falls back to CPU with a warning.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            _print_cuda_info(device)
        return device

    if prefer_cuda and verbose:
        warnings.warn(
            "CUDA is not available; falling back to CPU.",
            UserWarning,
            stacklevel=2,
        )
        print("Using device: cpu")
    elif verbose:
        print("Using device: cpu")

    return torch.device("cpu")


def _print_cuda_info(device: torch.device) -> None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(index) / (1024**3)
        print(
            f"Using device: cuda:{index} ({name}, "
            f"{total_gb:.1f} GB total, "
            f"{allocated_gb:.2f} GB allocated, "
            f"{reserved_gb:.2f} GB reserved)"
        )
    else:
        print(f"Using device: cuda:{index} ({name}, {total_gb:.1f} GB total)")


def setup_seed_and_device(seed: int, *, prefer_cuda: bool = True) -> torch.device:
    """Set random seeds and return the active compute device."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    import numpy as np

    np.random.seed(seed)
    return get_device(prefer_cuda=prefer_cuda)
