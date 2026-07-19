from __future__ import annotations

import torch
import torch.nn as nn

from ml.layers.spectral_conv import SpectralConv1d


class FNOBlock(nn.Module):
    """One FNO block: σ(spectral(x) + local(x)), matching the original paper."""

    def __init__(self, width: int, n_modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, n_modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Paper update: v ← σ(K(v) + W(v)). No outer identity skip.
        return self.activation(self.spectral(x) + self.local(x))
