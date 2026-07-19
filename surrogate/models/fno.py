from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ml.layers.fno_block import FNOBlock
from ml.layers.mlp import ChannelMLP


class FNO1d(nn.Module):
    """Thin orchestrator: lift -> FNO blocks -> project."""

    def __init__(
        self,
        n_modes: int = 16,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        residual: bool = False,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.width = width
        self.n_layers = n_layers
        self.residual = residual
        self.checkpoint_blocks = checkpoint_blocks
        self.lift = ChannelMLP(in_channels, width)
        self.blocks = nn.ModuleList(FNOBlock(width, n_modes) for _ in range(n_layers))
        self.project = ChannelMLP(width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N]
        residual_in = x
        h = x.unsqueeze(1)
        h = self.lift(h)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                # Recompute block activations during backward to save memory.
                # use_reentrant=False is required for complex spectral weights.
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        out = self.project(h).squeeze(1)
        if self.residual:
            return residual_in + out
        return out
