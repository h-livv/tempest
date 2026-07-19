from __future__ import annotations

import torch
import torch.nn as nn


class ChannelMLP(nn.Module):
    """Pointwise channel mixing via 1x1 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            act = activation if activation is not None else nn.GELU()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
                act,
                nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
