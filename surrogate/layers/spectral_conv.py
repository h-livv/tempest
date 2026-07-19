from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: rfft -> truncated mode multiply -> irfft.

    Input/output shape: [B, C_in, N] -> [B, C_out, N].
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.weights = nn.Parameter(
            _complex_xavier(in_channels, out_channels, n_modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex FFT / einsum is not implemented for ComplexHalf under AMP.
        # Keep the spectral path in float32 while outer layers may use float16.
        input_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            batch_size, _, n_points = x32.shape
            x_ft = torch.fft.rfft(x32, dim=-1)
            n_freq = x_ft.size(-1)
            n_active = min(self.n_modes, n_freq)

            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                n_freq,
                dtype=torch.cfloat,
                device=x32.device,
            )
            weights = self.weights
            if weights.dtype != torch.cfloat:
                weights = weights.to(torch.cfloat)
            out_ft[:, :, :n_active] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :n_active],
                weights[:, :, :n_active],
            )
            out = torch.fft.irfft(out_ft, n=n_points, dim=-1)
        return out.to(dtype=input_dtype)


def _complex_xavier(in_channels: int, out_channels: int, n_modes: int) -> torch.Tensor:
    """
    Complex Xavier init so the spectral branch starts near the local 1x1 scale.

    Real and imag are drawn independently from N(0, σ²) with
    σ = 1 / sqrt(2 * in_channels), matching the channel-sum variance of a
    unit-gain linear map. This replaces the previous 1/(C_in·C_out) scaling,
    which left ‖K(x)‖ ~100× smaller than ‖W(x)‖ at initialization.
    """
    std = 1.0 / math.sqrt(2.0 * in_channels)
    real = torch.randn(in_channels, out_channels, n_modes) * std
    imag = torch.randn(in_channels, out_channels, n_modes) * std
    return torch.complex(real, imag)
