import torch
import torch.nn as nn

class SpectralNet(nn.Module):
    """
    Minimal spectral neural network.

    Input -> FFT -> keep K modes -> complex multiply -> zero high modes
         -> inverse FFT -> (optional linear) -> output
    """

    def __init__(self, nx: int, n_modes: int, use_linear: bool = True) -> None:
        super().__init__()
        self.nx = nx
        self.n_modes = min(n_modes, nx // 2 + 1)
        self.n_freq = nx // 2 + 1

        # Learnable complex weights for the lowest K Fourier modes
        weight = torch.randn(self.n_modes, dtype=torch.cfloat) * 0.02
        self.spectral_weights = nn.Parameter(weight)

        self.linear: nn.Linear | None
        if use_linear:
            self.linear = nn.Linear(nx, nx)
        else:
            self.linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, nx) real spatial field
        x_ft = torch.fft.rfft(x, dim=-1)

        # Keep only the first K modes; zero out the rest
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, : self.n_modes] = x_ft[:, : self.n_modes] * self.spectral_weights

        out = torch.fft.irfft(out_ft, n=self.nx, dim=-1)

        if self.linear is not None:
            out = self.linear(out)

        return out
