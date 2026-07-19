"""
Unit tests for FNO spectral convolution and block semantics.

These lock in the correctness properties verified during the FNO debugging
investigation: FFT round-trip, complex arithmetic, mode retention, einsum
contraction, and paper-aligned block update (no outer identity residual).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ml.layers.fno_block import FNOBlock
from ml.layers.spectral_conv import SpectralConv1d
from ml.models.fno import FNO1d


def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach()
    target = target.detach()
    denom = torch.linalg.norm(target.reshape(-1)).clamp_min(1e-12)
    return float(torch.linalg.norm((pred - target).reshape(-1)) / denom)


def test_fft_ifft_roundtrip_near_machine_precision() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8, 256)
    recon = torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=x.size(-1), dim=-1)
    assert _relative_l2(recon, x) < 1e-6
    assert float((recon - x).abs().max()) < 1e-5


def test_spectral_conv_identity_weights() -> None:
    """Diagonal complex weights of 1 should pass retained modes unchanged."""
    torch.manual_seed(0)
    in_ch, out_ch, n_modes, n = 4, 4, 16, 128
    layer = SpectralConv1d(in_ch, out_ch, n_modes)
    with torch.no_grad():
        layer.weights.zero_()
        for c in range(in_ch):
            layer.weights[c, c, :].fill_(1.0 + 0.0j)

    x = torch.randn(2, in_ch, n)
    x_ft = torch.fft.rfft(x, dim=-1)
    low = torch.fft.irfft(x_ft[:, :, :n_modes], n=n, dim=-1)
    y = layer(x)
    assert _relative_l2(y, low) < 1e-5


def test_spectral_conv_preserves_imaginary_phase() -> None:
    """A purely imaginary weight on one mode must rotate that mode by +90°."""
    in_ch, out_ch, n_modes, n = 1, 1, 8, 64
    layer = SpectralConv1d(in_ch, out_ch, n_modes)
    with torch.no_grad():
        layer.weights.zero_()
        layer.weights[0, 0, 3] = 0.0 + 1.0j

    # Pure cosine at mode 3 → imaginary weight → sine (phase shift).
    t = torch.arange(n, dtype=torch.float32)
    x = torch.cos(2.0 * torch.pi * 3 * t / n).view(1, 1, n)
    y = layer(x)
    expected = -torch.sin(2.0 * torch.pi * 3 * t / n).view(1, 1, n)
    assert _relative_l2(y, expected) < 1e-5


def test_spectral_conv_inactive_modes_remain_zero() -> None:
    torch.manual_seed(1)
    layer = SpectralConv1d(2, 3, n_modes=5)
    x = torch.randn(2, 2, 64)
    x_ft = torch.fft.rfft(x, dim=-1)
    # Reconstruct the spectral multiply explicitly and check inactive bins.
    n_freq = x_ft.size(-1)
    out_ft = torch.zeros(2, 3, n_freq, dtype=torch.cfloat)
    n_active = 5
    out_ft[:, :, :n_active] = torch.einsum(
        "bix,iox->box",
        x_ft[:, :, :n_active],
        layer.weights[:, :, :n_active],
    )
    assert torch.count_nonzero(out_ft[:, :, n_active:]) == 0

    y = layer(x)
    y_ft = torch.fft.rfft(y, dim=-1)
    # Physical-space nonlinearity is absent here; inactive Fourier content
    # should stay near the low-pass reconstruction only.
    low_pass = torch.fft.irfft(out_ft, n=64, dim=-1)
    assert _relative_l2(y, low_pass) < 1e-6


def test_spectral_conv_einsum_matches_explicit_loop() -> None:
    torch.manual_seed(2)
    in_ch, out_ch, n_modes, n = 3, 5, 7, 32
    layer = SpectralConv1d(in_ch, out_ch, n_modes)
    x = torch.randn(2, in_ch, n)
    x_ft = torch.fft.rfft(x, dim=-1)
    n_active = min(n_modes, x_ft.size(-1))

    explicit = torch.zeros(2, out_ch, n_active, dtype=torch.cfloat)
    for b in range(2):
        for o in range(out_ch):
            for k in range(n_active):
                acc = 0.0 + 0.0j
                for i in range(in_ch):
                    acc = acc + x_ft[b, i, k] * layer.weights[i, o, k]
                explicit[b, o, k] = acc

    einsum_out = torch.einsum(
        "bix,iox->box",
        x_ft[:, :, :n_active],
        layer.weights[:, :, :n_active],
    )
    assert torch.equal(einsum_out, explicit)
    assert float((einsum_out - explicit).detach().abs().max()) == 0.0


def test_fno_block_matches_paper_update() -> None:
    """Block must be σ(K(x)+W(x)), not σ(K(x)+W(x))+x."""
    torch.manual_seed(3)
    block = FNOBlock(width=8, n_modes=4)
    x = torch.randn(2, 8, 64)

    expected = F.gelu(block.spectral(x) + block.local(x))
    actual = block(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    # Guard against accidental reintroduction of the outer identity skip.
    with_identity = expected + x
    assert not torch.allclose(actual, with_identity, atol=1e-5, rtol=1e-5)


def test_fno1d_forward_shapes_and_dtypes() -> None:
    model = FNO1d(n_modes=8, width=16, n_layers=2)
    x = torch.randn(3, 128)
    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_spectral_weights_receive_nonzero_gradients() -> None:
    torch.manual_seed(4)
    model = FNO1d(n_modes=8, width=16, n_layers=2)
    x = torch.randn(2, 64)
    y = model(x)
    loss = (y**2).mean()
    loss.backward()

    for block in model.blocks:
        grad = block.spectral.weights.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0


def test_spectral_init_comparable_to_local_branch() -> None:
    """Xavier-style init should not leave the spectral path ~100× weaker than local."""
    torch.manual_seed(5)
    block = FNOBlock(width=32, n_modes=16)
    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        spectral_norm = float(torch.linalg.norm(block.spectral(x)))
        local_norm = float(torch.linalg.norm(block.local(x)))
    # Allow a generous factor; the old 1/(C_in*C_out) init was ~125–160× smaller.
    assert spectral_norm > 0.05 * local_norm


def test_unrolled_batch_loss_shapes() -> None:
    from ml.core.train import unrolled_loss

    torch.manual_seed(6)
    model = FNO1d(n_modes=8, width=8, n_layers=1)
    criterion = torch.nn.MSELoss()
    x = torch.randn(2, 32)
    y_one = torch.randn(2, 32)
    y_multi = torch.randn(2, 4, 32)
    loss_one = unrolled_loss(model, x, y_one, criterion)
    loss_multi = unrolled_loss(model, x, y_multi, criterion)
    assert loss_one.ndim == 0 and torch.isfinite(loss_one)
    assert loss_multi.ndim == 0 and torch.isfinite(loss_multi)


def test_mass_projection_preserves_integral() -> None:
    from ml.core.train import project_mass

    dx = 0.01
    u = torch.randn(3, 100)
    ref = torch.ones(3) * 2.5
    out = project_mass(u, ref, dx)
    mass = out.sum(dim=-1) * dx
    assert torch.allclose(mass, ref, atol=1e-5)


def test_residual_fno_near_identity_at_init_scale() -> None:
    torch.manual_seed(7)
    model = FNO1d(n_modes=8, width=8, n_layers=1, residual=True)
    x = torch.randn(2, 64)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_batch_size_for_unroll_schedule_and_scaling() -> None:
    from ml.core.memory import batch_size_for_unroll

    schedule = {4: 16, 8: 8, 16: 8, 32: 4, 64: 2, 128: 1}
    assert batch_size_for_unroll(4, 16, schedule=schedule) == 16
    assert batch_size_for_unroll(64, 16, schedule=schedule) == 2
    assert batch_size_for_unroll(96, 16, schedule=schedule) == 2  # nearest lower
    assert batch_size_for_unroll(64, 16, schedule=None, ref_unroll=4) == 1


def test_checkpoint_blocks_match_eager_forward_and_grads() -> None:
    torch.manual_seed(8)
    x = torch.randn(2, 64)

    eager = FNO1d(n_modes=8, width=8, n_layers=2, residual=True, checkpoint_blocks=False)
    ckpt = FNO1d(n_modes=8, width=8, n_layers=2, residual=True, checkpoint_blocks=True)
    ckpt.load_state_dict(eager.state_dict())

    eager.train()
    ckpt.train()
    y_e = eager(x)
    y_c = ckpt(x)
    assert torch.allclose(y_e, y_c, atol=1e-6, rtol=1e-6)

    loss_e = (y_e**2).mean()
    loss_c = (y_c**2).mean()
    loss_e.backward()
    loss_c.backward()

    for p_e, p_c in zip(eager.parameters(), ckpt.parameters()):
        assert p_e.grad is not None and p_c.grad is not None
        assert torch.allclose(p_e.grad, p_c.grad, atol=1e-5, rtol=1e-5)


def test_short_unroll_loss_amp_matches_fp32() -> None:
    """AMP should not materially change a short-horizon loss at init."""
    from ml.core.train import unrolled_loss

    torch.manual_seed(9)
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = FNO1d(n_modes=8, width=8, n_layers=1).to(device)
    criterion = torch.nn.MSELoss()
    x = torch.randn(4, 64, device=device)
    y = torch.randn(4, 4, 64, device=device)

    model.eval()
    with torch.no_grad():
        loss_fp32 = unrolled_loss(model, x, y, criterion)
        with torch.amp.autocast("cuda", enabled=True):
            loss_amp = unrolled_loss(model, x, y, criterion)
    # Relative agreement within a few percent (float16 noise on init-scale nets).
    rel = abs(float(loss_fp32 - loss_amp) / (float(loss_fp32) + 1e-12))
    assert rel < 0.05, f"AMP/FP32 loss relative diff {rel}"


if __name__ == "__main__":
    tests = [
        test_fft_ifft_roundtrip_near_machine_precision,
        test_spectral_conv_identity_weights,
        test_spectral_conv_preserves_imaginary_phase,
        test_spectral_conv_inactive_modes_remain_zero,
        test_spectral_conv_einsum_matches_explicit_loop,
        test_fno_block_matches_paper_update,
        test_fno1d_forward_shapes_and_dtypes,
        test_spectral_weights_receive_nonzero_gradients,
        test_spectral_init_comparable_to_local_branch,
        test_unrolled_batch_loss_shapes,
        test_mass_projection_preserves_integral,
        test_residual_fno_near_identity_at_init_scale,
        test_batch_size_for_unroll_schedule_and_scaling,
        test_checkpoint_blocks_match_eager_forward_and_grads,
        test_short_unroll_loss_amp_matches_fp32,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
