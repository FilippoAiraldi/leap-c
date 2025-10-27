from functools import partial

import numpy as np
import torch
from gymnasium import spaces

from leap_c.torch.nn.bounded_distributions import ScaledBeta, asymmetric_tanh_squash


def test_scaled_beta() -> None:
    """Sanity checks for the ScaledBeta distribution."""

    test_space = spaces.Box(
        low=np.array([-10.0, -15.0, 31.0, 3.0]), high=np.array([-5.0, 20.0, 42.0, 4.0])
    )
    dist = ScaledBeta(test_space)

    # Define parameters
    def create_alpha_beta_tensors():
        alpha = torch.tensor([[1.0, -2.0, -3.0, -100.0], [3.0, 4.0, -5.0, 100.0]])
        beta = torch.tensor([[4.0, 3.0, 2.0, -100.0], [2.0, -1.0, 0.0, 100.0]])
        alpha.requires_grad = True
        beta.requires_grad = True
        return alpha, beta

    alpha, beta = create_alpha_beta_tensors()
    samples, log_prob, _ = dist(alpha, beta, deterministic=False)

    # Check shapes
    assert samples.shape == (2, 4)
    assert log_prob.shape == (2, 1)

    # Check that samples are within bounds
    samples_npy = samples.detach().numpy()
    for s in samples_npy:
        assert s in test_space

    # test backward of log_prob works
    log_prob.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    alpha, beta = create_alpha_beta_tensors()
    samples, log_prob, _ = dist(alpha, beta, deterministic=False)
    # test backward of samples works
    samples.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    # Test deterministic sampling (mode)
    alpha, beta = create_alpha_beta_tensors()
    mode_samples, mode_log_prob, _ = dist(alpha, beta, deterministic=True)

    # Check that mode samples are within bounds and their shapes
    assert mode_samples.shape == (2, 4)
    assert mode_log_prob.shape == (2, 1)
    mode_samples_npy = mode_samples.detach().numpy()
    for s in mode_samples_npy:
        assert s in test_space

    # Test mode_sample backward works
    mode_samples.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))

    alpha, beta = create_alpha_beta_tensors()
    mode_samples, mode_log_prob, _ = dist(alpha, beta, deterministic=True)

    # Test mode_log_prob backward works
    mode_log_prob.sum().backward()
    assert alpha.grad is not None and not torch.any(torch.isnan(alpha.grad))
    assert beta.grad is not None and not torch.any(torch.isnan(beta.grad))


def test_asymmetric_tanh_squash() -> None:
    """Check the output of `asymmetric_tanh_squash` falls between bounds and maps zeroes to
    `default` values."""
    device = torch.device("cpu")
    dtype = torch.float64
    test_close = partial(torch.testing.assert_close, rtol=1e-5, atol=1e-5)

    ndim = torch.randint(1, 10, (1,)).item()
    low = torch.rand(ndim, dtype=dtype, device=device) * -20.0
    high = torch.rand(ndim, dtype=dtype, device=device) * 20.0 + 1.0
    default = torch.rand(ndim, dtype=dtype, device=device) * (high - low) + low

    # test within bounds
    n_samples = 1_000
    x = 100.0 * torch.randn(n_samples, ndim, dtype=dtype, device=device)
    y = asymmetric_tanh_squash(x, low, high, default)
    assert y.shape == x.shape
    assert y.isfinite().all()
    test_close((low - y).clamp_min(0).max().item(), 0.0)
    test_close((y - high).clamp_min(0).max().item(), 0.0)

    # test zero maps to default
    x_zero = torch.zeros(ndim, dtype=dtype, device=device)
    y_zero = asymmetric_tanh_squash(x_zero, low, high, default)
    test_close(y_zero, default)
