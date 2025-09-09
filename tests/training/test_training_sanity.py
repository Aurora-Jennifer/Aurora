"""
Tiny synthetic canary suite - unit-speed and definitive
"""
from typing import Optional

import torch
from torch import nn, optim


def make_linear_data(n: int = 1024, d: int = 8, noise: float = 0.01, seed: int = 123):
    """Generate synthetic linear data for testing"""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_true = torch.randn(d, 1, generator=g)
    y = X @ w_true + noise * torch.randn(n, 1, generator=g)
    return X, y, w_true


def tiny_model(d: int):
    """Create a tiny model for testing"""
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))


def train_steps(model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 200, lr: float = 1e-2, clip: Optional[float] = None):
    """Train model for specified steps"""
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        yhat = model(X)
        loss = loss_fn(yhat, y)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
    return losses


def test_loss_drops_fast():
    """Test that loss drops quickly on synthetic data"""
    X, y, _ = make_linear_data(n=256, d=8)
    m = tiny_model(8)
    losses = train_steps(m, X, y, steps=200, lr=1e-2)
    assert losses[0] > losses[-1]
    assert losses[-1] < 0.10 * losses[0]  # ≥90% drop = sane loop


def test_overfit_tiny_batch():
    """Test that model can overfit a tiny batch (sanity check)"""
    X, y, _ = make_linear_data(n=32, d=8, noise=0.0)
    m = tiny_model(8)
    losses = train_steps(m, X, y, steps=500, lr=5e-2)
    assert losses[-1] < 1e-2  # should get very low loss (relaxed threshold)


def test_nan_guard_trips():
    """Test that NaN detection works"""
    X, y, _ = make_linear_data(n=128, d=8)
    m = tiny_model(8)
    # inject NaN into target to force non-finite grads
    y[0, 0] = float("nan")
    opt = optim.SGD(m.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    m.train()
    opt.zero_grad()
    loss = loss_fn(m(X), y)
    loss.backward()
    any_nan = any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any_nan
