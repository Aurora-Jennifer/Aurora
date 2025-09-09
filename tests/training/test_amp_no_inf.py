"""
Mixed precision (AMP) tests
"""
import pytest
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_amp_scaler_prevents_inf():
    """Test that AMP scaler prevents infinite gradients"""
    m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)).cuda()
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    scaler = GradScaler()
    X = torch.randn(512, 128, device="cuda")
    y = torch.randn(512, 1, device="cuda")
    loss_fn = nn.MSELoss()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        with autocast():
            loss = loss_fn(m(X), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # grads must be finite after unscale
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        scaler.step(opt)
        scaler.update()
