"""
DataLoader determinism and shuffling tests
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

# from aurora.training.guards import worker_init_fn  # unused


def iter_batches(dl):
    """Get first 2 rows from each batch"""
    return [b[0][:2].clone() for b in dl]


def test_dataloader_reproducible_order():
    """Test that DataLoader produces reproducible order"""
    X = torch.arange(0, 100).float().unsqueeze(1)
    y = X.clone()
    ds = TensorDataset(X, y)
    g = torch.Generator().manual_seed(42)
    dl1 = DataLoader(
        ds, batch_size=10, shuffle=True, generator=g, num_workers=0  # single-threaded for speed
    )
    g2 = torch.Generator().manual_seed(42)
    dl2 = DataLoader(
        ds, batch_size=10, shuffle=True, generator=g2, num_workers=0  # single-threaded for speed
    )
    
    batches1 = iter_batches(dl1)
    batches2 = iter_batches(dl2)
    
    # Compare tensors properly
    assert len(batches1) == len(batches2)
    for b1, b2 in zip(batches1, batches2):
        assert torch.allclose(b1, b2)
