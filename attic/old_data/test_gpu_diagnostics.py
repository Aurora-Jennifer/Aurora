#!/usr/bin/env python3
"""
GPU diagnostics test script
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.ml.diagnostics import diagnose_pytorch_step

# Set optimal DataLoader settings
NUM_WORKERS = max(4, (os.cpu_count() or 8) - 2)
PREFETCH = 4

print(f"Using {NUM_WORKERS} workers, prefetch={PREFETCH}")

# Create a simple model similar to our policy network
class TestModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Create synthetic data similar to our training data
batch_size = 256
input_dim = 128
num_samples = 1000

# Generate random data
states = torch.randn(num_samples, input_dim)
actions = torch.randint(0, 3, (num_samples,))
rewards = torch.randn(num_samples)

# Create dataset and dataloader with optimal settings
dataset = TensorDataset(states, actions, rewards)
loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=True,
    prefetch_factor=PREFETCH, 
    persistent_workers=True
)

# Create model
model = TestModel(input_dim, hidden_dim=256, output_dim=3)

# Enable fast kernels for RTX 3080
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Starting GPU diagnostics...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch size: {batch_size}")
print(f"DataLoader workers: {NUM_WORKERS}")

# Run diagnostics
res = diagnose_pytorch_step(model, loader, device="cuda", steps=120)

print("\n=== HARNESS RESULT ===")
for k, v in res.items():
    print(f"{k} => {v}")
