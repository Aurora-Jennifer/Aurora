#!/usr/bin/env python3
"""
Simple DL pipeline test
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Simple test without the complex trainer
def test_dl_pipeline():
    print("🧪 Testing simple DL pipeline...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 500
    features = pd.DataFrame(
        np.random.randn(n_samples, 28),
        index=pd.date_range('2023-01-01', periods=n_samples)
    )
    targets = pd.Series(
        np.random.choice([-1, 0, 1], n_samples),
        index=features.index
    )
    
    print(f"Data shape: {features.shape}, targets: {targets.shape}")
    
    # Test MLP model
    from dl.models import TinyMLP
    
    mlp = TinyMLP(input_dim=28, hidden_dims=[64, 32])
    print(f"MLP params: {mlp.get_param_count():,}")
    
    # Test forward pass
    X = torch.FloatTensor(features.values)
    y = torch.LongTensor(targets.values + 1)  # Convert to 0, 1, 2
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Forward pass
    logits = mlp(X)
    print(f"Logits shape: {logits.shape}")
    
    # Test loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    print(f"Loss: {loss.item():.4f}")
    
    # Test probability prediction
    probs = mlp.predict_proba(X)
    print(f"Probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(dim=1)[:5]}")
    
    # Test TCN model
    from dl.models import CompactTCN
    
    tcn = CompactTCN(input_dim=28, sequence_length=128)
    print(f"TCN params: {tcn.get_param_count():,}")
    
    # Create sequence data
    X_seq = torch.FloatTensor(features.values[:128]).unsqueeze(0)  # (1, seq_len, features)
    print(f"X_seq shape: {X_seq.shape}")
    
    # Forward pass
    logits_seq = tcn(X_seq)
    print(f"TCN logits shape: {logits_seq.shape}")
    
    # Test loss
    y_seq = torch.LongTensor([targets.values[127] + 1])  # Single target
    loss_seq = loss_fn(logits_seq, y_seq)
    print(f"TCN loss: {loss_seq.item():.4f}")
    
    print("✅ Simple DL pipeline working correctly!")
    return True

if __name__ == "__main__":
    test_dl_pipeline()
