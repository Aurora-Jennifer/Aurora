#!/usr/bin/env python3
"""Golden Crypto Snapshot for CI"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

GOLDEN_SEED = 42

def create_golden_crypto_data() -> pd.DataFrame:
    """Create deterministic crypto data for testing."""
    np.random.seed(GOLDEN_SEED)
    
    # 50 samples 
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(50)]
    
    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'timestamp': ts,
            'symbol': 'BTCUSDT',
            'Open': 50000.0 + i * 10,
            'High': 50100.0 + i * 10,
            'Low': 49900.0 + i * 10,
            'Close': 50050.0 + i * 10,
            'Volume': 1000.0,
            'returns_1h': 0.001,
            'volatility_24h': 0.02,
            'volume_ratio': 1.0,
        })
    
    return pd.DataFrame(data).set_index('timestamp')


def create_golden_crypto_features():
    """Create deterministic feature matrix for golden snapshot."""
    np.random.seed(GOLDEN_SEED)
    
    # Create feature matrix (50 samples, 5 features)
    n_samples = 50
    
    X = pd.DataFrame({
        'volatility_24h': np.full(n_samples, 0.02),
        'volume_ratio': np.full(n_samples, 1.0),
        'returns_lag1': np.full(n_samples, 0.001),
        'returns_lag2': np.full(n_samples, 0.001),
        'momentum_5': np.full(n_samples, 0.0),
    })
    
    # Create target
    y = pd.Series(np.full(n_samples, 0.001), name='target_1h')
    
    return {'X': X, 'y': y}


def get_golden_crypto_metrics():
    """Get expected crypto model evaluation metrics."""
    return {
        'information_coefficient': {
            'value': 0.15,
            'p_value': 0.04,
            'is_significant': True,
            'method': 'spearman',
            'n_samples': 50
        },
        'hit_rate': {
            'value': 0.60,
            'n_samples': 50,
            'n_correct': 30
        },
        'quality_score': {
            'value': 0.45
        }
    }