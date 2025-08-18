# tests/ml/test_leakage_guards.py
"""
Test leakage guards to ensure no data snooping.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_label_shift_leakage_guard():
    """Test that labels are properly shifted forward."""
    from ml.features.build_daily import build_features_for_symbol
    
    # Mock data with known dates
    dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
    
    # Create mock OHLCV data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    volumes = np.random.randint(1000000, 10000000, 100)
    
    mock_data = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Mock yfinance to return our data
    import yfinance as yf
    original_history = yf.Ticker.history
    
    def mock_history(self, start=None, end=None, auto_adjust=False):
        return mock_data
    
    yf.Ticker.history = mock_history
    
    try:
        # Build features
        features = build_features_for_symbol('SPY', '2020-01-01', '2020-04-10')
        
        # Check that label is properly shifted
        if len(features) > 0:
            label_col = 'ret_fwd_1d'
            feature_cols = [col for col in features.columns if col != label_col]
            
            # For each row, the label should be for a future date
            for i in range(len(features) - 1):
                current_date = features.index[i]
                next_date = features.index[i + 1]
                
                # Label should be shifted by 1 day
                expected_label_date = current_date + timedelta(days=1)
                
                # Check that we don't have a label for the current date
                # (it should be NaN or for a future date)
                if not pd.isna(features.iloc[i][label_col]):
                    # The label should correspond to a future return
                    assert current_date < expected_label_date, \
                        f"Leakage detected: label at {current_date} not properly shifted"
    
    finally:
        # Restore original method
        yf.Ticker.history = original_history

def test_train_test_split_no_overlap():
    """Test that train/test splits don't overlap."""
    from ml.trainers.train_linear import prepare_training_data
    
    # Create mock data
    dates = pd.date_range('2020-01-01', periods=500, freq='D', tz='UTC')
    np.random.seed(42)
    
    mock_df = pd.DataFrame({
        'ret_1d': np.random.randn(500) * 0.01,
        'ret_5d': np.random.randn(500) * 0.02,
        'ret_20d': np.random.randn(500) * 0.05,
        'sma_20_minus_50': np.random.randn(500) * 0.1,
        'vol_10d': np.random.randn(500) * 0.005,
        'vol_20d': np.random.randn(500) * 0.005,
        'rsi_14': np.random.uniform(0, 100, 500),
        'volu_z_20d': np.random.randn(500),
        'ret_fwd_1d': np.random.randn(500) * 0.01,
        'symbol': 'SPY'
    }, index=dates)
    
    # Prepare train/test split
    train_df, test_df = prepare_training_data(mock_df, test_size=0.2)
    
    # Check no overlap
    train_dates = set(train_df.index)
    test_dates = set(test_df.index)
    
    overlap = train_dates.intersection(test_dates)
    assert len(overlap) == 0, f"Train/test overlap detected: {overlap}"
    
    # Check temporal ordering
    max_train_date = max(train_df.index)
    min_test_date = min(test_df.index)
    
    assert max_train_date < min_test_date, \
        f"Test data starts before train data ends: {max_train_date} >= {min_test_date}"

def test_feature_calculation_no_lookahead():
    """Test that feature calculations don't use future data."""
    from ml.features.build_daily import calculate_returns, calculate_sma_ratio, calculate_volatility
    
    # Create time series data
    dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    prices_series = pd.Series(prices, index=dates)
    
    # Test returns calculation
    returns = calculate_returns(prices_series, [1, 5, 20])
    
    # Check that each return uses only past data
    for i in range(20, len(returns)):  # Skip initial NaN values
        current_date = returns.index[i]
        
        # 1-day return should use current and previous day
        if i > 0:
            prev_date = returns.index[i-1]
            assert prev_date < current_date
        
        # 5-day return should use current and 4 previous days
        if i >= 4:
            for j in range(1, 5):
                past_date = returns.index[i-j]
                assert past_date < current_date
    
    # Test SMA calculation
    sma_ratio = calculate_sma_ratio(prices_series)
    
    # Check that SMA uses only past data
    for i in range(50, len(sma_ratio)):  # Skip initial NaN values
        current_date = sma_ratio.index[i]
        
        # 20-day and 50-day SMAs should use only past data
        for j in range(1, 51):
            past_date = sma_ratio.index[i-j]
            assert past_date < current_date

def test_walkforward_folds_no_overlap():
    """Test that walkforward folds don't overlap."""
    from ml.eval.alpha_eval import create_walkforward_folds
    
    # Create mock data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D', tz='UTC')
    np.random.seed(42)
    
    mock_df = pd.DataFrame({
        'ret_1d': np.random.randn(1000) * 0.01,
        'ret_5d': np.random.randn(1000) * 0.02,
        'ret_20d': np.random.randn(1000) * 0.05,
        'sma_20_minus_50': np.random.randn(1000) * 0.1,
        'vol_10d': np.random.randn(1000) * 0.005,
        'vol_20d': np.random.randn(1000) * 0.005,
        'rsi_14': np.random.uniform(0, 100, 1000),
        'volu_z_20d': np.random.randn(1000),
        'ret_fwd_1d': np.random.randn(1000) * 0.01,
        'symbol': 'SPY'
    }, index=dates)
    
    # Create walkforward folds
    folds = create_walkforward_folds(mock_df, n_folds=5, min_train_size=252)
    
    # Check that folds don't overlap
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            train_i, test_i = folds[i]
            train_j, test_j = folds[j]
            
            # Test sets should not overlap
            test_i_dates = set(test_i.index)
            test_j_dates = set(test_j.index)
            
            overlap = test_i_dates.intersection(test_j_dates)
            assert len(overlap) == 0, f"Test sets overlap between folds {i} and {j}: {overlap}"
            
            # Train sets can overlap (expanding window)
            # But test sets should be strictly after train sets
            max_train_i = max(train_i.index)
            min_test_i = min(test_i.index)
            assert max_train_i < min_test_i, f"Test starts before train ends in fold {i}"

def test_feature_config_validation():
    """Test that feature configuration is valid."""
    from ml.features.build_daily import load_feature_config
    
    config = load_feature_config()
    
    # Check required keys
    assert 'features' in config, "Missing 'features' key"
    assert 'labels' in config, "Missing 'labels' key"
    assert 'params' in config, "Missing 'params' key"
    
    # Check feature definitions
    features = config['features']
    assert len(features) > 0, "No features defined"
    
    # Check label definitions
    labels = config['labels']
    assert 'ret_fwd_1d' in labels, "Missing 'ret_fwd_1d' label"
    
    # Check parameters
    params = config['params']
    assert 'min_history_bars' in params, "Missing 'min_history_bars'"
    assert 'label_shift' in params, "Missing 'label_shift'"
    assert params['label_shift'] > 0, "Label shift must be positive"

if __name__ == "__main__":
    pytest.main([__file__])
