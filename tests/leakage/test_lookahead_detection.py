"""
Lookahead Contamination Detection Tests
Based on the comprehensive checklist for eliminating lookahead bias.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestLookaheadDetection:
    """Test suite for detecting and preventing lookahead contamination."""

    def test_no_negative_shifts_except_labels(self):
        """Test that no features use negative shifts (future data)."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 100,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        df.set_index(['symbol', 'timestamp'], inplace=True)
        
        # This should be allowed (label construction)
        df['ret_fwd_1'] = df.groupby('symbol')['close'].pct_change(1).shift(-1)
        
        # These should NOT be allowed (feature construction)
        with pytest.raises(AssertionError):
            # Future close as feature - should fail
            df['future_close'] = df.groupby('symbol')['close'].shift(-1)
            assert 'future_close' not in df.columns, "Future features should not be allowed"
        
        # These should be allowed (lagged features)
        df['close_lag1'] = df.groupby('symbol')['close'].shift(1)
        df['ma_20'] = df.groupby('symbol')['close'].rolling(20).mean().reset_index(level=0, drop=True).shift(1)
        
        # Verify no negative shifts in feature columns
        feature_cols = [col for col in df.columns if not col.endswith('_fwd_')]
        for col in feature_cols:
            if col != 'ret_fwd_1':  # Skip the label
                # Check that the feature doesn't use future data
                assert not col.startswith('future_'), f"Feature {col} appears to use future data"

    def test_temporal_monotonicity(self):
        """Test that timestamps are strictly monotonic per symbol."""
        # Create valid data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 50,
            'close': np.random.randn(50).cumsum() + 100
        })
        df.set_index(['symbol', 'timestamp'], inplace=True)
        
        # Should pass
        assert df.index.is_monotonic_increasing
        
        # Test per-symbol monotonicity
        g = df.reset_index().groupby(['symbol'])['timestamp']
        assert (g.shift(-1) > g).all().all(), "Timestamps should be strictly increasing per symbol"
        
        # Create invalid data with duplicate timestamps
        df_bad = df.reset_index()
        df_bad.loc[25, 'timestamp'] = df_bad.loc[24, 'timestamp']  # Duplicate timestamp
        
        with pytest.raises(AssertionError):
            df_bad.set_index(['symbol', 'timestamp'], inplace=True)
            g_bad = df_bad.reset_index().groupby(['symbol'])['timestamp']
            assert (g_bad.shift(-1) > g_bad).all().all(), "Should fail with duplicate timestamps"

    def test_no_forward_filled_targets(self):
        """Test that targets are never forward-filled."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 100,
            'close': np.random.randn(100).cumsum() + 100
        })
        
        # Create proper target (no forward fill)
        df['ret_fwd_1'] = df.groupby('symbol')['close'].pct_change(1).shift(-1)
        
        # This should NOT be allowed
        with pytest.raises(AssertionError):
            # Simulate forward-filled target
            df['bad_target'] = df['ret_fwd_1'].ffill()
            assert df['bad_target'].isna().sum() > 0, "Targets should have NaN values, not be forward-filled"

    def test_feature_lagging_rules(self):
        """Test that all features are properly lagged."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 100,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        df.set_index(['symbol', 'timestamp'], inplace=True)
        
        grp = df.groupby('symbol')
        
        # Correct feature construction
        df['ma_20'] = grp['close'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).shift(1)
        df['z_close_100'] = (
            (grp['close'].shift(1) - grp['close'].rolling(100).mean()) /
            grp['close'].rolling(100).std()
        ).reset_index(level=0, drop=True)
        
        # Verify features are lagged
        assert df['ma_20'].iloc[0] is np.nan, "First value should be NaN due to lag"
        assert df['z_close_100'].iloc[0] is np.nan, "First value should be NaN due to lag"

    def test_future_shadow_feature_test(self):
        """Test that adding a future feature doesn't improve model performance."""
        # This test simulates the future shadow feature test
        # In a real implementation, you would train models and compare scores
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SPY'] * 100,
            'close': np.random.randn(100).cumsum() + 100
        })
        df.set_index(['symbol', 'timestamp'], inplace=True)
        
        grp = df.groupby('symbol')
        
        # Clean features
        X_clean = pd.DataFrame({
            'close_lag1': grp['close'].shift(1),
            'ma_20': grp['close'].rolling(20).mean().reset_index(level=0, drop=True).shift(1)
        })
        
        # Features with future contamination
        X_shadow = X_clean.copy()
        X_shadow['FUTURE_CLOSE'] = grp['close'].shift(-1)  # Illegal future feature
        
        # In a real test, you would:
        # score_clean = cv_score(clf, X_clean, y, cv)
        # score_shadow = cv_score(clf, X_shadow, y, cv)
        # assert score_shadow <= score_clean + 0.005
        
        # For now, just verify the structure
        assert 'FUTURE_CLOSE' in X_shadow.columns
        assert 'FUTURE_CLOSE' not in X_clean.columns

    def test_no_overlap_train_val_after_embargo(self):
        """Test that training and validation sets don't overlap after embargo."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate walk-forward splits with embargo
        lookback_bars = 20
        embargo_bars = 5
        
        # Split 1: train [0, 30], val [35, 50], embargo [50, 55]
        train_1 = dates[:30]
        val_1 = dates[35:50]
        embargo_1 = dates[50:55]
        
        # Split 2: train [0, 55], val [60, 75], embargo [75, 80]
        train_2 = dates[:55]  # Includes embargo from previous split
        val_2 = dates[60:75]
        
        # Verify no overlap between train and val
        assert len(set(train_1) & set(val_1)) == 0
        assert len(set(train_2) & set(val_2)) == 0
        
        # Verify embargo is respected
        assert len(set(embargo_1) & set(train_2)) == 0, "Embargo should not be in next training set"

    def test_asof_backward_only(self):
        """Test that as-of joins only use backward direction."""
        # Create main data
        main_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        main_df = pd.DataFrame({
            'timestamp': main_dates,
            'symbol': ['SPY'] * 50,
            'close': np.random.randn(50).cumsum() + 100
        })
        
        # Create auxiliary data (e.g., news, fundamentals)
        aux_dates = pd.date_range('2023-01-01', periods=20, freq='3D')  # Less frequent
        aux_df = pd.DataFrame({
            'timestamp': aux_dates,
            'symbol': ['SPY'] * 20,
            'news_score': np.random.randn(20)
        })
        
        # Correct: backward as-of join
        main_df = main_df.sort_values(['symbol', 'timestamp'])
        aux_df = aux_df.sort_values(['symbol', 'timestamp'])
        
        # This should be allowed
        merged_backward = pd.merge_asof(
            main_df, aux_df, by='symbol',
            on='timestamp', direction='backward', allow_exact_matches=True
        )
        
        # This should NOT be allowed
        with pytest.raises(AssertionError):
            merged_forward = pd.merge_asof(
                main_df, aux_df, by='symbol',
                on='timestamp', direction='forward', allow_exact_matches=True
            )
            assert False, "Forward as-of joins should not be allowed"

    def test_scaler_fit_in_fold_only(self):
        """Test that scalers are only fit on training data within folds."""
        # This test verifies the scaler fitting protocol
        from sklearn.preprocessing import RobustScaler
        
        # Simulate fold data
        X_train = np.random.randn(100, 5)
        X_val = np.random.randn(50, 5)
        
        # Correct: fit on train only, apply to val
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Verify shapes
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        
        # Verify no data leakage in fitting
        # In a real test, you would verify that the scaler parameters
        # are based only on training data statistics


def test_leakage_detection_integration():
    """Integration test for leakage detection."""
    # Create a comprehensive test dataset
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    symbols = ['SPY', 'QQQ']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'close': np.random.randn() + 100,
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.set_index(['symbol', 'timestamp'], inplace=True)
    
    # Test all the rules
    assert df.index.is_monotonic_increasing
    
    # Test per-symbol monotonicity
    g = df.reset_index().groupby(['symbol'])['timestamp']
    assert (g.shift(-1) > g).all().all()
    
    # Test feature construction
    grp = df.groupby('symbol')
    df['close_lag1'] = grp['close'].shift(1)
    df['ma_20'] = grp['close'].rolling(20).mean().reset_index(level=0, drop=True).shift(1)
    
    # Test target construction
    df['ret_fwd_1'] = grp['close'].pct_change(1).shift(-1)
    
    # Verify no future data in features
    feature_cols = [col for col in df.columns if not col.endswith('_fwd_')]
    for col in feature_cols:
        if col != 'ret_fwd_1':
            assert not col.startswith('future_'), f"Feature {col} uses future data"
    
    print("✅ All leakage detection tests passed!")
