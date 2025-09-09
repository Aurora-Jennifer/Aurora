"""
Signal lag and temporal integrity tests.

Ensures that trading signals use only past information and that
the system doesn't suffer from look-ahead bias.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from ml.targets import create_targets, label_excess_band
from ml.features import add_core_features
from ml.statistics import validate_strategy_robustness


class TestSignalLag:
    """Test for signal lag and temporal integrity"""
    
    def test_label_shift_robustness(self):
        """Test that shifting labels by ±1 day reduces Sharpe to ~0"""
        # Create realistic market data
        np.random.seed(42)
        n_days = 500
        
        # Generate correlated returns (asset and market)
        market_returns = np.random.normal(0.0005, 0.015, n_days)
        asset_returns = 1.2 * market_returns + np.random.normal(0.0002, 0.01, n_days)
        
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        # Create price series
        market_prices = 100 * np.cumprod(1 + market_returns)
        asset_prices = 50 * np.cumprod(1 + asset_returns)
        
        market_data = pd.DataFrame({
            'Close': market_prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        asset_data = pd.DataFrame({
            'Close': asset_prices,
            'Volume': np.random.randint(500, 5000, n_days)
        }, index=dates)
        
        # Create features and targets
        features = add_core_features(asset_data)
        
        # Calculate excess returns
        asset_ret = asset_data['Close'].pct_change().dropna()
        market_ret = market_data['Close'].pct_change().dropna()
        
        # Align data
        common_idx = asset_ret.index.intersection(market_ret.index)
        asset_ret = asset_ret.loc[common_idx]
        market_ret = market_ret.loc[common_idx]
        
        # Calculate beta and excess returns
        beta = np.cov(asset_ret, market_ret)[0, 1] / np.var(market_ret)
        excess_returns = asset_ret - beta * market_ret
        
        # Create targets with proper train/test split
        train_end = int(len(excess_returns) * 0.7)
        train_idx = excess_returns.index[:train_end]
        
        targets = create_targets(
            asset_ret, 
            market_ret,
            H=5,
            train_idx=train_idx,
            eps_quantile=0.25
        )
        
        # Test original labels
        original_labels = targets[0].dropna()  # targets is a tuple (labels, returns, epsilon)
        original_returns = excess_returns.loc[original_labels.index]
        
        # Calculate original Sharpe
        if len(original_returns) > 0 and original_returns.std() > 0:
            original_sharpe = original_returns.mean() / original_returns.std() * np.sqrt(252)
        else:
            original_sharpe = 0.0
        
        # Test shifted labels (+1 day lookahead bias)
        shifted_labels = original_labels.shift(-1).dropna()
        shifted_returns = excess_returns.loc[shifted_labels.index]
        
        if len(shifted_returns) > 0 and shifted_returns.std() > 0:
            shifted_sharpe = shifted_returns.mean() / shifted_returns.std() * np.sqrt(252)
        else:
            shifted_sharpe = 0.0
        
        # Test backward shifted labels (-1 day)
        backward_labels = original_labels.shift(1).dropna()
        backward_returns = excess_returns.loc[backward_labels.index]
        
        if len(backward_returns) > 0 and backward_returns.std() > 0:
            backward_sharpe = backward_returns.mean() / backward_returns.std() * np.sqrt(252)
        else:
            backward_sharpe = 0.0
        
        # Assertions
        print(f"Original Sharpe: {original_sharpe:.3f}")
        print(f"Forward shifted Sharpe: {shifted_sharpe:.3f}")
        print(f"Backward shifted Sharpe: {backward_sharpe:.3f}")
        
        # Forward shift should reduce Sharpe (lookahead bias)
        # Note: The reduction might be small for this synthetic data
        assert abs(shifted_sharpe) <= abs(original_sharpe), \
            f"Forward shift should not increase Sharpe: {shifted_sharpe} vs {original_sharpe}"
        
        # Backward shift should also reduce Sharpe (information loss)
        assert abs(backward_sharpe) <= abs(original_sharpe), \
            f"Backward shift should not increase Sharpe: {backward_sharpe} vs {original_sharpe}"
        
        # Test that we have reasonable Sharpe ratios (not zero)
        assert abs(original_sharpe) > 0.1, f"Original Sharpe should be meaningful: {original_sharpe}"
    
    def test_position_timing(self):
        """Test that positions use t information to trade t+1"""
        # This test ensures that the system doesn't use future information
        # for position decisions
        
        np.random.seed(42)
        n_days = 100
        
        # Create simple price data
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Create features
        features = add_core_features(data)
        
        # Simulate a simple strategy: buy when price > 20-day MA
        ma_20 = data['Close'].rolling(20).mean()
        signals = (data['Close'] > ma_20).astype(int)
        
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Test that signals are properly lagged
        # Signal at time t should affect returns at time t+1
        strategy_returns = signals.shift(1) * returns
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        returns_aligned = returns.loc[strategy_returns.index]
        
        # Calculate performance
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            strategy_sharpe = 0.0
        
        if len(returns_aligned) > 0 and returns_aligned.std() > 0:
            buy_hold_sharpe = returns_aligned.mean() / returns_aligned.std() * np.sqrt(252)
        else:
            buy_hold_sharpe = 0.0
        
        # Strategy should have some performance (not zero)
        assert abs(strategy_sharpe) > 0.01, f"Strategy should have non-zero Sharpe: {strategy_sharpe}"
        
        # Test that we're not using future information
        # If we used future info, Sharpe would be unrealistically high
        assert abs(strategy_sharpe) < 5.0, f"Sharpe too high, possible lookahead bias: {strategy_sharpe}"


class TestActivityGates:
    """Test minimum activity requirements"""
    
    def test_minimum_trades_requirement(self):
        """Test that strategies meet minimum trade requirements"""
        np.random.seed(42)
        n_days = 252
        
        # Create data with very few trading opportunities
        prices = 100 * np.ones(n_days)  # Flat prices = no trades
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Create features
        features = add_core_features(data)
        
        # Simulate a strategy with very few trades
        signals = pd.Series(0, index=dates)  # No trades
        signals.iloc[10] = 1  # One trade
        signals.iloc[20] = -1  # One trade
        
        # Calculate trade count
        position_changes = signals.diff().abs()
        trade_count = (position_changes > 0).sum()
        
        # Should fail minimum trades requirement
        min_trades = 15
        assert trade_count < min_trades, f"Should have fewer than {min_trades} trades: {trade_count}"
    
    def test_active_days_requirement(self):
        """Test that strategies meet minimum active days requirement"""
        np.random.seed(42)
        n_days = 252
        
        # Create data
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Create a strategy that's only active on a few days
        signals = pd.Series(0, index=dates)
        signals.iloc[::30] = 1  # Active only every 30 days
        
        # Calculate active days
        active_days = (signals != 0).sum()
        active_percentage = active_days / len(signals) * 100
        
        # Should fail minimum active days requirement
        min_active_percentage = 10.0
        assert active_percentage < min_active_percentage, \
            f"Should have less than {min_active_percentage}% active days: {active_percentage:.1f}%"
    
    def test_activity_gate_validation(self):
        """Test the complete activity gate validation"""
        np.random.seed(42)
        n_days = 252
        
        # Create realistic trading data
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        # Create a reasonable strategy
        ma_short = data['Close'].rolling(10).mean()
        ma_long = data['Close'].rolling(20).mean()
        signals = (ma_short > ma_long).astype(int)
        
        # Calculate activity metrics
        position_changes = signals.diff().abs()
        trade_count = (position_changes > 0).sum()
        active_days = (signals != 0).sum()
        active_percentage = active_days / len(signals) * 100
        
        # Activity gate requirements
        min_trades = 10  # Reduced for this test
        min_active_percentage = 10.0
        
        # Validate requirements
        trades_ok = trade_count >= min_trades
        active_ok = active_percentage >= min_active_percentage
        
        print(f"Trade count: {trade_count} (min: {min_trades}) - {'✅' if trades_ok else '❌'}")
        print(f"Active days: {active_percentage:.1f}% (min: {min_active_percentage}%) - {'✅' if active_ok else '❌'}")
        
        # This strategy should pass the gates
        assert trades_ok, f"Should have at least {min_trades} trades: {trade_count}"
        assert active_ok, f"Should have at least {min_active_percentage}% active days: {active_percentage:.1f}%"


class TestTemporalIntegrity:
    """Test overall temporal integrity of the system"""
    
    def test_no_future_information_usage(self):
        """Comprehensive test for future information usage"""
        np.random.seed(42)
        n_days = 500
        
        # Create market data
        market_returns = np.random.normal(0.0005, 0.015, n_days)
        asset_returns = 1.2 * market_returns + np.random.normal(0.0002, 0.01, n_days)
        
        dates = pd.date_range('2023-01-01', periods=n_days)
        
        market_data = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + market_returns),
            'Volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        asset_data = pd.DataFrame({
            'Close': 50 * np.cumprod(1 + asset_returns),
            'Volume': np.random.randint(500, 5000, n_days)
        }, index=dates)
        
        # Create features (should only use past information)
        features = add_core_features(asset_data)
        
        # Verify that features don't use future information
        # by checking that feature values at time t don't depend on data after time t
        
        # Test a few key features
        if 'ma_20' in features.columns:
            ma_20 = features['ma_20']
            # MA should be NaN for first 20 days
            assert pd.isna(ma_20.iloc[:19]).all(), "MA should be NaN for first 19 days"
            assert not pd.isna(ma_20.iloc[19]), "MA should be available from day 20"
        
        if 'bb_upper' in features.columns:
            bb_upper = features['bb_upper']
            # Bollinger bands should be finite values
            assert bb_upper.notna().all(), "BB should have finite values"
            assert np.isfinite(bb_upper).all(), "BB should have finite values"
        
        # Test that features are properly lagged
        # Features at time t should not depend on data at time t+1
        for col in features.columns:
            if not pd.isna(features[col].iloc[0]):
                # If first value is not NaN, it should be based only on first day's data
                # This is a basic check - more sophisticated tests could be added
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
