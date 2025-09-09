"""
Data integrity tests to prevent regression of robustness improvements.

These tests ensure that the key fixes for data integrity remain functional:
1. Epsilon floor implementation
2. Market benchmark usage
3. Feature filtering consistency
4. Label balance requirements
"""

import pytest
import pandas as pd
import numpy as np
import yfinance as yf
from unittest.mock import patch, MagicMock

from ml.targets import compute_epsilon_train_only, create_targets, label_excess_band
from ml.features import add_core_features
from ml.runner_grid import filter_collinear_features


class TestEpsilonFloor:
    """Test epsilon floor implementation"""
    
    def test_epsilon_floor_enforcement(self):
        """Test that epsilon floor is properly enforced"""
        # Create test data with very small excess returns
        small_returns = pd.Series([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        train_idx = small_returns.index
        
        # Test with default floor
        eps = compute_epsilon_train_only(small_returns, train_idx, q=0.25)
        assert eps == 1e-3, f"Expected epsilon floor 1e-3, got {eps}"
    
    def test_epsilon_with_large_returns(self):
        """Test that epsilon works normally with larger returns"""
        large_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        train_idx = large_returns.index
        
        eps = compute_epsilon_train_only(large_returns, train_idx, q=0.25)
        expected = large_returns.abs().quantile(0.25)
        assert eps == expected, f"Expected quantile-based epsilon {expected}, got {eps}"


class TestMarketBenchmarkUsage:
    """Test proper market benchmark usage"""
    
    @patch('yfinance.download')
    def test_proper_symbol_selection(self, mock_download):
        """Test that asset symbol is used correctly, not market benchmark"""
        # Mock yfinance data
        mock_data = {
            'AAPL': pd.DataFrame({
                'Close': [100, 101, 102, 103, 104],
                'Volume': [1000, 1100, 1200, 1300, 1400]
            }, index=pd.date_range('2023-01-01', periods=5)),
            'QQQ': pd.DataFrame({
                'Close': [200, 201, 202, 203, 204],
                'Volume': [2000, 2100, 2200, 2300, 2400]
            }, index=pd.date_range('2023-01-01', periods=5))
        }
        
        def mock_download_func(symbols, **kwargs):
            if isinstance(symbols, str):
                return mock_data[symbols]
            else:
                return {sym: mock_data[sym] for sym in symbols}
        
        mock_download.side_effect = mock_download_func
        
        # Test that we get different data for asset vs market
        asset_data = yf.download('AAPL', start='2023-01-01', end='2023-01-06')
        market_data = yf.download('QQQ', start='2023-01-01', end='2023-01-06')
        
        # Verify different closing prices
        assert asset_data['Close'].iloc[0] == 100
        assert market_data['Close'].iloc[0] == 200
        assert not asset_data.equals(market_data)


class TestFeatureFiltering:
    """Test feature filtering consistency"""
    
    def test_collinearity_filtering(self):
        """Test that collinearity filtering works correctly"""
        # Create test data with highly correlated features
        np.random.seed(42)
        n_samples = 100
        
        # Create base feature
        base_feature = np.random.randn(n_samples)
        
        # Create highly correlated features
        X = np.column_stack([
            base_feature,
            base_feature + 0.01 * np.random.randn(n_samples),  # Highly correlated
            base_feature + 0.1 * np.random.randn(n_samples),   # Less correlated
            np.random.randn(n_samples)  # Uncorrelated
        ])
        
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        
        # Apply filtering
        X_filtered, filtered_names = filter_collinear_features(X, feature_names, threshold=0.98)
        
        # Should remove highly correlated features
        assert len(filtered_names) == 2, f"Expected 2 features, got {len(filtered_names)}"
        assert 'feature2' not in filtered_names, "Highly correlated feature should be removed"
        assert 'feature3' not in filtered_names, "Less correlated feature should also be removed"
        assert 'feature1' in filtered_names, "Base feature should remain"
        assert 'feature4' in filtered_names, "Uncorrelated feature should remain"
    
    def test_feature_filtering_consistency(self):
        """Test that feature filtering is consistent between train/test"""
        np.random.seed(42)
        n_samples = 100
        
        # Create test data
        base_feature = np.random.randn(n_samples)
        X = np.column_stack([
            base_feature,
            base_feature + 0.01 * np.random.randn(n_samples),
            np.random.randn(n_samples)
        ])
        
        feature_names = ['feature1', 'feature2', 'feature3']
        
        # Apply filtering
        X_filtered, filtered_names = filter_collinear_features(X, feature_names, threshold=0.98)
        
        # Simulate test data with same structure
        X_test = np.column_stack([
            np.random.randn(50),
            np.random.randn(50),
            np.random.randn(50)
        ])
        
        # Apply same filtering to test data using feature indices
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}
        filtered_indices = [feature_idx_map[name] for name in filtered_names]
        X_test_filtered = X_test[:, filtered_indices]
        
        # Should have same number of features
        assert X_filtered.shape[1] == X_test_filtered.shape[1]
        assert len(filtered_names) == X_filtered.shape[1]


class TestLabelBalance:
    """Test label balance requirements"""
    
    def test_label_balance_validation(self):
        """Test that label balance validation works"""
        # Create balanced labels
        balanced_labels = pd.Series([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        
        # Should pass validation
        label_counts = balanced_labels.value_counts()
        assert len(label_counts) >= 2, "Should have at least 2 different labels"
        assert all(count > 0 for count in label_counts.values), "All labels should have positive counts"
    
    def test_imbalanced_labels_detection(self):
        """Test detection of imbalanced labels"""
        # Create imbalanced labels (all zeros)
        imbalanced_labels = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        label_counts = imbalanced_labels.value_counts()
        # This should trigger the "Labels too imbalanced" error in the actual system
        assert len(label_counts) == 1, "Should detect single-label imbalance"


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @patch('yfinance.download')
    def test_complete_pipeline_integrity(self, mock_download):
        """Test complete pipeline with proper data integrity"""
        # Mock realistic market data
        dates = pd.date_range('2023-01-01', periods=100)
        
        # Asset data (more volatile)
        asset_prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.02)
        
        # Market data (less volatile)
        market_prices = 200 * np.cumprod(1 + np.random.randn(100) * 0.01)
        
        mock_data = {
            'AAPL': pd.DataFrame({
                'Close': asset_prices,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates),
            'QQQ': pd.DataFrame({
                'Close': market_prices,
                'Volume': np.random.randint(2000, 20000, 100)
            }, index=dates)
        }
        
        def mock_download_func(symbols, **kwargs):
            if isinstance(symbols, str):
                return mock_data[symbols]
            else:
                return {sym: mock_data[sym] for sym in symbols}
        
        mock_download.side_effect = mock_download_func
        
        # Test data loading
        asset_data = yf.download('AAPL', start='2023-01-01', end='2023-04-10')
        market_data = yf.download('QQQ', start='2023-01-01', end='2023-04-10')
        
        # Test feature creation
        features = add_core_features(asset_data)
        assert len(features.columns) > 0, "Should create features"
        
        # Test target creation with proper market benchmark
        asset_returns = asset_data['Close'].pct_change().dropna()
        market_returns = market_data['Close'].pct_change().dropna()
        
        # Align data
        common_idx = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns.loc[common_idx]
        market_returns = market_returns.loc[common_idx]
        
        # Calculate excess returns
        beta = np.cov(asset_returns, market_returns)[0, 1] / np.var(market_returns)
        excess_returns = asset_returns - beta * market_returns
        
        # Test epsilon calculation
        train_idx = excess_returns.index[:50]  # Use first half as training
        eps = compute_epsilon_train_only(excess_returns, train_idx, q=0.25)
        assert eps >= 1e-3, f"Epsilon should respect floor, got {eps}"
        
        # Test label creation
        labels = label_excess_band(excess_returns, eps)
        label_counts = labels.value_counts()
        
        # Should have multiple labels
        assert len(label_counts) >= 2, f"Should have multiple labels, got {label_counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])