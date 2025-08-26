"""
Test the demo feature engineering.

Demonstrates testing patterns for quantitative systems:
- Data quality validation
- Feature engineering verification
- Property-based testing
"""

import pytest
import pandas as pd
import numpy as np
from core.ml.build_features import build_demo_features, build_matrix, validate_features

class TestDemoFeatures:
    """Test demo feature engineering functions."""
    
    def create_sample_data(self, n_days: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        
        dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
        np.random.seed(42)  # Deterministic for testing
        
        close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.02)
        
        return pd.DataFrame({
            'close': close_prices,
            'volume': np.random.lognormal(10, 0.5, n_days),
            'high': close_prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'low': close_prices * (1 - np.random.uniform(0, 0.02, n_days)),
            'open': close_prices + np.random.randn(n_days) * 0.5
        }, index=dates)
    
    def test_build_demo_features_basic(self):
        """Test basic feature building functionality."""
        
        data = self.create_sample_data(50)
        features = build_demo_features(data)
        
        # Check output structure
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features) < len(data)  # Should be shorter due to rolling windows
        
        # Check expected columns exist
        expected_cols = ['sma_5', 'sma_20', 'volatility', 'returns', 
                        'momentum_3d', 'momentum_5d', 'volume_ratio', 'price_position']
        for col in expected_cols:
            assert col in features.columns
    
    def test_build_demo_features_no_nans(self):
        """Test that output has no NaN values."""
        
        data = self.create_sample_data(50)
        features = build_demo_features(data)
        
        # Should have no NaN values after dropna()
        assert not features.isnull().any().any()
    
    def test_build_matrix_integration(self):
        """Test the build_matrix function."""
        
        data = self.create_sample_data(100)
        matrix = build_matrix(data)
        
        # Check output properties
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) > 0
        assert matrix.shape[1] == 8  # Expected number of features
        
        # No NaN or infinite values
        assert not matrix.isnull().any().any()
        assert not np.isinf(matrix.select_dtypes(include=[np.number])).any().any()
    
    def test_validate_features(self):
        """Test feature validation function."""
        
        data = self.create_sample_data(50)
        features = build_matrix(data)
        
        validation = validate_features(features)
        
        # Check validation structure
        assert 'shape' in validation
        assert 'na_count' in validation
        assert 'infinite_count' in validation
        assert 'feature_names' in validation
        
        # Check validation results
        assert validation['na_count'] == 0
        assert validation['infinite_count'] == 0
        assert len(validation['feature_names']) == features.shape[1]
    
    def test_feature_ranges(self):
        """Test that features are in reasonable ranges."""
        
        data = self.create_sample_data(100)
        features = build_matrix(data)
        
        # Volatility should be positive
        assert (features['volatility'] >= 0).all()
        
        # Returns should be bounded (reasonable for daily returns)
        assert (features['returns'].abs() < 0.5).all()  # 50% daily move is extreme
        
        # Volume ratio should be positive
        assert (features['volume_ratio'] > 0).all()
    
    @pytest.mark.parametrize("n_days", [30, 50, 100, 200])
    def test_different_data_sizes(self, n_days):
        """Test feature building with different data sizes."""
        
        data = self.create_sample_data(n_days)
        features = build_matrix(data)
        
        # Should work for all reasonable sizes
        assert len(features) > 0
        assert not features.isnull().any().any()
        
        # Should have reasonable size reduction due to rolling windows
        size_reduction = len(data) - len(features)
        assert size_reduction >= 20  # At least 20 days lost to rolling windows
        assert size_reduction <= 30  # But not too much
    
    def test_deterministic_output(self):
        """Test that feature building is deterministic."""
        
        data = self.create_sample_data(50)
        
        features1 = build_matrix(data)
        features2 = build_matrix(data)
        
        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        
        # Empty dataframe
        empty_data = pd.DataFrame(columns=['close', 'volume', 'high', 'low', 'open'])
        
        with pytest.raises((ValueError, IndexError)):
            build_matrix(empty_data)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        
        # Very small dataset (less than required for rolling windows)
        small_data = self.create_sample_data(10)
        
        features = build_matrix(small_data)
        # Should either be empty or very small
        assert len(features) <= 5
