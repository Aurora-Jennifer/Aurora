#!/usr/bin/env python3
"""
Research Module Tests

Tests for the research pipeline components:
- Target construction
- Decision making
- Feature engineering
- Baseline models
- Grid runner
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ml.targets import create_targets, validate_targets, compute_epsilon_train_only
from ml.decision import align_proba, temperature_scale, edge_from_P, pick_tau_from_train, decide_hysteresis
from ml.baselines import RidgeExcessModel, buy_and_hold_daily_pnl, simple_rule_daily_pnl
from ml.features import add_core_features, get_feature_columns, validate_feature_schema


class TestTargets:
    """Test target construction"""
    
    def test_labels_band(self):
        """Test epsilon > 0 and labels in {-1,0,1}"""
        # Create sample data
        np.random.seed(42)
        n = 1000
        asset_ret = pd.Series(np.random.randn(n) * 0.02, index=pd.date_range('2023-01-01', periods=n))
        market_ret = pd.Series(np.random.randn(n) * 0.015, index=asset_ret.index)
        
        # Create targets
        train_idx = asset_ret.index[:700]
        labels, targets, eps = create_targets(asset_ret, market_ret, H=5, train_idx=train_idx)
        
        # Test epsilon
        assert eps > 0, f"Epsilon should be positive, got {eps}"
        assert eps < 0.1, f"Epsilon should be reasonable, got {eps}"
        
        # Test labels
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({-1, 0, 1}), f"Labels should be in {{-1,0,1}}, got {unique_labels}"
        
        # Test class proportions
        label_counts = labels.value_counts()
        assert len(label_counts) >= 2, f"Should have at least 2 classes, got {label_counts}"
        
        # Test validation
        validate_targets(labels, targets, eps)
    
    def test_epsilon_computation(self):
        """Test epsilon computation from training data"""
        # Create sample excess returns
        np.random.seed(42)
        n = 1000
        excess_returns = pd.Series(np.random.randn(n) * 0.01, index=pd.date_range('2023-01-01', periods=n))
        train_idx = excess_returns.index[:700]
        
        # Test different quantiles
        for q in [0.4, 0.5, 0.6]:
            eps = compute_epsilon_train_only(excess_returns, train_idx, q)
            assert eps > 0, f"Epsilon should be positive for q={q}, got {eps}"
            assert np.isfinite(eps), f"Epsilon should be finite for q={q}, got {eps}"
    
    def test_epsilon_invalid_inputs(self):
        """Test epsilon computation with invalid inputs"""
        # Empty training data
        excess_returns = pd.Series([], index=pd.DatetimeIndex([]))
        train_idx = pd.DatetimeIndex([])
        
        with pytest.raises(AssertionError):
            compute_epsilon_train_only(excess_returns, train_idx)


class TestDecision:
    """Test decision making and calibration"""
    
    def test_align_proba(self):
        """Test probability alignment to canonical order"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data
        np.random.seed(42)
        X, y = np.random.randn(100, 5), np.random.randint(0, 3, 100)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test alignment
        proba = model.predict_proba(X[:10])
        aligned = align_proba(model, proba)
        
        # Check shape and normalization
        assert aligned.shape == (10, 3), f"Expected shape (10, 3), got {aligned.shape}"
        assert np.allclose(aligned.sum(1), 1.0, atol=1e-6), "Probabilities should sum to 1"
    
    def test_temperature_scale(self):
        """Test temperature scaling"""
        # Create sample probabilities
        P = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3]])
        
        # Test temperature scaling
        P_scaled = temperature_scale(P, T=2.0)
        
        # Check normalization
        assert np.allclose(P_scaled.sum(1), 1.0, atol=1e-6), "Scaled probabilities should sum to 1"
        assert P_scaled.shape == P.shape, "Shape should be preserved"
    
    def test_edges_nonconstant(self):
        """Test edge calculation produces non-constant values"""
        # Create sample probabilities
        P = np.array([
            [0.1, 0.8, 0.1],  # Edge = 0.1 - 0.1 = 0.0
            [0.2, 0.6, 0.2],  # Edge = 0.2 - 0.2 = 0.0
            [0.3, 0.4, 0.3],  # Edge = 0.3 - 0.3 = 0.0
        ])
        
        # This should raise an error for constant edges
        with pytest.raises(AssertionError):
            edge_from_P(P)
        
        # Test with non-constant probabilities
        P_var = np.array([
            [0.1, 0.8, 0.1],  # Edge = 0.0
            [0.2, 0.6, 0.2],  # Edge = 0.0
            [0.1, 0.6, 0.3],  # Edge = 0.2
        ])
        
        edges = edge_from_P(P_var)
        assert np.std(edges) > 1e-6, "Edges should have variation"
    
    def test_tau_selection(self):
        """Test tau selection yields reasonable turnover"""
        # Create sample edges
        np.random.seed(42)
        edges_train = np.random.randn(500) * 0.1
        
        # Test tau selection
        tau = pick_tau_from_train(edges_train, turnover_band=(0.08, 0.18))
        
        assert tau > 0, f"Tau should be positive, got {tau}"
        assert tau < 1.0, f"Tau should be reasonable, got {tau}"
    
    def test_hysteresis(self):
        """Test hysteresis decision making"""
        # Create sample edges
        edges = np.array([0.0, 0.1, 0.05, -0.1, -0.05, 0.0])
        tau_in, tau_out = 0.08, 0.04
        
        positions = decide_hysteresis(edges, tau_in, tau_out)
        
        # Check positions are valid
        assert np.all(np.isin(positions, [-1, 0, 1])), f"Invalid positions: {positions}"
        assert len(positions) == len(edges), "Length should match"


class TestBaselines:
    """Test baseline models"""
    
    def test_buy_and_hold(self):
        """Test buy and hold baseline"""
        # Create sample price data
        np.random.seed(42)
        n = 100
        prices = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), 
                          index=pd.date_range('2023-01-01', periods=n))
        
        pnl = buy_and_hold_daily_pnl(prices)
        
        assert len(pnl) == len(prices), "Length should match"
        assert np.allclose(pnl.iloc[0], 0.0), "First return should be 0"
    
    def test_simple_rule(self):
        """Test simple rule baseline"""
        # Create sample price data
        np.random.seed(42)
        n = 100
        prices = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), 
                          index=pd.date_range('2023-01-01', periods=n))
        
        pnl = simple_rule_daily_pnl(prices, costs_bps=3.0)
        
        assert len(pnl) == len(prices), "Length should match"
        assert np.allclose(pnl.iloc[0], 0.0), "First return should be 0"
    
    def test_ridge_model(self):
        """Test Ridge regression model"""
        # Create sample data
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randn(n) * 0.01)
        
        # Train model
        model = RidgeExcessModel(alpha=1.0)
        model.fit(X, y)
        
        # Test predictions
        edges = model.predict_edge(X[:100])
        proba = model.predict_proba(X[:100])
        
        assert len(edges) == 100, "Edge length should match"
        assert proba.shape == (100, 3), "Proba shape should be (100, 3)"
        assert np.allclose(proba.sum(1), 1.0, atol=1e-6), "Probabilities should sum to 1"


class TestFeatures:
    """Test feature engineering"""
    
    def test_core_features(self):
        """Test core feature creation"""
        # Create sample data
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + np.random.randn(n) * 0.01),
            'Open': 100 * np.cumprod(1 + np.random.randn(n) * 0.01),
            'Volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2023-01-01', periods=n))
        
        # Add features
        df_features = add_core_features(df)
        
        # Check feature columns
        feature_cols = get_feature_columns(df_features)
        assert len(feature_cols) > 10, f"Should have many features, got {len(feature_cols)}"
        
        # Check no NaN values
        assert not df_features.isnull().any().any(), "Features should not contain NaN values"
        
        # Check no infinite values
        assert np.all(np.isfinite(df_features.values)), "Features should be finite"
    
    def test_feature_schema(self):
        """Test feature schema validation"""
        # Create sample data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + np.random.randn(n) * 0.01),
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n)
        })
        
        # Create schema
        feature_cols = get_feature_columns(df)
        schema = {
            'feature_columns': feature_cols,
            'num_features': len(feature_cols)
        }
        
        # Test validation
        validate_feature_schema(df[feature_cols], schema)
        
        # Test with wrong columns
        with pytest.raises(AssertionError):
            validate_feature_schema(df, schema)


class TestIntegration:
    """Integration tests"""
    
    def test_no_leakage(self):
        """Test that calibrations use train indices only"""
        # This is a critical test - ensure no lookahead bias
        np.random.seed(42)
        n = 1000
        
        # Create sample data
        asset_ret = pd.Series(np.random.randn(n) * 0.02, index=pd.date_range('2023-01-01', periods=n))
        market_ret = pd.Series(np.random.randn(n) * 0.015, index=asset_ret.index)
        
        # Split train/test
        train_idx = asset_ret.index[:700]
        test_idx = asset_ret.index[700:]
        
        # Create targets using train only
        labels, targets, eps = create_targets(asset_ret, market_ret, H=5, train_idx=train_idx)
        
        # Verify epsilon was computed from train only
        train_excess = targets.loc[train_idx].abs()
        expected_eps = np.quantile(train_excess.dropna(), 0.5)
        assert abs(eps - expected_eps) < 1e-10, "Epsilon should be computed from train only"
    
    def test_costs_applied(self):
        """Test that position changes incur costs"""
        # Create sample data with known costs
        np.random.seed(42)
        n = 100
        prices = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), 
                          index=pd.date_range('2023-01-01', periods=n))
        
        # Test with high costs
        pnl_high_costs = simple_rule_daily_pnl(prices, costs_bps=100.0)  # 1% costs
        pnl_low_costs = simple_rule_daily_pnl(prices, costs_bps=1.0)     # 0.01% costs
        
        # High costs should result in lower returns
        assert pnl_high_costs.sum() < pnl_low_costs.sum(), "High costs should reduce returns"
    
    def test_baselines_clean(self):
        """Test that baselines are clean (no lookahead)"""
        # Create trending data
        np.random.seed(42)
        n = 100
        trend = np.linspace(100, 110, n)
        noise = np.random.randn(n) * 0.5
        prices = pd.Series(trend + noise, index=pd.date_range('2023-01-01', periods=n))
        
        # Test simple rule
        pnl = simple_rule_daily_pnl(prices)
        
        # Should not have perfect returns (which would indicate lookahead)
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
        assert abs(sharpe) < 10.0, f"Sharpe too high ({sharpe}), possible lookahead bias"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
