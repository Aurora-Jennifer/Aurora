#!/usr/bin/env python3
"""
Unit tests for PortfolioAggregator constructor and top_k parameter handling
"""

import pytest
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from portfolio_aggregate import PortfolioAggregator


def test_portfolio_aggregator_accepts_top_k():
    """Test that PortfolioAggregator accepts top_k parameter."""
    agg = PortfolioAggregator(top_k=2, min_sharpe=0.0, min_trades=1)
    assert agg.k == 2
    assert agg.min_sharpe == 0.0
    assert agg.min_trades == 1


def test_portfolio_aggregator_accepts_k():
    """Test that PortfolioAggregator accepts k parameter."""
    agg = PortfolioAggregator(k=3, min_sharpe=0.1, min_trades=5)
    assert agg.k == 3
    assert agg.min_sharpe == 0.1
    assert agg.min_trades == 5


def test_portfolio_aggregator_prefers_k_over_top_k():
    """Test that k parameter takes precedence over top_k."""
    agg = PortfolioAggregator(top_k=2, k=4, min_sharpe=0.0, min_trades=1)
    assert agg.k == 4  # k should take precedence


def test_portfolio_aggregator_requires_k_or_top_k():
    """Test that PortfolioAggregator raises error when neither k nor top_k provided."""
    with pytest.raises(TypeError, match="Provide 'top_k' or 'k' to PortfolioAggregator"):
        PortfolioAggregator(min_sharpe=0.0, min_trades=1)


def test_portfolio_aggregator_with_config():
    """Test that PortfolioAggregator works with config-based initialization."""
    config = {
        'portfolio': {
            'top_k': 5,
            'min_sharpe': 0.2,
            'min_trades': 10
        },
        'risk': {
            'max_position_weight': 0.3,
            'max_turnover': 0.4
        }
    }
    
    agg = PortfolioAggregator(config=config)
    assert agg.k == 5
    assert agg.min_sharpe == 0.2
    assert agg.max_position_weight == 0.3
    assert agg.max_turnover == 0.4


def test_portfolio_aggregator_defaults():
    """Test that PortfolioAggregator sets reasonable defaults."""
    agg = PortfolioAggregator(top_k=1, min_sharpe=0.0, min_trades=0)
    assert agg.max_position_weight == 1.0
    assert agg.max_net_exposure == 1.0
    assert agg.max_turnover == 0.5
    assert agg.target_volatility == 0.10
    assert agg.hysteresis_factor == 0.1
