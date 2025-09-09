#!/usr/bin/env python3
"""
Debug script to investigate MaxDD calculation issues.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/Jennifer/secure/trader')

from ml.panel_builder import PanelBuilder
from core.config import load_config

def debug_maxdd_calculation():
    """Debug MaxDD calculation with detailed output."""
    
    print("🔍 Debugging MaxDD calculation...")
    
    # Load a small sample of data
    config = load_config("config/universe_top300.yaml")
    universe = config["universe"][:5]  # Just first 5 tickers for debugging
    
    print(f"Loading data for universe: {universe}")
    
    # Build panel
    builder = PanelBuilder(universe, market_proxy="SPY")
    panel = builder.build_panel(start_date="2023-01-01", end_date="2024-01-01")
    
    print(f"Panel shape: {panel.shape}")
    
    # Simulate a simple portfolio construction
    print(f"\n🧪 Simulating portfolio construction...")
    
    # Get test data for one ticker
    test_ticker = 'TSLA'
    ticker_data = panel[panel['symbol'] == test_ticker].set_index('date')
    
    # Simulate strategy returns (using ret_fwd_5 as a proxy)
    strategy_returns = ticker_data['ret_fwd_5'].dropna()
    
    print(f"Strategy returns shape: {strategy_returns.shape}")
    print(f"Strategy returns stats: mean={strategy_returns.mean():.6f}, std={strategy_returns.std():.6f}")
    print(f"Strategy returns range: [{strategy_returns.min():.6f}, {strategy_returns.max():.6f}]")
    print(f"Negative returns count: {(strategy_returns < 0).sum()}")
    print(f"Zero returns count: {(strategy_returns == 0).sum()}")
    
    # Test different equity curve calculations
    print(f"\n📊 Testing equity curve calculations:")
    
    # Method 1: cumprod(1 + r)
    equity1 = (1 + strategy_returns).cumprod()
    print(f"Method 1 (cumprod): min={equity1.min():.6f}, max={equity1.max():.6f}")
    
    # Method 2: cumsum (wrong but sometimes used)
    equity2 = (1 + strategy_returns).cumsum()
    print(f"Method 2 (cumsum): min={equity2.min():.6f}, max={equity2.max():.6f}")
    
    # Method 3: with fillna(0) before cumprod
    equity3 = (1 + strategy_returns.fillna(0)).cumprod()
    print(f"Method 3 (fillna+cumprod): min={equity3.min():.6f}, max={equity3.max():.6f}")
    
    # Calculate drawdowns
    print(f"\n📉 Testing drawdown calculations:")
    
    # Method 1: Standard drawdown
    dd1 = equity1 / equity1.cummax() - 1
    maxdd1 = dd1.min()
    print(f"Method 1 drawdown: min={dd1.min():.6f}, max={dd1.max():.6f}, maxdd={maxdd1:.6f}")
    
    # Method 2: Alternative calculation
    dd2 = (equity1 - equity1.cummax()) / equity1.cummax()
    maxdd2 = dd2.min()
    print(f"Method 2 drawdown: min={dd2.min():.6f}, max={dd2.max():.6f}, maxdd={maxdd2:.6f}")
    
    # Method 3: With fillna
    dd3 = equity3 / equity3.cummax() - 1
    maxdd3 = dd3.min()
    print(f"Method 3 drawdown: min={dd3.min():.6f}, max={dd3.max():.6f}, maxdd={maxdd3:.6f}")
    
    # Check for any issues
    print(f"\n🔍 Diagnostic checks:")
    print(f"Any negative equity values: {(equity1 < 0).any()}")
    print(f"Any zero equity values: {(equity1 == 0).any()}")
    print(f"Any infinite equity values: {np.isinf(equity1).any()}")
    print(f"Any NaN equity values: {equity1.isna().any()}")
    
    # Check if equity curve is monotonic
    equity_diff = equity1.diff()
    print(f"Equity curve monotonic: {equity_diff.min():.6f} to {equity_diff.max():.6f}")
    print(f"Decreasing periods: {(equity_diff < 0).sum()}")
    
    # Show first few values
    print(f"\n📈 First 10 equity values:")
    print(equity1.head(10))
    
    print(f"\n📉 First 10 drawdown values:")
    print(dd1.head(10))
    
    # Test with a known bad case
    print(f"\n🧪 Testing with known bad case (all positive returns):")
    bad_returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01], index=pd.date_range('2023-01-01', periods=5))
    bad_equity = (1 + bad_returns).cumprod()
    bad_dd = bad_equity / bad_equity.cummax() - 1
    bad_maxdd = bad_dd.min()
    print(f"Bad case: returns={bad_returns.tolist()}")
    print(f"Bad case: equity={bad_equity.tolist()}")
    print(f"Bad case: drawdown={bad_dd.tolist()}")
    print(f"Bad case: maxdd={bad_maxdd:.6f}")
    
    # Test with a known good case (has negative returns)
    print(f"\n🧪 Testing with known good case (has negative returns):")
    good_returns = pd.Series([0.01, -0.02, 0.01, -0.03, 0.01], index=pd.date_range('2023-01-01', periods=5))
    good_equity = (1 + good_returns).cumprod()
    good_dd = good_equity / good_equity.cummax() - 1
    good_maxdd = good_dd.min()
    print(f"Good case: returns={good_returns.tolist()}")
    print(f"Good case: equity={good_equity.tolist()}")
    print(f"Good case: drawdown={good_dd.tolist()}")
    print(f"Good case: maxdd={good_maxdd:.6f}")

if __name__ == "__main__":
    debug_maxdd_calculation()
