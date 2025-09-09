#!/usr/bin/env python3
"""
Debug script to investigate IR_mkt calculation issues.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/Jennifer/secure/trader')

from ml.metrics_market_neutral import capm_metrics, _newey_west_sigma
from ml.panel_builder import PanelBuilder
from core.config import load_config

def debug_ir_mkt_calculation():
    """Debug IR_mkt calculation with high precision output."""
    
    print("🔍 Debugging IR_mkt calculation...")
    
    # Load a small sample of data
    config = load_config("config/universe_top300.yaml")
    universe = config["universe"][:5]  # Just first 5 tickers for debugging
    
    print(f"Loading data for universe: {universe}")
    
    # Build panel
    builder = PanelBuilder(universe, market_proxy="SPY")
    panel = builder.build_panel(start_date="2023-01-01", end_date="2024-01-01")
    
    print(f"Panel shape: {panel.shape}")
    print(f"Panel columns: {list(panel.columns)}")
    
    # Get market returns (SPY) - use date as index for proper alignment
    spy_data = panel[panel['symbol'] == 'SPY'].set_index('date')
    market_returns = spy_data['ret_fwd_5'].dropna()
    print(f"Market returns (SPY) shape: {market_returns.shape}")
    print(f"Market returns stats: mean={market_returns.mean():.6f}, std={market_returns.std():.6f}")
    print(f"Market returns range: [{market_returns.min():.6f}, {market_returns.max():.6f}]")
    
    # Test with one ticker
    test_ticker = universe[1] if len(universe) > 1 else universe[0]
    if test_ticker == 'SPY':
        test_ticker = universe[0] if universe[0] != 'SPY' else universe[1]
    
    print(f"\n🧪 Testing with ticker: {test_ticker}")
    
    ticker_data = panel[panel['symbol'] == test_ticker].set_index('date')
    strategy_returns = ticker_data['ret_fwd_5'].dropna()
    
    print(f"Strategy returns shape: {strategy_returns.shape}")
    print(f"Strategy returns stats: mean={strategy_returns.mean():.6f}, std={strategy_returns.std():.6f}")
    print(f"Strategy returns range: [{strategy_returns.min():.6f}, {strategy_returns.max():.6f}]")
    
    # Debug date alignment
    print(f"Strategy returns index type: {type(strategy_returns.index)}")
    print(f"Market returns index type: {type(market_returns.index)}")
    print(f"Strategy returns index sample: {strategy_returns.index[:5]}")
    print(f"Market returns index sample: {market_returns.index[:5]}")
    
    # Align dates
    common_dates = strategy_returns.index.intersection(market_returns.index)
    print(f"Common dates: {len(common_dates)}")
    
    if len(common_dates) < 10:
        print("❌ Not enough common dates for analysis")
        print("Trying to align by reindexing...")
        
        # Try reindexing market returns to strategy returns index
        mkt_reindexed = market_returns.reindex(strategy_returns.index)
        print(f"Market returns after reindexing: {mkt_reindexed.dropna().shape}")
        
        # Use the reindexed version
        market_returns = mkt_reindexed
        common_dates = strategy_returns.index
    
    strat_aligned = strategy_returns.reindex(common_dates)
    mkt_aligned = market_returns.reindex(common_dates)
    
    print(f"\n📊 Aligned data stats:")
    print(f"Strategy: mean={strat_aligned.mean():.6f}, std={strat_aligned.std():.6f}")
    print(f"Market: mean={mkt_aligned.mean():.6f}, std={mkt_aligned.std():.6f}")
    print(f"Correlation: {strat_aligned.corr(mkt_aligned):.6f}")
    
    # Manual CAPM calculation with detailed output
    print(f"\n🔬 Manual CAPM calculation:")
    
    # OLS regression
    import statsmodels.api as sm
    X = sm.add_constant(mkt_aligned.values)
    y = strat_aligned.values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X constant column: {X[:, 0].mean():.6f} (should be 1.0)")
    print(f"X market column: mean={X[:, 1].mean():.6f}, std={X[:, 1].std():.6f}")
    print(f"y: mean={y.mean():.6f}, std={y.std():.6f}")
    
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})  # horizon=5, so h-1=4
    
    alpha = ols.params[0]
    beta = ols.params[1]
    alpha_se = ols.bse[0]
    alpha_t = alpha / alpha_se if alpha_se > 0 else np.nan
    
    print(f"OLS Results:")
    print(f"  Alpha: {alpha:.8f}")
    print(f"  Beta: {beta:.8f}")
    print(f"  Alpha SE: {alpha_se:.8f}")
    print(f"  Alpha t-stat: {alpha_t:.8f}")
    
    # Calculate residuals and IR
    resid = ols.resid
    resid_mu = np.mean(resid)
    resid_sigma = _newey_west_sigma(pd.Series(resid), lags=4)
    
    print(f"\nResiduals:")
    print(f"  Mean: {resid_mu:.8f}")
    print(f"  Std (simple): {np.std(resid, ddof=1):.8f}")
    print(f"  Std (Newey-West): {resid_sigma:.8f}")
    print(f"  Range: [{np.min(resid):.8f}, {np.max(resid):.8f}]")
    
    # Information Ratio
    annualizer = np.sqrt(252 / 5)  # horizon=5
    ir_mkt = (resid_mu / resid_sigma) * annualizer if resid_sigma > 0 else np.nan
    
    print(f"\nInformation Ratio:")
    print(f"  Annualizer: {annualizer:.6f}")
    print(f"  IR_mkt: {ir_mkt:.8f}")
    
    # Test the capm_metrics function
    print(f"\n🧪 Testing capm_metrics function:")
    result = capm_metrics(strategy_returns, market_returns, horizon=5)
    
    print(f"Function results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.8f}")
        else:
            print(f"  {key}: {value}")
    
    # Compare manual vs function
    print(f"\n🔍 Comparison:")
    print(f"Manual IR_mkt: {ir_mkt:.8f}")
    print(f"Function IR_mkt: {result['ir_mkt']:.8f}")
    print(f"Match: {np.isclose(ir_mkt, result['ir_mkt'], rtol=1e-10)}")

if __name__ == "__main__":
    debug_ir_mkt_calculation()
