#!/usr/bin/env python3
"""
Test script to check if identical close prices are causing lookahead contamination.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sanity.main import DataSanityValidator


def test_identical_closes():
    """Test if identical close prices are causing lookahead contamination."""
    
    # Load the data
    data_path = "artifacts/snapshots/golden_ml_v1/SPY.parquet"
    print(f"📁 Loading data from: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"📊 Original data shape: {df.shape}")
    
    # Check for identical consecutive close prices
    close_prices = df["Close"]
    print(f"📈 Close prices stats: min={close_prices.min()}, max={close_prices.max()}, unique={close_prices.nunique()}")
    
    # Check for consecutive identical values
    consecutive_identical = (close_prices == close_prices.shift(1)) & close_prices.notna()
    identical_count = consecutive_identical.sum()
    print(f"🔍 Consecutive identical close prices: {identical_count}")
    
    if identical_count > 0:
        print("📋 Sample identical close prices:")
        identical_indices = consecutive_identical[consecutive_identical].index[:5]
        for idx in identical_indices:
            print(f"  {idx}: Close={close_prices.loc[idx]}")
    
    # Calculate returns manually to see if they match
    returns = np.log(close_prices / close_prices.shift(1))
    print(f"📊 Returns stats: min={returns.min()}, max={returns.max()}, unique={returns.nunique()}")
    
    # Check for consecutive identical returns
    consecutive_identical_returns = (returns == returns.shift(1)) & returns.notna()
    identical_returns_count = consecutive_identical_returns.sum()
    print(f"🔍 Consecutive identical returns: {identical_returns_count}")
    
    # Check for exact matches with future values
    if len(returns) > 1:
        future_1 = returns.shift(-1)
        exact_matches_1 = (returns == future_1) & returns.notna() & future_1.notna()
        match_count_1 = exact_matches_1.sum()
        print(f"🔍 Exact matches with offset 1: {match_count_1}")
        
        if match_count_1 > 0:
            print("📋 Sample exact matches:")
            match_indices = exact_matches_1[exact_matches_1].index[:5]
            for idx in match_indices:
                print(f"  {idx}: Returns={returns.loc[idx]}")
    
    # Test the lookahead detection logic
    print("\n🧪 Testing lookahead detection logic:")
    
    # Simulate the lookahead detection
    if "Returns" in df.columns and len(df) > 1:
        r = df["Returns"].to_numpy()
        eq_next = np.isfinite(r[:-1]) & np.isfinite(r[1:]) & (np.abs(r[:-1] - r[1:]) < 1e-12)
        
        if eq_next.any():
            print(f"🔍 Lookahead detection found {eq_next.sum()} matches")
            
            # Check if this is legitimate (consecutive identical close prices)
            if "Close" in df.columns:
                close_prices = df["Close"].to_numpy()
                identical_closes = np.isfinite(close_prices[:-1]) & np.isfinite(close_prices[1:]) & (np.abs(close_prices[:-1] - close_prices[1:]) < 1e-12)
                
                print(f"🔍 Identical close prices: {identical_closes.sum()}")
                print(f"🔍 Arrays equal: {np.array_equal(eq_next, identical_closes)}")
                
                # Show some examples
                if eq_next.sum() > 0:
                    print("📋 Sample lookahead matches vs identical closes:")
                    for i in range(min(5, eq_next.sum())):
                        idx = np.where(eq_next)[0][i]
                        print(f"  Index {idx}: Lookahead={eq_next[idx]}, IdenticalClose={identical_closes[idx]}")
                        if idx < len(df):
                            print(f"    Close[{idx}]={close_prices[idx]}, Close[{idx+1}]={close_prices[idx+1]}")
                            if "Returns" in df.columns:
                                print(f"    Returns[{idx}]={df['Returns'].iloc[idx]}, Returns[{idx+1}]={df['Returns'].iloc[idx+1]}")


if __name__ == "__main__":
    test_identical_closes()
