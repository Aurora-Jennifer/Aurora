#!/usr/bin/env python3
"""
Simple audit script to check the walkforward validation metrics bugs
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import yfinance as yf
from typing import Dict, Any, List, Tuple


def sharpe_from_daily_pnl(daily_pnl: np.ndarray) -> float:
    """Calculate Sharpe ratio from daily PnL (proper method)"""
    if len(daily_pnl) < 2:
        return 0.0
    
    mu = np.mean(daily_pnl)
    sd = np.std(daily_pnl, ddof=1)
    
    if sd == 0:
        return 0.0
    
    return (mu / sd) * np.sqrt(252)


def count_round_trip_trades(positions: np.ndarray) -> List[Tuple[int, int]]:
    """Count actual round-trip trades (flat->nonzero->flat)"""
    entries, exits = [], []
    prev = 0
    
    for i, pos in enumerate(positions):
        if prev == 0 and pos != 0:
            entries.append(i)
        if prev != 0 and pos == 0:
            exits.append(i)
        prev = pos
    
    # Only count complete round trips
    return list(zip(entries, exits))


def simulate_trading_simple(fold_data: pd.DataFrame) -> Dict[str, Any]:
    """Simple trading simulation to test metric calculation"""
    
    initial_capital = 10000
    cash = initial_capital
    position_shares = 0.0
    
    # Track daily portfolio values
    daily_portfolio_values = []
    daily_positions = []
    
    # Simple strategy: buy on day 1, sell on day 2, repeat
    for i, (timestamp, row) in enumerate(fold_data.iterrows()):
        current_price = row['Close']
        
        # Simple alternating strategy
        if i % 2 == 0 and cash > 0:  # Buy every other day
            shares_to_buy = (cash * 0.1) / current_price
            cash -= shares_to_buy * current_price
            position_shares += shares_to_buy
        elif i % 2 == 1 and position_shares > 0:  # Sell every other day
            cash += position_shares * current_price
            position_shares = 0.0
        
        # Calculate daily portfolio value
        portfolio_value = cash + position_shares * current_price
        daily_portfolio_values.append(portfolio_value)
        daily_positions.append(position_shares)
    
    # Calculate daily returns
    if len(daily_portfolio_values) > 1:
        daily_returns = np.diff(daily_portfolio_values) / daily_portfolio_values[:-1]
    else:
        daily_returns = np.array([0.0])
    
    # Calculate metrics properly
    total_return = (daily_portfolio_values[-1] - initial_capital) / initial_capital if len(daily_portfolio_values) > 0 else 0.0
    sharpe_ratio = sharpe_from_daily_pnl(daily_returns)
    
    # Count actual round-trip trades
    round_trips = count_round_trip_trades(np.array(daily_positions))
    num_round_trips = len(round_trips)
    
    # Calculate win rate from round trips
    winning_trades = 0
    for entry_idx, exit_idx in round_trips:
        if exit_idx < len(daily_portfolio_values) and entry_idx < len(daily_portfolio_values):
            entry_value = daily_portfolio_values[entry_idx]
            exit_value = daily_portfolio_values[exit_idx]
            if exit_value > entry_value:
                winning_trades += 1
    
    win_rate = winning_trades / num_round_trips if num_round_trips > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': num_round_trips,
        'win_rate': win_rate,
        'daily_returns': daily_returns,
        'daily_portfolio_values': daily_portfolio_values,
        'num_daily_returns': len(daily_returns),
        'daily_return_mean': np.mean(daily_returns) if len(daily_returns) > 0 else 0.0,
        'daily_return_std': np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    }


def main():
    print("🔍 AUDITING WALKFORWARD METRICS")
    print("=" * 50)
    
    # Download test data
    print("Downloading test data...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(start="2023-01-01", end="2023-06-01")  # Longer period for testing
    
    if len(data) < 50:
        print(f"❌ Insufficient data: {len(data)} days")
        return
    
    print(f"Downloaded {len(data)} days of data")
    
    # Create test folds
    fold_length = 20  # Short folds for testing
    step_size = 10
    
    folds = []
    for i in range(0, len(data) - fold_length, step_size):
        fold_data = data.iloc[i:i + fold_length]
        if len(fold_data) == fold_length:
            folds.append(fold_data)
    
    print(f"Created {len(folds)} test folds")
    print()
    
    # Run simple simulation on each fold
    fold_results = []
    
    for i, fold_data in enumerate(folds):
        print(f"Fold {i+1}:")
        
        result = simulate_trading_simple(fold_data)
        fold_results.append(result)
        
        # Print detailed metrics for each fold
        print(f"  Return: {result['total_return']:.4f}")
        print(f"  Sharpe: {result['sharpe_ratio']:.4f}")
        print(f"  Trades: {result['num_trades']}")
        print(f"  Win Rate: {result['win_rate']:.4f}")
        print(f"  Daily Returns: {result['num_daily_returns']} days")
        print(f"  Daily Return Mean: {result['daily_return_mean']:.6f}")
        print(f"  Daily Return Std: {result['daily_return_std']:.6f}")
        print()
    
    # Calculate aggregate metrics
    returns = [r['total_return'] for r in fold_results]
    sharpes = [r['sharpe_ratio'] for r in fold_results]
    trade_counts = [r['num_trades'] for r in fold_results]
    win_rates = [r['win_rate'] for r in fold_results]
    
    print("=" * 50)
    print("AUDIT RESULTS")
    print("=" * 50)
    print(f"Number of Folds: {len(fold_results)}")
    print(f"Mean Return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    print(f"Mean Sharpe: {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f}")
    print(f"Sharpe Range: [{np.min(sharpes):.4f}, {np.max(sharpes):.4f}]")
    print(f"Mean Trades: {np.mean(trade_counts):.1f} ± {np.std(trade_counts):.1f}")
    print(f"Trade Range: [{np.min(trade_counts)}, {np.max(trade_counts)}]")
    print(f"Mean Win Rate: {np.mean(win_rates):.4f} ± {np.std(win_rates):.4f}")
    print(f"Win Rate Range: [{np.min(win_rates):.4f}, {np.max(win_rates):.4f}]")
    
    # Check for suspicious patterns
    print("\n" + "=" * 50)
    print("SUSPICIOUS PATTERN DETECTION")
    print("=" * 50)
    
    # Check Sharpe variance
    sharpe_std = np.std(sharpes)
    if sharpe_std < 0.1:
        print(f"⚠️  LOW SHARPE VARIANCE: {sharpe_std:.4f} (should be > 0.1)")
    else:
        print(f"✅ Sharpe variance looks normal: {sharpe_std:.4f}")
    
    # Check trade count variance
    trade_std = np.std(trade_counts)
    if trade_std < 1.0:
        print(f"⚠️  LOW TRADE COUNT VARIANCE: {trade_std:.1f} (should vary more)")
    else:
        print(f"✅ Trade count variance looks normal: {trade_std:.1f}")
    
    # Check for constant values
    if len(set(sharpes)) == 1:
        print(f"⚠️  CONSTANT SHARPE RATIO: {sharpes[0]:.4f} (all folds identical)")
    else:
        print(f"✅ Sharpe ratios vary across folds")
    
    if len(set(trade_counts)) == 1:
        print(f"⚠️  CONSTANT TRADE COUNT: {trade_counts[0]} (all folds identical)")
    else:
        print(f"✅ Trade counts vary across folds")
    
    print(f"\n🎉 Simple audit completed with {len(fold_results)} folds")


if __name__ == "__main__":
    main()
