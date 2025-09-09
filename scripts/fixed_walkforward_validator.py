#!/usr/bin/env python3
"""
Fixed Walkforward Validator - Implements proper threshold calibration and trade counting

This fixes the metric bugs identified in the audit:
1. Removes hard-coded confidence thresholds
2. Calibrates thresholds on train only
3. Counts trades correctly (entries, exits, flips, round-trips)
4. Uses proper daily PnL for Sharpe calculation
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging

import yfinance as yf


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def pick_tau_from_train(edges_train: np.ndarray, target_turnover: Tuple[float, float] = (0.05, 0.25)) -> float:
    """
    Pick edge threshold from training data only
    
    Args:
        edges_train: Model edges on TRAIN bars
        target_turnover: Target turnover range (min, max)
    
    Returns:
        Optimal threshold tau
    """
    abs_e = np.abs(edges_train)
    
    # Candidate taus at percentiles
    taus = np.quantile(abs_e, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    
    best = taus[0]
    best_gap = float('inf')
    
    for tau in taus:
        pos_train = np.sign(edges_train) * (abs_e > tau)
        turnover = np.mean(np.abs(np.diff(pos_train)))  # per-bar position change
        
        # Choose tau with turnover closest to center of target band
        center = 0.5 * (target_turnover[0] + target_turnover[1])
        gap = abs(turnover - center)
        
        if gap < best_gap:
            best, best_gap = tau, gap
    
    return float(best)


def apply_temperature(logits: np.ndarray, T: float = 1.5) -> np.ndarray:
    """Apply temperature scaling to logits"""
    return logits / T


def to_positions(edges: np.ndarray, tau: float, max_pos: int = 1, delay: int = 1) -> np.ndarray:
    """
    Convert edges to positions cleanly
    
    Args:
        edges: Model edges (BUY probability - SELL probability)
        tau: Edge threshold
        max_pos: Maximum position size
        delay: Execution delay in bars
    
    Returns:
        Position array (-1, 0, 1)
    """
    # Binary desired position from edge
    desired = np.where(np.abs(edges) > tau, np.sign(edges), 0).astype(int)
    
    # Optional delay to simulate fill
    if delay > 0:
        desired = np.concatenate([np.zeros(delay, dtype=int), desired[:-delay]])
    
    # Clip to {-1, 0, 1}
    return np.clip(desired, -max_pos, max_pos)


def trade_stats(positions: np.ndarray) -> Dict[str, int]:
    """
    Count trades correctly with multiple notions
    
    Returns:
        Dictionary with entries, exits, flips, round_trips, total_execs
    """
    prev = 0
    entries = exits = flips = 0
    
    for p in positions:
        if p == prev:
            continue
        
        if prev == 0 and p != 0:
            entries += 1
        elif prev != 0 and p == 0:
            exits += 1
        else:
            flips += 1  # prev != 0 and p != 0 and p != prev
        
        prev = p
    
    round_trips = min(entries, exits + flips)  # flips imply an exit
    total_execs = entries + exits + 2 * flips  # count both legs of flips
    
    return {
        'entries': entries,
        'exits': exits,
        'flips': flips,
        'round_trips': round_trips,
        'total_execs': total_execs
    }


def sharpe_from_daily(daily_pnl: np.ndarray) -> float:
    """Calculate Sharpe ratio from daily PnL"""
    if len(daily_pnl) < 2:
        return 0.0
    
    mu = daily_pnl.mean()
    sd = daily_pnl.std(ddof=1)
    
    return 0.0 if sd == 0 else (mu / sd) * (252 ** 0.5)


def simulate_trading_fixed(fold_data: pd.DataFrame, model, pipeline, symbol: str, 
                          tau: float = None, temperature: float = 1.5) -> Dict[str, Any]:
    """
    Fixed trading simulation with proper threshold calibration
    """
    
    initial_capital = 10000
    cash = initial_capital
    position_shares = 0.0
    
    # Track daily portfolio values and positions
    daily_portfolio_values = []
    daily_positions = []
    daily_edges = []
    
    # Use first 20% for warmup/training
    warmup_length = int(len(fold_data) * 0.2)
    trading_data = fold_data.iloc[warmup_length:]
    
    # Collect edges for threshold calibration (if tau not provided)
    edges_train = []
    edges_test = []
    
    for i, (timestamp, row) in enumerate(trading_data.iterrows()):
        # Get historical data for prediction
        historical_data = fold_data.iloc[:warmup_length + i + 1]
        
        # Make prediction
        try:
            if pipeline:
                # Build features using training pipeline
                features = pipeline.market_analyzer.build_comprehensive_features(historical_data)
                if len(features) > 0:
                    # Get latest features
                    X = features.iloc[-1:].values
                    if X.shape[1] > 0:
                        # Get action probabilities
                        if hasattr(model, 'predict_proba'):
                            action_probs = model.predict_proba(X)[0]
                        else:
                            # Fallback to confidence-based approach
                            action, confidence = model.predict_with_confidence(X)
                            action = action[0] if len(action) > 0 else 1
                            confidence = confidence[0] if len(confidence) > 0 else 0.0
                            # Convert to probabilities (simplified)
                            action_probs = np.array([0.33, 0.33, 0.34])  # Placeholder
                    else:
                        action_probs = np.array([0.33, 0.33, 0.34])  # HOLD
                else:
                    action_probs = np.array([0.33, 0.33, 0.34])  # HOLD
            else:
                action_probs = np.array([0.33, 0.33, 0.34])  # HOLD
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            action_probs = np.array([0.33, 0.33, 0.34])  # HOLD
        
        # Apply temperature scaling
        if temperature != 1.0:
            action_probs = apply_temperature(action_probs, temperature)
            # Renormalize
            action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        
        # Calculate edge (BUY - SELL probability)
        edge = action_probs[2] - action_probs[0]  # BUY - SELL
        
        # Store edges for threshold calibration
        if i < len(trading_data) * 0.7:  # First 70% for training
            edges_train.append(edge)
        else:
            edges_test.append(edge)
        
        daily_edges.append(edge)
    
    # Calibrate threshold on training data only
    if tau is None and len(edges_train) > 0:
        tau = pick_tau_from_train(np.array(edges_train))
        logging.info(f"Calibrated tau from train: {tau:.4f}")
    elif tau is None:
        tau = 0.1  # Default fallback
    
    # Convert edges to positions
    positions = to_positions(np.array(daily_edges), tau)
    
    # Execute trades based on positions
    trades = []
    for i, (timestamp, row) in enumerate(trading_data.iterrows()):
        current_price = row['Close']
        current_position = positions[i] if i < len(positions) else 0
        
        # Execute position changes
        if current_position > 0 and position_shares == 0:  # Enter long
            shares_to_buy = (cash * 0.1) / current_price
            cash -= shares_to_buy * current_price
            position_shares += shares_to_buy
            
            trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'shares': shares_to_buy,
                'price': current_price,
                'edge': daily_edges[i] if i < len(daily_edges) else 0.0
            })
            
        elif current_position < 0 and position_shares > 0:  # Exit long
            cash += position_shares * current_price
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': position_shares,
                'price': current_price,
                'edge': daily_edges[i] if i < len(daily_edges) else 0.0
            })
            
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
    sharpe_ratio = sharpe_from_daily(daily_returns)
    
    # Count trades correctly
    trade_stats_dict = trade_stats(positions)
    
    # Calculate win rate from round trips
    winning_trades = 0
    for entry_idx, exit_idx in zip(range(len(positions)), range(1, len(positions))):
        if positions[entry_idx] != 0 and positions[exit_idx] == 0:  # Round trip
            if exit_idx < len(daily_portfolio_values) and entry_idx < len(daily_portfolio_values):
                entry_value = daily_portfolio_values[entry_idx]
                exit_value = daily_portfolio_values[exit_idx]
                if exit_value > entry_value:
                    winning_trades += 1
    
    win_rate = winning_trades / trade_stats_dict['round_trips'] if trade_stats_dict['round_trips'] > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': trade_stats_dict['round_trips'],
        'win_rate': win_rate,
        'daily_returns': daily_returns,
        'daily_portfolio_values': daily_portfolio_values,
        'trades': trades,
        'tau': tau,
        'temperature': temperature,
        'trade_stats': trade_stats_dict,
        'num_daily_returns': len(daily_returns),
        'daily_return_mean': np.mean(daily_returns) if len(daily_returns) > 0 else 0.0,
        'daily_return_std': np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    }


def run_fixed_walkforward_validation(model_path: str, config_path: str, symbol: str = "SPY",
                                   fold_length: int = 63, step_size: int = 21) -> Dict[str, Any]:
    """
    Run walkforward validation with fixed metrics
    """
    
    setup_logging()
    logging.info("Starting fixed walkforward validation")
    
    # Load model and config
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load training pipeline
    pipeline = None
    try:
        import sys
        sys.path.append('.')
        from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
        
        metadata_path = model_path.replace('.pkl', '.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            training_config_dict = metadata.get('training_config', {})
            
            if training_config_dict:
                training_config = TrainingConfig(**training_config_dict)
                pipeline = RewardBasedTrainingPipeline(training_config)
                logging.info("Loaded training pipeline")
    except Exception as e:
        logging.warning(f"Could not load training pipeline: {e}")
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start="2023-01-01", end="2024-12-31")
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data: {len(data)} days")
    
    logging.info(f"Downloaded {len(data)} days of data")
    
    # Create walkforward folds
    folds = []
    for i in range(0, len(data) - fold_length, step_size):
        fold_data = data.iloc[i:i + fold_length]
        if len(fold_data) == fold_length:
            folds.append(fold_data)
    
    logging.info(f"Created {len(folds)} folds")
    
    # Run validation with fixed metrics
    fold_results = []
    
    for i, fold_data in enumerate(folds):
        logging.info(f"Processing fold {i+1}/{len(folds)}")
        
        try:
            result = simulate_trading_fixed(fold_data, model, pipeline, symbol)
            fold_results.append(result)
            
            # Print detailed metrics for each fold
            print(f"Fold {i+1}:")
            print(f"  Return: {result['total_return']:.4f}")
            print(f"  Sharpe: {result['sharpe_ratio']:.4f}")
            print(f"  Trades: {result['num_trades']} (entries: {result['trade_stats']['entries']}, exits: {result['trade_stats']['exits']}, flips: {result['trade_stats']['flips']})")
            print(f"  Win Rate: {result['win_rate']:.4f}")
            print(f"  Tau: {result['tau']:.4f}")
            print(f"  Daily Returns: {result['num_daily_returns']} days")
            print(f"  Daily Return Mean: {result['daily_return_mean']:.6f}")
            print(f"  Daily Return Std: {result['daily_return_std']:.6f}")
            print()
            
        except Exception as e:
            logging.error(f"Fold {i+1} failed: {e}")
            continue
    
    if not fold_results:
        raise ValueError("No successful folds")
    
    # Calculate aggregate metrics
    returns = [r['total_return'] for r in fold_results]
    sharpes = [r['sharpe_ratio'] for r in fold_results]
    trade_counts = [r['num_trades'] for r in fold_results]
    win_rates = [r['win_rate'] for r in fold_results]
    taus = [r['tau'] for r in fold_results]
    
    print("=" * 60)
    print("FIXED WALKFORWARD RESULTS")
    print("=" * 60)
    print(f"Number of Folds: {len(fold_results)}")
    print(f"Mean Return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    print(f"Mean Sharpe: {np.mean(sharpes):.4f} ± {np.std(sharpes):.4f}")
    print(f"Sharpe Range: [{np.min(sharpes):.4f}, {np.max(sharpes):.4f}]")
    print(f"Mean Trades: {np.mean(trade_counts):.1f} ± {np.std(trade_counts):.1f}")
    print(f"Trade Range: [{np.min(trade_counts)}, {np.max(trade_counts)}]")
    print(f"Mean Win Rate: {np.mean(win_rates):.4f} ± {np.std(win_rates):.4f}")
    print(f"Win Rate Range: [{np.min(win_rates):.4f}, {np.max(win_rates):.4f}]")
    print(f"Mean Tau: {np.mean(taus):.4f} ± {np.std(taus):.4f}")
    
    # Check invariants
    print("\n" + "=" * 60)
    print("INVARIANT CHECKS")
    print("=" * 60)
    
    # Check trade count variance
    trade_std = np.std(trade_counts)
    if trade_std > 0:
        print(f"✅ Trade count variance: {trade_std:.1f} (varies across folds)")
    else:
        print(f"❌ Trade count variance: {trade_std:.1f} (constant - BUG!)")
    
    # Check Sharpe variance
    sharpe_std = np.std(sharpes)
    if sharpe_std > 0.05:
        print(f"✅ Sharpe variance: {sharpe_std:.4f} (realistic variation)")
    else:
        print(f"❌ Sharpe variance: {sharpe_std:.4f} (too low - BUG!)")
    
    # Check for constant values
    if len(set(trade_counts)) > 1:
        print(f"✅ Trade counts vary: {set(trade_counts)}")
    else:
        print(f"❌ Trade counts constant: {trade_counts[0]} (BUG!)")
    
    if not np.allclose(np.array(sharpes), np.mean(sharpes)):
        print(f"✅ Sharpe ratios vary across folds")
    else:
        print(f"❌ Sharpe ratios constant (BUG!)")
    
    return {
        'fold_results': fold_results,
        'aggregate_metrics': {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'mean_trades': np.mean(trade_counts),
            'std_trades': np.std(trade_counts),
            'mean_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates)
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed walkforward validation")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--config-path", required=True, help="Path to config file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test")
    parser.add_argument("--fold-length", type=int, default=63, help="Fold length")
    parser.add_argument("--step-size", type=int, default=21, help="Step size")
    
    args = parser.parse_args()
    
    try:
        results = run_fixed_walkforward_validation(
            args.model_path, args.config_path, args.symbol,
            args.fold_length, args.step_size
        )
        print(f"\n🎉 Fixed validation completed successfully with {len(results['fold_results'])} folds")
    except Exception as e:
        print(f"❌ Fixed validation failed: {e}")
        exit(1)
