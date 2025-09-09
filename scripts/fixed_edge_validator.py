#!/usr/bin/env python3
"""
Fixed Edge-Based Walkforward Validator

Implements all the fixes from the superprompt:
1. Class-probability alignment
2. Feature schema & scaler persistence
3. Train-only ε & τ calibration
4. Proper trade counting & metrics
5. Comprehensive logging & invariants
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

# Import our helper modules
from class_alignment_helpers import get_model_edge, assert_edge_quality, print_edge_diagnostics
from feature_schema_helpers import load_training_artifacts, validate_feature_pipeline, print_feature_diagnostics
from calibration_helpers import calibrate_fold_parameters, decide_with_hysteresis, print_calibration_diagnostics
from trade_metrics_helpers import (
    trade_stats_from_positions, sharpe_from_daily, calculate_daily_pnl,
    calculate_max_drawdown, calculate_profit_factor, assert_metrics_invariants,
    calculate_baseline_metrics, print_fold_diagnostics
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_enhanced_features(data: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
    """Build enhanced features matching the training pipeline"""
    features = pd.DataFrame(index=data.index)
    
    # Basic price features
    features['returns_1d'] = data['Close'].pct_change()
    features['returns_5d'] = data['Close'].pct_change(5)
    features['returns_20d'] = data['Close'].pct_change(20)
    
    # Volatility features (regime & risk)
    features['vol_5d'] = data['Close'].pct_change().rolling(5).std()
    features['vol_20d'] = data['Close'].pct_change().rolling(20).std()
    features['vol_60d'] = data['Close'].pct_change().rolling(60).std()
    features['vol_ratio'] = features['vol_5d'] / features['vol_20d']
    
    # Realized skew and kurtosis
    features['skew_20d'] = data['Close'].pct_change().rolling(20).skew()
    features['kurt_20d'] = data['Close'].pct_change().rolling(20).kurt()
    
    # Drawdown state
    rolling_max = data['Close'].rolling(20).max()
    features['drawdown'] = (data['Close'] - rolling_max) / rolling_max
    
    # Price position features
    features['high_20d'] = data['High'].rolling(20).max()
    features['low_20d'] = data['Low'].rolling(20).min()
    features['price_position'] = (data['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])
    
    # Multi-horizon momentum
    features['momentum_3d'] = data['Close'].pct_change(3)
    features['momentum_5d'] = data['Close'].pct_change(5)
    features['momentum_10d'] = data['Close'].pct_change(10)
    features['momentum_20d'] = data['Close'].pct_change(20)
    
    # RSI-style oscillator
    features['rsi'] = calculate_rsi(data['Close'], 14)
    
    # MACD
    features['macd'] = calculate_macd(data['Close'])
    
    # Volume features
    features['volume_ma'] = data['Volume'].rolling(20).mean()
    features['volume_ratio'] = data['Volume'] / features['volume_ma']
    
    # Gap indicators
    features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    features['gap_abs'] = np.abs(features['gap'])
    
    # Cross-asset features (if market data provided)
    if market_data is not None:
        market_returns = market_data['Close'].pct_change()
        features['market_returns_1d'] = market_returns
        features['market_returns_5d'] = market_returns.rolling(5).sum()
        features['relative_strength'] = features['returns_5d'] - features['market_returns_5d']
        
        # Market volatility
        market_vol = market_returns.rolling(20).std()
        features['market_vol'] = market_vol
        features['vol_spread'] = features['vol_20d'] - market_vol
    
    # Forward fill NaN values instead of dropping them
    # This allows us to use partial data when some features aren't available yet
    features = features.ffill().fillna(0.0)
    
    return features

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def simulate_fixed_edge_trading(fold_data: pd.DataFrame, model, scaler, schema: Dict[str, Any],
                               symbol: str, market_data: pd.DataFrame = None,
                               transaction_cost_bps: float = 4.0) -> Dict[str, Any]:
    """
    Simulate trading using fixed edge-based approach
    """
    
    initial_capital = 10000
    cash = initial_capital
    position_shares = 0.0
    
    # Use first 20% for warmup, but ensure we have at least 30 days for feature building
    # We'll handle missing features gracefully in the feature building
    warmup_length = max(int(len(fold_data) * 0.2), 30)
    trading_data = fold_data.iloc[warmup_length:]
    
    # Collect edges for threshold calibration
    edges = []
    daily_portfolio_values = []
    daily_positions = []
    trades = []
    
    for i, (timestamp, row) in enumerate(trading_data.iterrows()):
        # Get historical data for prediction
        historical_data = fold_data.iloc[:warmup_length + i + 1]
        
        # Build enhanced features matching training pipeline
        try:
            # Get market data for this period if available
            if market_data is not None:
                market_subset = market_data.loc[historical_data.index]
            else:
                market_subset = None
                
            features = build_enhanced_features(historical_data, market_subset)
            if len(features) > 0:
                # Validate feature pipeline
                X_scaled = validate_feature_pipeline(features, model, scaler, schema)
                
                # Get properly aligned edges
                edge = get_model_edge(model, X_scaled[-1:])[0]
            else:
                edge = 0.0
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            edge = 0.0
        
        edges.append(edge)
        
        # Store position for later portfolio calculation
        daily_positions.append(position_shares)
    
    # Calibrate parameters on first 70% of edges
    train_edges = np.array(edges[:int(len(edges) * 0.7)])
    
    # Calculate future horizon excess returns for epsilon calibration
    # Use the first 70% of the fold data to match train_edges
    train_data = fold_data.iloc[:int(len(fold_data) * 0.7)]
    horizon = 5  # Match the horizon used in training
    
    # Check if we're using the same symbol as market (e.g., SPY vs SPY)
    if market_data is not None and symbol != 'SPY':  # SPY is our market benchmark
        # Calculate daily excess returns: asset_return - market_return
        asset_returns = train_data['Close'].pct_change().dropna()
        market_subset = market_data.loc[train_data.index]
        market_returns = market_subset['Close'].pct_change().dropna()
        
        # Align the indices
        common_idx = asset_returns.index.intersection(market_returns.index)
        if len(common_idx) > 0:
            daily_excess = asset_returns.loc[common_idx] - market_returns.loc[common_idx]
            # Calculate future horizon excess returns (shifted by H days)
            future_excess = daily_excess.rolling(horizon).sum().shift(-horizon).dropna()
            abs_excess_train = np.abs(future_excess.values)
        else:
            # Fallback: use asset returns only
            daily_asset = asset_returns
            future_asset = daily_asset.rolling(horizon).sum().shift(-horizon).dropna()
            abs_excess_train = np.abs(future_asset.values)
    else:
        # No market data or same symbol as market: use asset returns only
        asset_returns = train_data['Close'].pct_change().dropna()
        future_asset = asset_returns.rolling(horizon).sum().shift(-horizon).dropna()
        abs_excess_train = np.abs(future_asset.values)
    
    # Debug: print excess returns info (only if empty)
    if len(abs_excess_train) == 0:
        print(f"Debug: abs_excess_train is empty!")
    elif np.count_nonzero(abs_excess_train) == 0:
        print(f"Debug: abs_excess_train all zeros! min={np.min(abs_excess_train):.6f}, max={np.max(abs_excess_train):.6f}")
    
    # Calibrate parameters
    params = calibrate_fold_parameters(train_edges, abs_excess_train)
    
    # Apply hysteresis trading
    positions = decide_with_hysteresis(np.array(edges), params['tau_enter'], params['tau_exit'])
    
    # Execute trades based on positions
    current_position = 0
    for i, (timestamp, row) in enumerate(trading_data.iterrows()):
        target_position = positions[i]
        current_price = row['Close']
        
        # Execute position changes
        if target_position != current_position:
            if target_position > current_position:  # Buy
                shares_to_buy = (cash * 0.1) / current_price
                cost = shares_to_buy * current_price * transaction_cost_bps / 10000
                cash -= shares_to_buy * current_price + cost
                position_shares += shares_to_buy
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'edge': edges[i]
                })
                
            elif target_position < current_position:  # Sell
                if position_shares > 0:
                    cost = position_shares * current_price * transaction_cost_bps / 10000
                    cash += position_shares * current_price - cost
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'shares': position_shares,
                        'price': current_price,
                        'edge': edges[i]
                    })
                    
                    position_shares = 0.0
        
        current_position = target_position
    
    # Calculate portfolio values after trading simulation
    # We need to reconstruct the portfolio values day by day
    daily_portfolio_values = []
    current_cash = initial_capital
    current_position = 0.0
    
    for i, (timestamp, row) in enumerate(trading_data.iterrows()):
        current_price = row['Close']
        target_position = positions[i]
        
        # Execute position changes
        if target_position != current_position:
            if target_position > current_position:  # Buy
                shares_to_buy = (current_cash * 0.1) / current_price
                cost = shares_to_buy * current_price * transaction_cost_bps / 10000
                current_cash -= shares_to_buy * current_price + cost
                current_position += shares_to_buy
            elif target_position < current_position:  # Sell
                shares_to_sell = current_position
                proceeds = shares_to_sell * current_price
                cost = proceeds * transaction_cost_bps / 10000
                current_cash += proceeds - cost
                current_position = 0.0
        
        # Calculate portfolio value with current position
        portfolio_value = current_cash + current_position * current_price
        daily_portfolio_values.append(portfolio_value)
    
    # Calculate final portfolio value
    final_price = trading_data['Close'].iloc[-1]
    final_value = cash + position_shares * final_price
    
    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital
    
    # Daily returns for Sharpe
    daily_returns = calculate_daily_pnl(daily_portfolio_values)
    sharpe = sharpe_from_daily(daily_returns)
    
    # Trade statistics
    trade_stats = trade_stats_from_positions(positions)
    
    # Edge statistics
    edge_stats = {
        'mean': np.mean(edges),
        'std': np.std(edges),
        'abs_mean': np.mean(np.abs(edges)),
        'above_tau': np.mean(np.abs(edges) > params['tau_enter'])
    }
    
    # Additional metrics
    mdd = calculate_max_drawdown(daily_portfolio_values)
    
    # Calculate profit factor from portfolio performance
    # Note: This is a simplified calculation based on final portfolio value
    # For proper trade-by-trade PnL, we'd need to track entry/exit pairs
    gross_profits = max(0, final_value - initial_capital)
    gross_losses = max(0, initial_capital - final_value)
    pf = calculate_profit_factor(gross_profits, gross_losses)
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'mdd': mdd,
        'profit_factor': pf,
        'trade_stats': trade_stats,
        'params': params,
        'edge_stats': edge_stats,
        'trades_list': trades,
        'positions': positions,
        'edges': edges
    }

def run_fixed_edge_validation(model_path: str, scaler_path: str, schema_path: str,
                             symbol: str = "SPY", fold_length: int = 63, step_size: int = 63):
    """
    Run fixed edge-based walkforward validation
    """
    
    logging.info("Starting fixed edge-based walkforward validation")
    
    # Load training artifacts
    model, scaler, schema = load_training_artifacts(model_path, scaler_path, schema_path)
    
    symbol = schema.get('symbol', symbol)
    logging.info(f"Loaded model for {symbol}")
    
    # Load and validate class mapping from config
    config_path = model_path.replace('.pkl', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_mapping = config.get('class_mapping', {})
        print(f"Class mapping: {class_mapping}")
        
        # Validate class mapping consistency
        if 'model_classes' in class_mapping:
            expected_classes = class_mapping['model_classes']
            if not np.array_equal(model.classes_, expected_classes):
                raise AssertionError(f"Class mismatch: model.classes_={model.classes_} != expected={expected_classes}")
    
    # Print model diagnostics
    print(f"Model classes: {model.classes_}")
    print(f"Feature schema: {schema['num_features']} features")
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start="2023-01-01", end="2024-12-31")
    
    # Download market data for cross-asset features
    market_ticker = yf.Ticker("SPY")
    market_data = market_ticker.history(start="2023-01-01", end="2024-12-31")
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data: {len(data)} days")
    
    logging.info(f"Downloaded {len(data)} days of data for {symbol}")
    logging.info(f"Downloaded {len(market_data)} days of market data")
    
    # Create walkforward folds (non-overlapping for first pass)
    folds = []
    for i in range(0, len(data) - fold_length, step_size):
        fold_data = data.iloc[i:i + fold_length]
        if len(fold_data) == fold_length:
            folds.append(fold_data)
    
    logging.info(f"Created {len(folds)} folds")
    
    # Run validation
    fold_results = []
    all_sharpes = []
    all_trade_counts = []
    all_positions = []
    
    for i, fold_data in enumerate(folds):
        logging.info(f"Processing fold {i+1}/{len(folds)}")
        
        try:
            # Model results
            model_result = simulate_fixed_edge_trading(fold_data, model, scaler, schema, symbol, market_data)
            
            # Baseline results
            returns = fold_data['Close'].pct_change().dropna()
            baseline_result = calculate_baseline_metrics(returns)
            
            fold_result = {
                'fold_id': i + 1,
                'model': model_result,
                'baseline': baseline_result
            }
            
            fold_results.append(fold_result)
            
            # Collect for invariants
            all_sharpes.append(model_result['sharpe'])
            all_trade_counts.append(model_result['trade_stats']['round_trips'])
            all_positions.append(model_result['positions'])
            
            # Print fold diagnostics
            print_fold_diagnostics(
                i + 1, model_result['params'], model_result['trade_stats'],
                model_result['edge_stats'], model_result['sharpe'],
                baseline_result['buy_hold_sharpe'], baseline_result['momentum_sharpe'],
                model_result['mdd'], model_result['profit_factor'], model.classes_
            )
            
        except Exception as e:
            logging.error(f"Fold {i+1} failed: {e}")
            continue
    
    if not fold_results:
        raise ValueError("No successful folds")
    
    # Assert metrics invariants
    assert_metrics_invariants(all_sharpes, all_trade_counts, all_positions, model.classes_)
    
    # Calculate aggregate metrics
    model_returns = [r['model']['total_return'] for r in fold_results]
    model_sharpes = [r['model']['sharpe'] for r in fold_results]
    model_trades = [r['model']['trade_stats']['round_trips'] for r in fold_results]
    
    bh_returns = [r['baseline']['buy_hold_return'] for r in fold_results]
    bh_sharpes = [r['baseline']['buy_hold_sharpe'] for r in fold_results]
    
    rule_returns = [r['baseline']['momentum_return'] for r in fold_results]
    rule_sharpes = [r['baseline']['momentum_sharpe'] for r in fold_results]
    
    print("=" * 80)
    print("FIXED EDGE-BASED WALKFORWARD RESULTS")
    print("=" * 80)
    print(f"Number of Folds: {len(fold_results)}")
    print(f"Model - Mean Return: {np.mean(model_returns):.4f} ± {np.std(model_returns):.4f}")
    print(f"Model - Mean Sharpe: {np.mean(model_sharpes):.4f} ± {np.std(model_sharpes):.4f}")
    print(f"Model - Mean Trades: {np.mean(model_trades):.1f} ± {np.std(model_trades):.1f}")
    print()
    print(f"Buy & Hold - Mean Return: {np.mean(bh_returns):.4f} ± {np.std(bh_returns):.4f}")
    print(f"Buy & Hold - Mean Sharpe: {np.mean(bh_sharpes):.4f} ± {np.std(bh_sharpes):.4f}")
    print()
    print(f"Simple Rule - Mean Return: {np.mean(rule_returns):.4f} ± {np.std(rule_returns):.4f}")
    print(f"Simple Rule - Mean Sharpe: {np.mean(rule_sharpes):.4f} ± {np.std(rule_sharpes):.4f}")
    
    # Gate on median Sharpe
    median_model_sharpe = np.median(model_sharpes)
    median_bh_sharpe = np.median(bh_sharpes)
    median_rule_sharpe = np.median(rule_sharpes)
    
    print(f"\nMedian Sharpe Comparison:")
    print(f"Model: {median_model_sharpe:.3f}")
    print(f"Buy & Hold: {median_bh_sharpe:.3f}")
    print(f"Simple Rule: {median_rule_sharpe:.3f}")
    
    # Check if model beats baselines
    baseline_sharpe = max(median_bh_sharpe, median_rule_sharpe)
    beats_baseline = median_model_sharpe > baseline_sharpe + 0.1
    
    print(f"\nGate Result: {'✅ PASS' if beats_baseline else '❌ FAIL'}")
    print(f"Model median Sharpe ({median_model_sharpe:.3f}) {'>' if beats_baseline else '≤'} baseline + 0.1 ({baseline_sharpe + 0.1:.3f})")
    
    return fold_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed edge-based walkforward validation")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--scaler-path", required=True, help="Path to scaler file")
    parser.add_argument("--schema-path", required=True, help="Path to schema file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test")
    parser.add_argument("--fold-length", type=int, default=63, help="Fold length")
    parser.add_argument("--step-size", type=int, default=63, help="Step size")
    
    args = parser.parse_args()
    
    try:
        results = run_fixed_edge_validation(
            args.model_path, args.scaler_path, args.schema_path,
            args.symbol, args.fold_length, args.step_size
        )
        print(f"\n🎉 Fixed edge-based validation completed successfully with {len(results)} folds")
    except Exception as e:
        print(f"❌ Fixed edge-based validation failed: {e}")
        exit(1)
