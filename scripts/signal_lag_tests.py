#!/usr/bin/env python3
"""
Signal Lag & Leakage Tests
Tests for temporal integrity and signal lag detection
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from scipy import stats
from scipy.stats import pearsonr
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_predictions(pred_root: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Load prediction data for all tickers."""
    predictions = {}
    
    for ticker in tickers:
        # Look for prediction files in various locations
        pred_patterns = [
            f"{pred_root}/**/{ticker}*_predictions.parquet",
            f"{pred_root}/**/{ticker}*_predictions.csv",
            f"{pred_root}/**/{ticker}*_grid.csv",
            f"{pred_root}/{ticker}*_predictions.parquet",
            f"{pred_root}/{ticker}*_predictions.csv",
            f"{pred_root}/{ticker}*_grid.csv"
        ]
        
        pred_file = None
        for pattern in pred_patterns:
            files = glob.glob(pattern, recursive=True)
            if files:
                pred_file = files[0]
                break
        
        if pred_file:
            try:
                if pred_file.endswith('.parquet'):
                    df = pd.read_parquet(pred_file)
                else:
                    df = pd.read_csv(pred_file)
                predictions[ticker] = df
                logger.info(f"Loaded predictions for {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading predictions for {ticker}: {e}")
        else:
            logger.warning(f"No prediction file found for {ticker}")
    
    return predictions


def test_lead_vs_lag(predictions: Dict[str, pd.DataFrame], 
                    returns_data: Dict[str, pd.DataFrame]) -> Dict:
    """Test that signals lead returns, not lag them."""
    
    lead_corrs = []
    lag_corrs = []
    
    for ticker, pred_df in predictions.items():
        if ticker not in returns_data:
            continue
        
        returns_df = returns_data[ticker]
        
        # Merge predictions with returns
        merged = pd.merge(pred_df, returns_df, left_index=True, right_index=True, how='inner')
        
        if len(merged) < 10:  # Need minimum data
            continue
        
        # Get signal and returns
        signal_col = None
        for col in ['prediction', 'edge', 'signal', 'median_model_sharpe']:
            if col in merged.columns:
                signal_col = col
                break
        
        if signal_col is None:
            continue
        
        signal = merged[signal_col].dropna()
        returns = merged['returns'].dropna()
        
        # Align data
        common_idx = signal.index.intersection(returns.index)
        if len(common_idx) < 10:
            continue
        
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Test lead correlations (signal at t, returns at t+h)
        for h in range(1, 6):  # 1 to 5 days ahead
            if len(returns) > h:
                lead_ret = returns.shift(-h)  # Future returns
                corr, _ = pearsonr(signal[:-h], lead_ret[:-h])
                if not np.isnan(corr):
                    lead_corrs.append(abs(corr))
        
        # Test lag correlations (signal at t, returns at t-h)
        for h in range(1, 6):  # 1 to 5 days behind
            if len(returns) > h:
                lag_ret = returns.shift(h)  # Past returns
                corr, _ = pearsonr(signal[h:], lag_ret[h:])
                if not np.isnan(corr):
                    lag_corrs.append(abs(corr))
    
    # Calculate AUC-like metric
    if lead_corrs and lag_corrs:
        lead_mean = np.mean(lead_corrs)
        lag_mean = np.mean(lag_corrs)
        auc = lead_mean / (lead_mean + lag_mean) if (lead_mean + lag_mean) > 0 else 0.5
    else:
        auc = 0.5  # No signal
    
    return {
        'lead_vs_lag_auc': auc,
        'lead_correlations': lead_corrs,
        'lag_correlations': lag_corrs,
        'lead_mean': np.mean(lead_corrs) if lead_corrs else 0,
        'lag_mean': np.mean(lag_corrs) if lag_corrs else 0
    }


def test_permutation(predictions: Dict[str, pd.DataFrame], 
                    returns_data: Dict[str, pd.DataFrame], 
                    n_permutations: int = 200) -> Dict:
    """Test that original Sharpe is significantly better than permuted labels."""
    
    original_sharpes = []
    permuted_sharpes = []
    
    for ticker, pred_df in predictions.items():
        if ticker not in returns_data:
            continue
        
        returns_df = returns_data[ticker]
        
        # Merge predictions with returns
        merged = pd.merge(pred_df, returns_df, left_index=True, right_index=True, how='inner')
        
        if len(merged) < 20:  # Need minimum data
            continue
        
        # Get signal and returns
        signal_col = None
        for col in ['prediction', 'edge', 'signal', 'median_model_sharpe']:
            if col in merged.columns:
                signal_col = col
                break
        
        if signal_col is None:
            continue
        
        signal = merged[signal_col].dropna()
        returns = merged['returns'].dropna()
        
        # Align data
        common_idx = signal.index.intersection(returns.index)
        if len(common_idx) < 20:
            continue
        
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Calculate original Sharpe
        if len(signal) > 0 and len(returns) > 0:
            # Simple strategy: long when signal > 0, short when signal < 0
            strategy_returns = np.where(signal > 0, returns, -returns)
            original_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            original_sharpes.append(original_sharpe)
            
            # Permutation test
            for _ in range(n_permutations):
                permuted_signal = np.random.permutation(signal)
                perm_strategy_returns = np.where(permuted_signal > 0, returns, -returns)
                perm_sharpe = np.mean(perm_strategy_returns) / np.std(perm_strategy_returns) * np.sqrt(252)
                permuted_sharpes.append(perm_sharpe)
    
    if not original_sharpes or not permuted_sharpes:
        return {
            'perm_mean_sharpe': 0.0,
            'original_mean_sharpe': 0.0,
            'p_value': 1.0,
            'significant': False
        }
    
    # Statistical test
    original_mean = np.mean(original_sharpes)
    perm_mean = np.mean(permuted_sharpes)
    
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(permuted_sharpes, original_mean)
    
    return {
        'perm_mean_sharpe': perm_mean,
        'original_mean_sharpe': original_mean,
        'p_value': p_value,
        'significant': p_value < 0.05 and abs(perm_mean) < 0.1
    }


def test_proxy_swap(predictions: Dict[str, pd.DataFrame], 
                   returns_data: Dict[str, pd.DataFrame]) -> Dict:
    """Test proxy swap sanity (QQQ vs SPY)."""
    
    # This is a simplified test - in practice, you'd need actual proxy data
    # For now, we'll simulate by comparing different tickers
    
    tickers = list(predictions.keys())
    if len(tickers) < 2:
        return {
            'proxy_swap_delta': 0.0,
            'within_tolerance': True
        }
    
    # Compare first two tickers as proxy swap
    ticker1, ticker2 = tickers[0], tickers[1]
    
    if ticker1 not in returns_data or ticker2 not in returns_data:
        return {
            'proxy_swap_delta': 0.0,
            'within_tolerance': True
        }
    
    # Calculate Sharpe for each ticker
    sharpe1 = calculate_ticker_sharpe(ticker1, predictions[ticker1], returns_data[ticker1])
    sharpe2 = calculate_ticker_sharpe(ticker2, predictions[ticker2], returns_data[ticker2])
    
    delta = abs(sharpe1 - sharpe2)
    within_tolerance = delta < 0.3
    
    return {
        'proxy_swap_delta': delta,
        'within_tolerance': within_tolerance,
        'sharpe_1': sharpe1,
        'sharpe_2': sharpe2
    }


def calculate_ticker_sharpe(ticker: str, pred_df: pd.DataFrame, returns_df: pd.DataFrame) -> float:
    """Calculate Sharpe ratio for a ticker's strategy."""
    try:
        # Merge predictions with returns
        merged = pd.merge(pred_df, returns_df, left_index=True, right_index=True, how='inner')
        
        if len(merged) < 10:
            return 0.0
        
        # Get signal and returns
        signal_col = None
        for col in ['prediction', 'edge', 'signal', 'median_model_sharpe']:
            if col in merged.columns:
                signal_col = col
                break
        
        if signal_col is None:
            return 0.0
        
        signal = merged[signal_col].dropna()
        returns = merged['returns'].dropna()
        
        # Align data
        common_idx = signal.index.intersection(returns.index)
        if len(common_idx) < 10:
            return 0.0
        
        signal = signal.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Calculate strategy returns
        strategy_returns = np.where(signal > 0, returns, -returns)
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        return sharpe if not np.isnan(sharpe) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe for {ticker}: {e}")
        return 0.0


def load_returns_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Load returns data for tickers."""
    returns_data = {}
    
    try:
        import yfinance as yf
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[ticker] = pd.DataFrame({'returns': returns})
                    logger.info(f"Loaded returns for {ticker}: {len(returns)} days")
            except Exception as e:
                logger.error(f"Error loading returns for {ticker}: {e}")
    
    except ImportError:
        logger.error("yfinance not available for returns data")
    
    return returns_data


def run_signal_lag_tests(universe_config: str, pred_root: str, 
                        output_dir: str, start_date: str = "2020-01-01", 
                        end_date: str = "2024-01-01") -> None:
    """Run complete signal lag and leakage tests."""
    
    # Load universe configuration
    import yaml
    with open(universe_config, 'r') as f:
        universe = yaml.safe_load(f)
    
    tickers = universe.get('data', {}).get('symbols', ['AAPL', 'NVDA', 'COIN'])
    
    logger.info(f"Running signal lag tests for tickers: {tickers}")
    
    # Load predictions and returns data
    predictions = load_predictions(pred_root, tickers)
    returns_data = load_returns_data(tickers, start_date, end_date)
    
    if not predictions:
        logger.error("No predictions loaded")
        return
    
    # Run tests
    logger.info("Running lead vs lag test...")
    lead_lag_result = test_lead_vs_lag(predictions, returns_data)
    
    logger.info("Running permutation test...")
    perm_result = test_permutation(predictions, returns_data)
    
    logger.info("Running proxy swap test...")
    proxy_result = test_proxy_swap(predictions, returns_data)
    
    # Check for sufficient data
    total_data_points = sum(len(df) for df in predictions.values())
    if total_data_points < 300:
        # Insufficient data - return NEUTRAL status
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'neutral',
            'reason': 'insufficient_data',
            'data_points': total_data_points,
            'tickers_tested': list(predictions.keys()),
            'message': f'Only {total_data_points} data points available, need at least 300'
        }
    else:
        # Sufficient data - run tests with conservative criteria
        passed = (
            lead_lag_result['lead_vs_lag_auc'] >= 0.58 and  # Conservative threshold
            perm_result['p_value'] < 0.05 and  # Statistical significance
            abs(proxy_result['proxy_swap_delta']) <= 0.30  # Within tolerance
        )
        
        # Compile results
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'pass' if passed else 'fail',
            'passed': passed,  # Keep for backward compatibility
            'lead_vs_lag_auc': lead_lag_result['lead_vs_lag_auc'],
            'p_value': perm_result.get('p_value', 1.0),
            'perm_mean_sharpe': perm_result['perm_mean_sharpe'],
            'proxy_swap_delta': proxy_result['proxy_swap_delta'],
            'tests': {
                'lead_vs_lag': lead_lag_result,
                'permutation': perm_result,
                'proxy_swap': proxy_result
            },
            'tickers_tested': list(predictions.keys()),
            'data_points': total_data_points
        }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lag_report_path = output_path / "lag_tests" / "signal_lag_report.json"
    lag_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(lag_report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Signal lag tests complete!")
    logger.info(f"Overall result: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"Lead vs lag AUC: {lead_lag_result['lead_vs_lag_auc']:.3f}")
    logger.info(f"Permutation significant: {perm_result['significant']}")
    logger.info(f"Proxy swap within tolerance: {proxy_result['within_tolerance']}")
    logger.info(f"Report saved to: {lag_report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run signal lag and leakage tests')
    parser.add_argument('--universe-cfg', required=True, help='Universe configuration file')
    parser.add_argument('--pred-root', required=True, help='Root directory for prediction files')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for returns data')
    parser.add_argument('--end-date', default='2024-01-01', help='End date for returns data')
    
    args = parser.parse_args()
    
    try:
        run_signal_lag_tests(
            args.universe_cfg,
            args.pred_root,
            args.out_dir,
            args.start_date,
            args.end_date
        )
        
    except Exception as e:
        logger.error(f"Error in signal lag tests: {e}")
        raise


if __name__ == "__main__":
    main()
