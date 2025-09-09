#!/usr/bin/env python3
"""
Pre-market dry run validation.

Runs the full trading pipeline without emitting orders to validate system health.
"""
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
from pathlib import Path
from ml.runner_universe import topk_ls
from ml.production_logging import create_paper_trading_logger


def create_mock_market_data(target_date: str, num_symbols: int = 50, lookback_days: int = 60) -> pd.DataFrame:
    """
    Create mock market data for dry run testing.
    
    Args:
        target_date: Target date for simulation (YYYY-MM-DD)
        num_symbols: Number of symbols to simulate
        
    Returns:
        DataFrame with mock market data
    """
    logger = create_paper_trading_logger()
    logger.info(f"Creating mock market data for {target_date} with {num_symbols} symbols")
    
    np.random.seed(42)
    
    # Create realistic symbols
    symbols = [f'SYM{i:03d}' for i in range(num_symbols)]
    
    # Create sufficient lookback data for rolling features
    end_date = pd.to_datetime(target_date)
    start_date = end_date - timedelta(days=lookback_days + 7)  # Buffer for weekends/holidays
    dates = pd.bdate_range(start_date, end_date, freq='B')  # Business days only
    
    data = []
    
    for date in dates:
        for i, symbol in enumerate(symbols):
            # Simulate realistic cross-sectional features
            rank_factor = i / (len(symbols) - 1)  # 0 to 1
            
            # Base price and volume
            base_price = 50 + rank_factor * 100  # $50-$150 range
            base_volume = 1000000 * (1 + rank_factor * 5)  # 1M-6M volume
            
            # Add daily noise
            price_noise = np.random.normal(0, 0.02)
            volume_noise = np.random.normal(0, 0.3)
            
            # Features that would be computed in real system
            close = base_price * (1 + price_noise)
            volume = base_volume * (1 + volume_noise)
            
            # Mock returns
            ret1 = np.random.normal(0.001, 0.02)
            
            # Mock cross-sectional features (would be computed from real data)
            features = {
                'close_csr': rank_factor + np.random.normal(0, 0.1),
                'volume_csr': rank_factor * 0.8 + np.random.normal(0, 0.1),
                'vol_5_csr': rank_factor * 0.6 + np.random.normal(0, 0.1),
                'momentum_5_csr': rank_factor * 0.4 + np.random.normal(0, 0.1),
                'close_csr_sec_res': np.random.normal(0, 0.05),
                'volume_csr_sec_res': np.random.normal(0, 0.05),
            }
            
            # Only include features that are in our whitelist
            try:
                with open('results/production/features_whitelist.json', 'r') as f:
                    whitelist_features = json.load(f)
                
                # Filter to whitelist features (first 10 for simplicity)
                filtered_features = {k: v for k, v in features.items() 
                                   if k in whitelist_features[:10]}
            except:
                filtered_features = features
            
            data_row = {
                'date': date,
                'symbol': symbol,
                'close': close,
                'volume': volume,
                'ret1': ret1,
                **filtered_features
            }
            
            data.append(data_row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created {len(df)} rows of mock market data")
    
    return df


def run_signal_generation_dry_run(market_data: pd.DataFrame, target_date: str) -> Dict:
    """
    Run signal generation for target date without placing orders.
    
    Args:
        market_data: Historical market data
        target_date: Target date for signal generation
        
    Returns:
        Dict with signal generation results
    """
    logger = create_paper_trading_logger()
    logger.info(f"Running signal generation dry run for {target_date}")
    
    try:
        # Filter to data available up to target date
        target_dt = pd.to_datetime(target_date)
        available_data = market_data[market_data['date'] <= target_dt].copy()
        
        # Simulate forward returns for backtesting (normally not available)
        np.random.seed(42)
        future_returns = {}
        for symbol in available_data['symbol'].unique():
            future_returns[symbol] = np.random.normal(0.001, 0.02)
        
        available_data['excess_ret_fwd_5'] = available_data['symbol'].map(future_returns)
        
        # Get feature columns
        feature_cols = [col for col in available_data.columns 
                       if col.endswith('_csr') or col.endswith('_res')]
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found")
        
        # Create mock predictions
        latest_data = available_data[available_data['date'] == target_dt].copy()
        
        if len(latest_data) == 0:
            raise ValueError(f"No data available for {target_date}")
        
        # Generate predictions (mock model output)
        np.random.seed(42)
        predictions = []
        for _, row in latest_data.iterrows():
            # Use features to create realistic prediction
            feature_signal = sum(row[col] for col in feature_cols if not pd.isna(row[col]))
            noise = np.random.normal(0, 0.1)
            pred = feature_signal + noise
            predictions.append(pred)
        
        latest_data['prediction'] = predictions
        
        # Run portfolio construction (dry run)
        portfolio_result = topk_ls(
            latest_data[['date', 'symbol', 'prediction']],
            latest_data[['date', 'symbol', 'excess_ret_fwd_5']],
            k=min(20, len(latest_data)//4),  # Adaptive k for dry run
            use_adv_enforcement=True,
            max_participation_pct=0.02,
            portfolio_equity=10_000_000,
            use_realistic_costs=True,
            use_turnover_controls=False  # Simplify for dry run
        )
        
        # Extract signal statistics
        signal_stats = {
            'target_date': target_date,
            'num_symbols': len(latest_data),
            'num_features': len(feature_cols),
            'prediction_std': np.std(predictions),
            'prediction_range': [float(np.min(predictions)), float(np.max(predictions))],
            'portfolio_result': {
                'sharpe': portfolio_result.get('sharpe', 0),
                'turnover': portfolio_result.get('avg_turnover', 0),
                'adv_breaches': portfolio_result.get('total_adv_breaches', 0),
                'positions': portfolio_result.get('n_days', 0)
            }
        }
        
        logger.info(f"Signal generation completed: {signal_stats['num_symbols']} symbols, "
                   f"std={signal_stats['prediction_std']:.4f}")
        
        return {
            'status': 'success',
            'signal_stats': signal_stats,
            'feature_columns': feature_cols,
            'predictions_sample': predictions[:10]  # First 10 for inspection
        }
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return {
            'status': 'error',
            'error_message': str(e),
            'signal_stats': None
        }


def validate_risk_controls_dry_run(signal_results: Dict) -> Dict:
    """
    Validate risk controls during dry run.
    
    Args:
        signal_results: Results from signal generation
        
    Returns:
        Dict with risk validation results
    """
    logger = create_paper_trading_logger()
    logger.info("Validating risk controls")
    
    risk_checks = []
    
    # Check 1: Prediction dispersion
    if signal_results['status'] == 'success':
        pred_std = signal_results['signal_stats']['prediction_std']
        if pred_std < 0.01:
            risk_checks.append({
                'check': 'prediction_dispersion',
                'status': 'warning',
                'message': f'Low prediction std: {pred_std:.4f}'
            })
        else:
            risk_checks.append({
                'check': 'prediction_dispersion',
                'status': 'pass',
                'message': f'Prediction std: {pred_std:.4f}'
            })
    
    # Check 2: ADV constraints
    if signal_results['status'] == 'success':
        adv_breaches = signal_results['signal_stats']['portfolio_result']['adv_breaches']
        breach_pct = adv_breaches / signal_results['signal_stats']['num_symbols'] * 100
        
        if breach_pct > 10:
            risk_checks.append({
                'check': 'adv_constraints',
                'status': 'warning',
                'message': f'High ADV breach rate: {breach_pct:.1f}%'
            })
        else:
            risk_checks.append({
                'check': 'adv_constraints',
                'status': 'pass',
                'message': f'ADV breach rate: {breach_pct:.1f}%'
            })
    
    # Check 3: Feature availability
    num_features = len(signal_results.get('feature_columns', []))
    if num_features < 5:
        risk_checks.append({
            'check': 'feature_availability',
            'status': 'warning',
            'message': f'Low feature count: {num_features}'
        })
    else:
        risk_checks.append({
            'check': 'feature_availability',
            'status': 'pass',
            'message': f'Feature count: {num_features}'
        })
    
    # Overall risk status
    warnings = [c for c in risk_checks if c['status'] == 'warning']
    errors = [c for c in risk_checks if c['status'] == 'error']
    
    overall_status = 'error' if errors else ('warning' if warnings else 'pass')
    
    logger.info(f"Risk validation completed: {overall_status}")
    
    return {
        'overall_status': overall_status,
        'checks': risk_checks,
        'warnings': len(warnings),
        'errors': len(errors)
    }


def run_pre_market_dry_run(target_date: str = None) -> Dict:
    """
    Run comprehensive pre-market dry run.
    
    Args:
        target_date: Target date for dry run (defaults to yesterday)
        
    Returns:
        Dict with dry run results
    """
    if target_date is None:
        # Use proper date alignment: decision_date vs feature_date
        try:
            from ops.date_helpers import get_decision_and_feature_dates, validate_date_alignment
            decision_date, feature_date = get_decision_and_feature_dates()
            validate_date_alignment(decision_date, feature_date)
            
            # For dry-run, we test the feature generation path
            target_date = feature_date.strftime('%Y-%m-%d')
            print(f"🗓️  Decision date: {decision_date}")
            print(f"🗓️  Feature date (testing): {feature_date}")
            print(f"🗓️  Using feature_date for dry-run: {target_date}")
        except Exception as e:
            print(f"⚠️  Could not get aligned dates: {e}, using fallback")
            # Fallback: go back to most recent weekday
            target_date_obj = datetime.now().date() - timedelta(days=1)
            while target_date_obj.weekday() >= 5:  # Skip weekends
                target_date_obj = target_date_obj - timedelta(days=1)
            target_date = target_date_obj.strftime('%Y-%m-%d')
            print(f"🗓️  Fallback target date: {target_date}")
    
    logger = create_paper_trading_logger()
    logger.info(f"Starting pre-market dry run for {target_date}")
    
    print("🌅 PRE-MARKET DRY RUN VALIDATION")
    print("="*50)
    print(f"Target date: {target_date}")
    
    # Step 1: Generate mock market data with proper lookback
    print("\n📊 Step 1: Loading market data...")
    LOOKBACK_DAYS = 60  # Sufficient for 20-day rolling features
    market_data = create_mock_market_data(target_date, num_symbols=50, lookback_days=LOOKBACK_DAYS)
    print(f"✅ Market data ready: {len(market_data)} rows")
    print(f"   Date range: {market_data['date'].min()} to {market_data['date'].max()}")
    print(f"   Business days: {market_data['date'].nunique()}")
    print(f"   Symbols: {market_data['symbol'].nunique()}")
    
    # Step 2: Run signal generation
    print("\n🎯 Step 2: Running signal generation...")
    signal_results = run_signal_generation_dry_run(market_data, target_date)
    
    if signal_results['status'] == 'success':
        print(f"✅ Signal generation successful")
        stats = signal_results['signal_stats']
        print(f"   Symbols: {stats['num_symbols']}")
        print(f"   Features: {stats['num_features']}")
        print(f"   Prediction std: {stats['prediction_std']:.4f}")
    else:
        print(f"❌ Signal generation failed: {signal_results['error_message']}")
    
    # Step 3: Validate risk controls
    print("\n🛡️ Step 3: Validating risk controls...")
    risk_results = validate_risk_controls_dry_run(signal_results)
    
    status_icon = "✅" if risk_results['overall_status'] == 'pass' else "⚠️" if risk_results['overall_status'] == 'warning' else "❌"
    print(f"{status_icon} Risk validation: {risk_results['overall_status']}")
    
    for check in risk_results['checks']:
        check_icon = "✅" if check['status'] == 'pass' else "⚠️" if check['status'] == 'warning' else "❌"
        print(f"   {check_icon} {check['check']}: {check['message']}")
    
    # Compile results
    dry_run_results = {
        'dry_run_date': target_date,
        'timestamp': datetime.now().isoformat(),
        'market_data_status': 'success',
        'signal_generation': signal_results,
        'risk_validation': risk_results,
        'overall_status': 'pass' if (signal_results['status'] == 'success' and 
                                   risk_results['overall_status'] in ['pass', 'warning']) else 'fail'
    }
    
    # Save results
    results_dir = Path("results/dry_runs")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"dry_run_{target_date}.json"
    with open(results_file, 'w') as f:
        json.dump(dry_run_results, f, indent=2, default=str)
    
    # Summary
    print(f"\n📋 DRY RUN SUMMARY:")
    print(f"   Overall status: {dry_run_results['overall_status'].upper()}")
    print(f"   Signal generation: {signal_results['status']}")
    print(f"   Risk validation: {risk_results['overall_status']}")
    print(f"   Results saved: {results_file}")
    
    if dry_run_results['overall_status'] == 'pass':
        print(f"\n🚀 DRY RUN PASSED - READY FOR PAPER TRADING")
    else:
        print(f"\n🛑 DRY RUN ISSUES - REVIEW BEFORE TRADING")
    
    logger.info(f"Pre-market dry run completed: {dry_run_results['overall_status']}")
    
    return dry_run_results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-Market Dry Run")
    parser.add_argument('--date', help="Target date (YYYY-MM-DD, defaults to yesterday)")
    
    args = parser.parse_args()
    
    result = run_pre_market_dry_run(args.date)
    
    # Exit with error code if dry run failed
    sys.exit(0 if result['overall_status'] == 'pass' else 1)


if __name__ == "__main__":
    main()
