#!/usr/bin/env python3
"""
Neutralization A/B/C Test Script

Tests three neutralization approaches using the existing runner infrastructure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging

# Import from our existing modules
from ml.runner_universe import newey_west_t, partial_neutralize_series, _load_yaml
from ml.risk_neutralization import create_sector_dummies, winsorize_by_date, cs_zscore_features
from ml.panel_builder import build_panel_from_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_ic_metrics(predictions: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Calculate IC and Rank-IC metrics."""
    # Merge predictions and returns
    merged = predictions.merge(returns, on=['date', 'symbol'], how='inner')
    
    # Calculate daily IC and Rank-IC
    daily_ic = []
    daily_rank_ic = []
    
    for date in merged['date'].unique():
        date_data = merged[merged['date'] == date]
        if len(date_data) < 10:  # Need minimum observations
            continue
            
        # Calculate IC (correlation between prediction and return)
        ic = date_data['prediction'].corr(date_data['excess_ret_fwd_5'])
        if not np.isnan(ic):
            daily_ic.append(ic)
        
        # Calculate Rank-IC (Spearman correlation)
        rank_ic = date_data['prediction'].corr(date_data['excess_ret_fwd_5'], method='spearman')
        if not np.isnan(rank_ic):
            daily_rank_ic.append(rank_ic)
    
    if not daily_ic or not daily_rank_ic:
        return {
            'ic_mean': 0.0, 'ic_std': 0.0, 'ic_tstat': 0.0,
            'rank_ic_mean': 0.0, 'rank_ic_std': 0.0, 'rank_ic_tstat': 0.0,
            'n_days': 0
        }
    
    ic_series = pd.Series(daily_ic)
    rank_ic_series = pd.Series(daily_rank_ic)
    
    return {
        'ic_mean': ic_series.mean(),
        'ic_std': ic_series.std(),
        'ic_tstat': newey_west_t(ic_series),
        'rank_ic_mean': rank_ic_series.mean(),
        'rank_ic_std': rank_ic_series.std(),
        'rank_ic_tstat': newey_west_t(rank_ic_series),
        'n_days': len(daily_ic)
    }


def calculate_decile_spread(predictions: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Calculate Q10-Q1 decile spread."""
    merged = predictions.merge(returns, on=['date', 'symbol'], how='inner')
    
    spreads = []
    for date in merged['date'].unique():
        date_data = merged[merged['date'] == date]
        if len(date_data) < 20:  # Need enough for deciles
            continue
            
        # Rank predictions and calculate decile returns
        date_data = date_data.sort_values('prediction')
        n = len(date_data)
        q1_size = n // 10
        q10_size = n // 10
        
        q1_ret = date_data.iloc[:q1_size]['excess_ret_fwd_5'].mean()
        q10_ret = date_data.iloc[-q10_size:]['excess_ret_fwd_5'].mean()
        
        if not np.isnan(q1_ret) and not np.isnan(q10_ret):
            spreads.append(q10_ret - q1_ret)
    
    return np.mean(spreads) if spreads else 0.0


def calculate_monthly_ic_stability(predictions: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Calculate monthly IC stability metrics."""
    merged = predictions.merge(returns, on=['date', 'symbol'], how='inner')
    merged['year_month'] = merged['date'].dt.to_period('M')
    
    monthly_ics = []
    for period in merged['year_month'].unique():
        period_data = merged[merged['year_month'] == period]
        if len(period_data) < 10:
            continue
            
        # Calculate monthly IC
        ic = period_data['prediction'].corr(period_data['excess_ret_fwd_5'])
        if not np.isnan(ic):
            monthly_ics.append(ic)
    
    if not monthly_ics:
        return {'monthly_ic_mean': 0.0, 'monthly_ic_std': 0.0, 'positive_months': 0, 'total_months': 0}
    
    monthly_ic_series = pd.Series(monthly_ics)
    positive_months = (monthly_ic_series > 0).sum()
    
    return {
        'monthly_ic_mean': monthly_ic_series.mean(),
        'monthly_ic_std': monthly_ic_series.std(),
        'positive_months': positive_months,
        'total_months': len(monthly_ics),
        'sign_rate': positive_months / len(monthly_ics) if monthly_ics else 0.0
    }


def run_neutralization_ab_test(universe_cfg_path: str, oos_days: int = 180, embargo_days: int = 5) -> dict:
    """Run the neutralization A/B/C test."""
    
    logger.info("🚀 Starting neutralization A/B/C test...")
    
    # Load universe config
    universe_cfg = _load_yaml(universe_cfg_path)
    
    # Build panel dataset
    logger.info("📊 Building panel dataset...")
    temp_file = "temp_panel.csv"
    panel = build_panel_from_universe(universe_cfg_path, temp_file)
    
    # Clean up temp file
    try:
        os.remove(temp_file)
    except:
        pass
    
    # Apply winsorization
    ret_cols = ['ret_fwd_3', 'ret_fwd_5', 'ret_fwd_10']
    available_ret_cols = [col for col in ret_cols if col in panel.columns]
    if available_ret_cols:
        panel = winsorize_by_date(panel, available_ret_cols, 0.025, 0.975)
    
    # Create train/test split
    panel = panel.sort_values(['symbol', 'date'])
    train_end = panel['date'].max() - pd.Timedelta(days=oos_days + embargo_days)
    test_start = panel['date'].max() - pd.Timedelta(days=oos_days)
    
    train = panel[panel['date'] <= train_end].copy()
    test = panel[panel['date'] >= test_start].copy()
    
    logger.info(f"📆 Train: {train['date'].min()} → {train['date'].max()} ({len(train)} rows)")
    logger.info(f"📆 Test: {test['date'].min()} → {test['date'].max()} ({len(test)} rows)")
    
    # Apply leakage guard
    train = train.groupby('symbol').apply(lambda x: x.shift(1)).reset_index(drop=True)
    test = test.groupby('symbol').apply(lambda x: x.shift(1)).reset_index(drop=True)
    
    # Remove rows with NaN after shift
    train = train.dropna()
    test = test.dropna()
    
    # Feature columns (exclude forbidden/forward-looking)
    forbidden_cols = ['symbol', 'date', 'ret_fwd_3', 'excess_ret_fwd_3', 
                     'ret_fwd_5', 'excess_ret_fwd_5', 'ret_fwd_10', 'excess_ret_fwd_10']
    feature_cols = [col for col in train.columns if col not in forbidden_cols]
    
    # Apply per-date z-scoring
    train = cs_zscore_features(train, feature_cols)
    test = cs_zscore_features(test, feature_cols)
    
    # Create cross-sectional target
    train['cs_target'] = train.groupby('date')['excess_ret_fwd_5'].rank(pct=True)
    test['cs_target'] = test.groupby('date')['excess_ret_fwd_5'].rank(pct=True)
    
    # Create exposure factors (simplified)
    logger.info("🔍 Creating exposure factors...")
    
    # Market beta (simplified - using correlation with market)
    train_market_ret = train.groupby('date')['excess_ret_fwd_5'].mean()
    test_market_ret = test.groupby('date')['excess_ret_fwd_5'].mean()
    
    # Create market beta using a simpler approach - just use a constant for now
    train['market_beta'] = 1.0  # Simplified - use constant beta
    test['market_beta'] = 1.0   # Simplified - use constant beta
    
    # Size factor (simplified - using log market cap proxy)
    train['size_factor'] = np.log(train['close'] * train['volume'].rolling(60, min_periods=10).mean())
    test['size_factor'] = np.log(test['close'] * test['volume'].rolling(60, min_periods=10).mean())
    
    # Sector dummies
    train_sectors = create_sector_dummies(train['symbol'].unique())
    test_sectors = create_sector_dummies(test['symbol'].unique())
    
    # Merge sector dummies
    train = train.merge(train_sectors, left_on='symbol', right_index=True, how='left')
    test = test.merge(test_sectors, left_on='symbol', right_index=True, how='left')
    
    # Fill NaN sector dummies with 0
    sector_cols = [col for col in train.columns if col.startswith('sector_')]
    train[sector_cols] = train[sector_cols].fillna(0)
    test[sector_cols] = test[sector_cols].fillna(0)
    
    # Prepare exposure columns
    expo_cols = ['market_beta', 'size_factor'] + sector_cols
    
    # Remove rows with NaN exposures
    train = train.dropna(subset=expo_cols + ['cs_target'])
    test = test.dropna(subset=expo_cols + ['cs_target'])
    
    logger.info(f"📊 Final train: {len(train)} rows, test: {len(test)} rows")
    
    # Import XGBoost here to avoid issues
    import xgboost as xgb
    
    # Run the three neutralization approaches
    results = {}
    
    for approach in ['A', 'B', 'C']:
        logger.info(f"🔄 Testing approach {approach}...")
        
        if approach == 'A':  # Target-only neutralization
            # Neutralize training targets
            train_processed = train.copy()
            test_processed = test.copy()
            
            def neutralize_targets_per_date(group):
                date = group['date'].iloc[0]
                symbols = group['symbol'].tolist()
                
                # Get exposures for this date
                exposures = group[expo_cols].values
                target = group['cs_target'].values
                
                if len(target) > len(expo_cols) + 1:
                    # Full neutralization (γ=1)
                    neutralized_target = partial_neutralize_series(
                        pd.Series(target, index=symbols),
                        pd.DataFrame(exposures, index=symbols, columns=expo_cols),
                        gamma=1.0
                    )
                    group['cs_target'] = neutralized_target.values
                
                return group
            
            train_processed = train_processed.groupby('date', group_keys=False).apply(neutralize_targets_per_date)
            
            # Train model
            X_train = train_processed[feature_cols]
            y_train = train_processed['cs_target']
            
            model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            # Make predictions (no neutralization)
            X_test = test_processed[feature_cols]
            predictions = model.predict(X_test)
            
        elif approach == 'B':  # Prediction-only neutralization
            # Use raw targets
            train_processed = train.copy()
            test_processed = test.copy()
            
            # Train model
            X_train = train_processed[feature_cols]
            y_train = train_processed['cs_target']
            
            model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            # Make predictions and neutralize them
            X_test = test_processed[feature_cols]
            raw_predictions = model.predict(X_test)
            
            # Neutralize predictions per date
            test_processed['prediction'] = raw_predictions
            predictions = []
            
            for date in test_processed['date'].unique():
                date_data = test_processed[test_processed['date'] == date]
                symbols = date_data['symbol'].tolist()
                
                # Get exposures and predictions for this date
                exposures = date_data[expo_cols].values
                preds = date_data['prediction'].values
                
                if len(preds) > len(expo_cols) + 1:
                    # Partial neutralization (γ=0.6)
                    neutralized_preds = partial_neutralize_series(
                        pd.Series(preds, index=symbols),
                        pd.DataFrame(exposures, index=symbols, columns=expo_cols),
                        gamma=0.6
                    )
                    predictions.extend(neutralized_preds.values)
                else:
                    predictions.extend(preds)
            
        elif approach == 'C':  # One projection layer
            # Use raw targets
            train_processed = train.copy()
            test_processed = test.copy()
            
            # Train model
            X_train = train_processed[feature_cols]
            y_train = train_processed['cs_target']
            
            model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train, y_train)
            
            # Make predictions and apply projection matrix
            X_test = test_processed[feature_cols]
            raw_predictions = model.predict(X_test)
            
            # Apply projection matrix per date
            test_processed['prediction'] = raw_predictions
            predictions = []
            
            for date in test_processed['date'].unique():
                date_data = test_processed[test_processed['date'] == date]
                symbols = date_data['symbol'].tolist()
                
                # Get exposures and predictions for this date
                X = date_data[expo_cols].values
                y = date_data['prediction'].values
                
                if len(y) > len(expo_cols) + 1:
                    # Create projection matrix M = I - X(X'X)^(-1)X'
                    try:
                        # Ensure proper dtype
                        X = X.astype(np.float64)
                        y = y.astype(np.float64)
                        XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1]))
                        M = np.eye(len(y)) - X @ XtX_inv @ X.T
                        projected_y = M @ y
                        predictions.extend(projected_y)
                    except (np.linalg.LinAlgError, np._core._exceptions._UFuncInputCastingError):
                        # Fallback to original predictions if matrix is singular or dtype issues
                        predictions.extend(y)
                else:
                    predictions.extend(y)
        
        # Create predictions DataFrame
        test_processed['prediction'] = predictions
        
        # Calculate metrics
        predictions_df = test_processed[['date', 'symbol', 'prediction']].copy()
        returns_df = test_processed[['date', 'symbol', 'excess_ret_fwd_5']].copy()
        
        ic_metrics = calculate_ic_metrics(predictions_df, returns_df)
        decile_spread = calculate_decile_spread(predictions_df, returns_df)
        monthly_stability = calculate_monthly_ic_stability(predictions_df, returns_df)
        
        results[approach] = {
            'ic_metrics': ic_metrics,
            'decile_spread': decile_spread,
            'monthly_stability': monthly_stability
        }
        
        logger.info(f"✅ Approach {approach} completed:")
        logger.info(f"   Rank-IC: {ic_metrics['rank_ic_mean']:.4f} (t={ic_metrics['rank_ic_tstat']:.2f})")
        logger.info(f"   Decile spread: {decile_spread:.4f}")
        logger.info(f"   Monthly sign rate: {monthly_stability['sign_rate']:.2f}")
    
    return results


def main():
    """Main function to run the neutralization test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run neutralization A/B/C test')
    parser.add_argument('--universe-cfg', required=True, help='Universe config path')
    parser.add_argument('--oos-days', type=int, default=180, help='OOS days')
    parser.add_argument('--embargo-days', type=int, default=5, help='Embargo days')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the test
    results = run_neutralization_ab_test(
        args.universe_cfg,
        args.oos_days,
        args.embargo_days
    )
    
    # Save results
    results_file = out_dir / 'neutralization_ab_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("NEUTRALIZATION A/B/C TEST RESULTS")
    print("="*80)
    
    for approach, result in results.items():
        ic = result['ic_metrics']
        monthly = result['monthly_stability']
        
        print(f"\nApproach {approach}:")
        print(f"  Rank-IC: {ic['rank_ic_mean']:.4f} (NW-t: {ic['rank_ic_tstat']:.2f})")
        print(f"  IC: {ic['ic_mean']:.4f} (NW-t: {ic['ic_tstat']:.2f})")
        print(f"  Decile spread: {result['decile_spread']:.4f}")
        print(f"  Monthly sign rate: {monthly['sign_rate']:.2f} ({monthly['positive_months']}/{monthly['total_months']})")
        print(f"  N days: {ic['n_days']}")
    
    # Find winner
    winner = max(results.keys(), key=lambda x: results[x]['ic_metrics']['rank_ic_tstat'])
    print(f"\n🏆 WINNER: Approach {winner}")
    print(f"   Best Rank-IC t-stat: {results[winner]['ic_metrics']['rank_ic_tstat']:.2f}")
    
    print(f"\n📊 Results saved to: {results_file}")


if __name__ == "__main__":
    main()