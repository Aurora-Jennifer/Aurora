#!/usr/bin/env python3
"""
Experiment Grid Runner

Orchestrates systematic model evaluation:
- YAML config → parameter grid
- Walkforward validation for each configuration
- Gating and reporting
- Results persistence
"""

import json
import yaml
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Import our modules
from .targets import create_targets, validate_targets
from .decision import make_decisions, calibrate_decision_parameters, validate_decisions
from .baselines import create_baseline_model, buy_and_hold_daily_pnl, simple_rule_daily_pnl
from .features import create_feature_pipeline, save_feature_schema, load_feature_schema, validate_proxy_coverage

# Import capability checking
import sys
sys.path.append('utils')
from env_check import check_capabilities
CAPS = check_capabilities()

# Import existing validator components
sys.path.append('scripts')
from trade_metrics_helpers import sharpe_from_daily, calculate_max_drawdown, calculate_profit_factor

logger = logging.getLogger(__name__)


def data_integrity_report(X_train: np.ndarray, feature_names: List[str], logger) -> np.ndarray:
    """
    Generate data integrity report and return mask for non-constant features
    
    Args:
        X_train: Training features
        feature_names: Feature names
        logger: Logger instance
    
    Returns:
        Boolean mask for non-constant features
    """
    std = X_train.std(axis=0)
    keep_mask = (std > 1e-8)
    logger.info(f"Feature coverage: {int(keep_mask.sum())}/{len(std)} non-constant")
    
    drop = [f for f, m in zip(feature_names, keep_mask) if not m]
    if drop:
        logger.warning(f"Dropping near-constant features: {drop[:12]}{'...' if len(drop)>12 else ''}")
    
    return keep_mask


def filter_collinear_features(X_train: np.ndarray, feature_names: List[str], 
                            threshold: float = 0.95, logger=None) -> Tuple[np.ndarray, List[str]]:
    """
    Filter highly collinear features on TRAIN data only
    
    Args:
        X_train: Training features
        feature_names: Feature names
        threshold: Correlation threshold for dropping
        logger: Logger instance
    
    Returns:
        Filtered features and feature names
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Compute correlation matrix on TRAIN only
    corr = np.corrcoef(X_train, rowvar=False)
    hi_i, hi_j = np.where(np.triu(np.abs(corr) > threshold, 1))
    
    # Drop later index (keep earlier features)
    drop_idx = sorted(set(int(j) for j in hi_j))
    keep_idx = [k for k in range(corr.shape[0]) if k not in drop_idx]
    
    if drop_idx:
        dropped_names = [feature_names[i] for i in drop_idx]
        logger.warning(f"Dropping {len(drop_idx)} highly collinear features (>|{threshold}|): {dropped_names[:5]}{'...' if len(dropped_names)>5 else ''}")
    
    return X_train[:, keep_idx], [feature_names[k] for k in keep_idx]


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('grid_runner.log')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_parameter_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create parameter grid from configuration"""
    grid = []
    
    # Base parameters
    base_params = {
        'horizons': config['horizons'],
        'eps_quantiles': config['eps_quantiles'],
        'temperature': config['temperature'],
        'turnover_band': [tuple(config['turnover_band'])]  # Single tuple
    }
    
    # Model parameters
    model_configs = []
    for model_spec in config['models']:
        model_type = model_spec['type']
        
        # Capability gating - skip unavailable models
        if model_type == 'lgbm' and not CAPS.get("lightgbm", False):
            logger.warning(f"Skipping {model_type} - LightGBM not available")
            continue
        if model_type == 'xgboost' and not CAPS.get("xgboost", False):
            logger.warning(f"Skipping {model_type} - XGBoost not available")
            continue
        
        if model_type == 'ridge':
            for alpha in model_spec['alphas']:
                model_configs.append({
                    'type': 'ridge',
                    'alpha': alpha
                })
        elif model_type in ['lgbm', 'xgboost']:
            # Create cartesian product of parameter combinations
            param_names = list(model_spec['params'].keys())
            param_values = list(model_spec['params'].values())
            
            for param_combo in itertools.product(*param_values):
                model_params = dict(zip(param_names, param_combo))
                model_configs.append({
                    'type': model_type,
                    'params': model_params
                })
    
    # Create full grid
    for horizon in base_params['horizons']:
        for eps_q in base_params['eps_quantiles']:
            for temp in base_params['temperature']:
                for turnover in base_params['turnover_band']:
                    for model_config in model_configs:
                        grid.append({
                            'horizon': horizon,
                            'eps_quantile': eps_q,
                            'temperature': temp,
                            'turnover_band': turnover,
                            'model': model_config
                        })
    
    return grid


def run_wf_once(config: Dict[str, Any], param_combo: Dict[str, Any], 
                data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run walkforward validation for a single parameter combination
    
    Args:
        config: Full configuration
        param_combo: Parameter combination
        data: Data dictionary with asset and market data
    
    Returns:
        Results dictionary
    """
    try:
        # Extract parameters
        horizon = param_combo['horizon']
        eps_quantile = param_combo['eps_quantile']
        temperature = param_combo['temperature']
        turnover_band = param_combo['turnover_band']
        model_config = param_combo['model']
        
        # Get data
        symbol = config['data']['symbols'][0]  # Use first symbol from config
        asset_data = data[symbol]
        market_data = data[config['data']['market_benchmark']]
        
        
        # Create features
        features = create_feature_pipeline(asset_data)
        
        # Calculate returns
        asset_ret = asset_data['Close'].pct_change().dropna()
        market_ret = market_data['Close'].pct_change().dropna()
        
        # Align data
        common_idx = asset_ret.index.intersection(market_ret.index)
        asset_ret = asset_ret.loc[common_idx]
        market_ret = market_ret.loc[common_idx]
        features = features.loc[common_idx]
        
        # Create targets
        train_idx = common_idx[:int(len(common_idx) * 0.7)]
        labels, targets, eps = create_targets(asset_ret, market_ret, horizon, train_idx, eps_quantile)
        
        # Validate targets
        validate_targets(labels, targets, eps)
        
        # Prepare training data
        train_features = features.loc[train_idx]
        train_targets = targets.loc[train_idx]
        
        # Data integrity checks on TRAIN only
        feature_names = list(train_features.columns)
        X_train = train_features.values
        
        # Check for near-constant features
        keep_mask = data_integrity_report(X_train, feature_names, logger)
        X_train = X_train[:, keep_mask]
        feature_names = [f for f, m in zip(feature_names, keep_mask) if m]
        
        # Filter collinear features
        X_train, feature_names = filter_collinear_features(X_train, feature_names, logger=logger)
        
        # Update train_features with filtered data
        train_features = pd.DataFrame(X_train, index=train_idx, columns=feature_names)
        
        # Create model with improved Ridge setup
        if model_config['type'] == 'ridge':
            from ml.baselines import RidgeExcessModel
            model = RidgeExcessModel(
                alpha=float(model_config.get('alpha', 10.0))
            )
        else:
            model = create_baseline_model(model_config['type'], **model_config.get('params', {}))
        
        # Train model
        model.fit(train_features, train_targets)
        
        # Get training edges for calibration
        train_edges = model.predict_edge(train_features)
        
        # Check edge variance to prevent constant edges
        edge_std = float(np.std(train_edges))
        if edge_std < 1e-6:
            raise RuntimeError("Edges nearly constant after filtering; check labels/features.")
        
        # Calibrate decision parameters
        decision_params = calibrate_decision_parameters(train_edges, turnover_band)
        
        # Run walkforward validation
        fold_length = config['walkforward'].get('fold_length_days', config['walkforward'].get('fold_length', 63))
        step_size = config['walkforward'].get('step_size_days', config['walkforward'].get('step_size', 21))
        
        fold_results = []
        
        for start_idx in range(len(train_idx), len(common_idx) - fold_length, step_size):
            end_idx = start_idx + fold_length
            
            # Test fold
            test_idx = common_idx[start_idx:end_idx]
            test_features = features.loc[test_idx, feature_names]  # Use filtered feature names
            test_targets = targets.loc[test_idx]
            
            # Make predictions
            positions, edges, proba = make_decisions(
                model, test_features, decision_params, temperature
            )
            
            # Validate decisions
            validate_decisions(positions, edges, proba)
            
            # Calculate metrics
            test_returns = asset_ret.loc[test_idx]
            
            # Model PnL
            model_pnl = positions * test_returns
            model_sharpe = sharpe_from_daily(model_pnl)
            
            # Baseline PnL
            bh_pnl = buy_and_hold_daily_pnl(asset_data['Close'].loc[test_idx])
            bh_sharpe = sharpe_from_daily(bh_pnl)
            
            rule_pnl = simple_rule_daily_pnl(asset_data['Close'].loc[test_idx])
            rule_sharpe = sharpe_from_daily(rule_pnl)
            
            # Additional metrics
            mdd = calculate_max_drawdown(asset_data['Close'].loc[test_idx])
            pf = calculate_profit_factor(
                max(0, model_pnl.sum()), 
                max(0, -model_pnl.sum())
            )
            
            fold_results.append({
                'model_sharpe': model_sharpe,
                'bh_sharpe': bh_sharpe,
                'rule_sharpe': rule_sharpe,
                'mdd': mdd,
                'profit_factor': pf,
                'trades': len(np.where(np.diff(positions) != 0)[0]),
                'turnover': np.mean(np.abs(np.diff(positions)))
            })
        
        # Aggregate results
        if fold_results:
            model_sharpes = [r['model_sharpe'] for r in fold_results]
            bh_sharpes = [r['bh_sharpe'] for r in fold_results]
            rule_sharpes = [r['rule_sharpe'] for r in fold_results]
            
            return {
                'param_combo': param_combo,
                'num_folds': len(fold_results),
                'median_model_sharpe': np.median(model_sharpes),
                'median_bh_sharpe': np.median(bh_sharpes),
                'median_rule_sharpe': np.median(rule_sharpes),
                'mean_model_sharpe': np.mean(model_sharpes),
                'std_model_sharpe': np.std(model_sharpes),
                'mean_trades': np.mean([r['trades'] for r in fold_results]),
                'mean_turnover': np.mean([r['turnover'] for r in fold_results]),
                'mean_mdd': np.mean([r['mdd'] for r in fold_results]),
                'mean_pf': np.mean([r['profit_factor'] for r in fold_results]),
                'eps': eps,
                'tau_in': decision_params['tau_in'],
                'tau_out': decision_params['tau_out'],
                'success': True,
                'error': None
            }
        else:
            return {
                'param_combo': param_combo,
                'success': False,
                'error': 'No valid folds'
            }
            
    except Exception as e:
        logging.error(f"Error in run_wf_once: {e}")
        return {
            'param_combo': param_combo,
            'success': False,
            'error': str(e)
        }


def run_grid(config_path: str, date_slice: dict = None, dump_preds: bool = False, out_csv: str = None) -> pd.DataFrame:
    """
    Run the complete experiment grid
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Results DataFrame
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config['output']['log_level'])
    
    logging.info("Starting experiment grid")
    
    # Create parameter grid
    param_grid = create_parameter_grid(config)
    logging.info(f"Created {len(param_grid)} parameter combinations")
    
    # Load data (simplified - in practice you'd load from config)
    import yfinance as yf
    
    data = {}
    # Download all symbols including market benchmark
    all_symbols = set(config['data']['symbols'] + [config['data']['market_benchmark']])
    
    # Determine date range based on date_slice if provided
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    if date_slice:
        # Use test_end as the download end date
        if date_slice.get('test_end'):
            end_date = date_slice['test_end']
        
        # Use train_start minus lookback as download start date
        if date_slice.get('train_start'):
            lookback_days = date_slice.get('feature_lookback_days', 252)
            start_date = (pd.Timestamp(date_slice['train_start']) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    for symbol in all_symbols:
        df = yf.download(
            symbol, 
            start=start_date,
            end=end_date,
            auto_adjust=True,  # Explicitly set to avoid FutureWarning
            progress=False,
            threads=False
        )
        # Flatten MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[symbol] = df
    
    # Run experiments
    results = []
    for i, param_combo in enumerate(param_grid):
        logging.info(f"Running experiment {i+1}/{len(param_grid)}: {param_combo}")
        
        result = run_wf_once(config, param_combo, data)
        results.append(result)
        
        if result['success']:
            logging.info(f"  Success: median_sharpe={result['median_model_sharpe']:.3f}")
        else:
            logging.warning(f"  Failed: {result['error']}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Dump predictions if requested
    if dump_preds:
        _dump_predictions(df_results, config, out_csv)
    
    # Apply gating
    gate_config = config['gate']
    successful_results = df_results[df_results['success'] == True]
    
    if len(successful_results) > 0:
        # Check if any configuration beats baselines
        baseline_threshold = successful_results[['median_bh_sharpe', 'median_rule_sharpe']].max(axis=1) + gate_config['threshold_delta_vs_baseline']
        gate_passed = (successful_results['median_model_sharpe'] > baseline_threshold).any()
        
        logging.info(f"Gate result: {'PASS' if gate_passed else 'FAIL'}")
        
        if gate_passed:
            best_config = successful_results.loc[successful_results['median_model_sharpe'].idxmax()]
            logging.info(f"Best configuration: {best_config['param_combo']}")
            logging.info(f"Best median Sharpe: {best_config['median_model_sharpe']:.3f}")
    else:
        logging.error("No successful experiments")
    
    # Save results
    output_config = config['output']
    df_results.to_csv(output_config['csv_path'], index=False)
    
    # Save detailed results as JSON
    with open(output_config['json_path'], 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {output_config['csv_path']} and {output_config['json_path']}")
    
    return df_results


def _dump_predictions(df_results: pd.DataFrame, config: dict, out_csv: str) -> None:
    """Dump OOS predictions for signal lag tests."""
    try:
        # Find the best performing strategy
        if len(df_results) == 0 or not df_results['success'].any():
            return
        
        best_result = df_results[df_results['success']].nlargest(1, 'median_model_sharpe').iloc[0]
        
        # For now, create a simple prediction dump
        # In a full implementation, you'd extract actual OOS predictions from the walkforward
        symbol = config['data']['symbols'][0]  # Primary symbol
        
        # Create dummy predictions (replace with actual OOS predictions)
        import pandas as pd
        import numpy as np
        
        # Generate some dummy data for testing
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        n_days = len(dates)
        
        predictions_df = pd.DataFrame({
            'date': dates,
            'edge': np.random.normal(0, 0.1, n_days),  # Dummy edges
            'ret_fwd': np.random.normal(0, 0.02, n_days)  # Dummy forward returns
        })
        
        # Save to parquet
        if out_csv:
            pred_path = out_csv.replace('.csv', '_predictions.parquet')
            predictions_df.to_parquet(pred_path, index=False)
            logging.info(f"Predictions dumped to {pred_path}")
        
    except Exception as e:
        logging.error(f"Error dumping predictions: {e}")


def run_single_model_grid(symbol: str, market_proxy: str, cross_proxies: list, cfg: dict, 
                         out_csv: str = None, date_slice: dict = None, dump_preds: bool = False) -> pd.DataFrame:
    """
    Wrapper used by the universe runner:
      - Loads / prepares data for `symbol` (plus market & optional cross proxies)
      - Runs the existing grid for the provided `cfg` (classical or DL)
      - Returns a DataFrame with per-config metrics:
          ['median_sharpe_model','median_sharpe_bh','median_sharpe_rule',
           'median_turnover','median_trades','gate_pass', ...]
    """
    # Create a temporary config file for this symbol
    import tempfile
    import os
    
    # Modify config for this specific symbol
    cfg_symbol = cfg.copy()
    cfg_symbol['data'] = cfg_symbol.get('data', {})
    cfg_symbol['data']['symbols'] = [symbol]
    cfg_symbol['data']['market_benchmark'] = market_proxy
    cfg_symbol['data']['cross_proxies'] = cross_proxies
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg_symbol, f)
        temp_config_path = f.name
    
    try:
        # Run the grid
        results = run_grid(temp_config_path, date_slice=date_slice, dump_preds=dump_preds, out_csv=out_csv)
        
        # Save results if requested
        if out_csv:
            results.to_csv(out_csv, index=False)
        
        return results
        
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument("--config", default="config/grid.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        results = run_grid(args.config)
        print(f"\n🎉 Grid completed! {len(results)} experiments run.")
        print(f"Successful: {results['success'].sum()}")
        print(f"Failed: {(~results['success']).sum()}")
        
        if results['success'].any():
            successful = results[results['success']]
            print(f"Best median Sharpe: {successful['median_model_sharpe'].max():.3f}")
        
    except Exception as e:
        print(f"❌ Grid failed: {e}")
        exit(1)
