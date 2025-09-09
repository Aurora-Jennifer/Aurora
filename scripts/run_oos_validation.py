#!/usr/bin/env python3
"""
Out-of-Sample Validation Runner
Runs validation across multiple time slices with proper gate criteria
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_oos_config(config_file: str) -> Dict:
    """Load OOS validation configuration."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def run_slice_validation(slice_config: Dict, universe_config: str, 
                        grid_config: str, output_dir: str, dump_preds: bool = False) -> Dict:
    """Run validation for a single time slice."""
    
    slice_name = slice_config['name']
    train_period = slice_config['train']
    test_period = slice_config['test']
    
    logger.info(f"Running validation for slice: {slice_name}")
    logger.info(f"Train: {train_period['start']} to {train_period['end']}")
    logger.info(f"Test: {test_period['start']} to {test_period['end']}")
    
    # Create slice-specific output directory
    slice_output_dir = Path(output_dir) / "oos_validation" / slice_name
    slice_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary config with slice-specific date ranges
    temp_config = {
        'data': {
            'start_date': train_period['start'],
            'end_date': test_period['end'],
            'train_end_date': train_period['end'],
            'test_start_date': test_period['start']
        },
        'walkforward': {
            'train_start': train_period['start'],
            'train_end': train_period['end'],
            'test_start': test_period['start'],
            'test_end': test_period['end']
        }
    }
    
    temp_config_file = slice_output_dir / "temp_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(temp_config, f)
    
    try:
        # Run universe validation for this slice
        cmd = [
            'python', 'scripts/run_universe.py',
            '--universe-cfg', universe_config,
            '--grid-cfg', grid_config,
            '--out-dir', str(slice_output_dir),
            '--train-start', train_period['start'],
            '--train-end', train_period['end'],
            '--test-start', test_period['start'],
            '--test-end', test_period['end'],
            '--feature-lookback-days', '252'
        ]
        
        if dump_preds:
            cmd.append('--dump-preds')
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        slice_result = {
            'slice_name': slice_name,
            'train_period': train_period,
            'test_period': test_period,
            'status': 'passed' if result.returncode == 0 else 'failed',
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None,
            'tickers_total': 0,
            'tickers_pass': 0,
            'gate_passes': []
        }
        
        # Analyze results for gate passes
        if result.returncode == 0:
            slice_result.update(analyze_slice_results(slice_output_dir, slice_config))
        
        return slice_result
        
    except Exception as e:
        logger.error(f"Error running slice {slice_name}: {e}")
        return {
            'slice_name': slice_name,
            'status': 'error',
            'error': str(e),
            'tickers_total': 0,
            'tickers_pass': 0
        }
    
    finally:
        # Clean up temporary config
        if temp_config_file.exists():
            temp_config_file.unlink()


def analyze_slice_results(slice_output_dir: Path, slice_config: Dict) -> Dict:
    """Analyze slice results for gate passes."""
    
    # Use the same gate criteria as the main universe run
    # Just check if the universe-level gate_pass column is True
    gate_criteria = slice_config.get('gate_criteria', {})
    min_sharpe_vs_bh = gate_criteria.get('min_sharpe_vs_bh', 0.10)
    max_drawdown_vs_bh = gate_criteria.get('max_drawdown_vs_bh', 0.05)
    min_trades = gate_criteria.get('min_trades', 5)
    
    tickers_total = 0
    tickers_pass = 0
    gate_passes = []
    
    # Look for leaderboard first (universe-level gate results)
    leaderboard_files = list(slice_output_dir.glob("**/leaderboard.csv"))
    
    if leaderboard_files:
        # Use leaderboard which has universe-level gate results
        for leaderboard_file in leaderboard_files:
            try:
                df = pd.read_csv(leaderboard_file)
                if len(df) == 0:
                    continue
                
                tickers_total = len(df)
                tickers_pass = len(df[df['gate_pass'] == True])
                
                # Extract gate passes
                for _, row in df[df['gate_pass'] == True].iterrows():
                    gate_passes.append({
                        'ticker': row['ticker'],
                        'strategy_id': 'universe_leader',
                        'sharpe': row.get('best_median_sharpe', 0),
                        'bh_sharpe': row.get('best_vs_BH', 0),
                        'trades': row.get('median_trades', 0),
                        'mdd': row.get('median_mdd', 0)
                    })
                
                return {
                    'tickers_total': tickers_total,
                    'tickers_pass': tickers_pass,
                    'gate_passes': gate_passes
                }
                
            except Exception as e:
                logger.error(f"Error analyzing leaderboard {leaderboard_file}: {e}")
                continue
    
    # Fallback: Find all grid result files
    grid_files = list(slice_output_dir.glob("**/*_grid.csv"))
    
    for grid_file in grid_files:
        try:
            df = pd.read_csv(grid_file)
            if len(df) == 0:
                continue
            
            ticker = grid_file.stem.replace('_grid', '')
            tickers_total += 1
            
            # Calculate buy-and-hold baseline
            bh_sharpe = calculate_buy_and_hold_sharpe(ticker, slice_config)
            
            # Check if any strategies passed the universe-level gate
            has_gate_pass = False
            for _, row in df.iterrows():
                # Use the universe-level gate_pass column if available
                if 'gate_pass' in row and row['gate_pass'] == True:
                    has_gate_pass = True
                    strategy_sharpe = row.get('median_model_sharpe', 0)
                    strategy_trades = row.get('mean_trades', 0)
                    strategy_mdd = row.get('median_max_drawdown', 0)
                    
                    gate_passes.append({
                        'ticker': ticker,
                        'strategy_id': row.get('strategy_id', 'unknown'),
                        'sharpe': strategy_sharpe,
                        'bh_sharpe': bh_sharpe,
                        'trades': strategy_trades,
                        'mdd': strategy_mdd
                    })
            
            if has_gate_pass:
                tickers_pass += 1
                
        except Exception as e:
            logger.error(f"Error analyzing {grid_file}: {e}")
            continue
    
    return {
        'tickers_total': tickers_total,
        'tickers_pass': tickers_pass,
        'gate_passes': gate_passes
    }


def calculate_buy_and_hold_sharpe(ticker: str, slice_config: Dict) -> float:
    """Calculate buy-and-hold Sharpe ratio for the test period."""
    try:
        import yfinance as yf
        
        test_period = slice_config['test']
        data = yf.download(ticker, 
                          start=test_period['start'], 
                          end=test_period['end'], 
                          progress=False)
        
        if len(data) == 0:
            return 0.0
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        returns = data['Close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Convert to scalar values to avoid Series comparison issues
        mean_return = float(returns.mean())
        std_return = float(returns.std())
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)
        return sharpe if not np.isnan(sharpe) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating BH Sharpe for {ticker}: {e}")
        return 0.0


def run_oos_validation(universe_config: str, oos_config: str, 
                      grid_config: str, output_dir: str, dump_preds: bool = False) -> None:
    """Run complete OOS validation across all slices."""
    
    # Load configuration
    config = load_oos_config(oos_config)
    slices = config['slices']
    
    logger.info(f"Running OOS validation for {len(slices)} slices")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run validation for each slice
    slice_results = []
    all_slices_pass = True
    
    for slice_config in slices:
        slice_result = run_slice_validation(
            slice_config, universe_config, grid_config, output_dir, dump_preds
        )
        slice_results.append(slice_result)
        
        if slice_result['status'] != 'passed' or slice_result['tickers_pass'] == 0:
            all_slices_pass = False
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'slices': slice_results,
        'all_slices_pass': all_slices_pass,
        'total_slices': len(slices),
        'passed_slices': sum(1 for s in slice_results if s['status'] == 'passed' and s['tickers_pass'] > 0),
        'total_tickers': sum(s['tickers_total'] for s in slice_results),
        'total_tickers_pass': sum(s['tickers_pass'] for s in slice_results)
    }
    
    # Save summary report
    oos_report_path = output_path / "oos_validation" / "oos_report.json"
    oos_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(oos_report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"OOS validation complete!")
    logger.info(f"Total slices: {summary['total_slices']}")
    logger.info(f"Passed slices: {summary['passed_slices']}")
    logger.info(f"All slices pass: {all_slices_pass}")
    logger.info(f"Report saved to: {oos_report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run out-of-sample validation')
    parser.add_argument('--universe-cfg', required=True, help='Universe configuration file')
    parser.add_argument('--oos-cfg', required=True, help='OOS validation configuration file')
    parser.add_argument('--grid-cfg', required=True, help='Grid configuration file')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--dump-preds', action='store_true', help='Dump OOS predictions for signal lag tests')
    
    args = parser.parse_args()
    
    try:
        run_oos_validation(
            args.universe_cfg,
            args.oos_cfg,
            args.grid_cfg,
            args.out_dir,
            dump_preds=args.dump_preds
        )
        
    except Exception as e:
        logger.error(f"Error in OOS validation: {e}")
        raise


if __name__ == "__main__":
    main()
