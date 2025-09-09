#!/usr/bin/env python3
"""
Feature Ablation Analysis Runner
Runs ablation tests by dropping feature families and measuring impact
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
import glob
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ablation_config(config_file: str) -> Dict:
    """Load ablation test configuration."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def find_baseline_strategies(baseline_root: str, universe_config: str) -> Dict[str, pd.DataFrame]:
    """Find baseline strategies from previous runs."""
    
    baseline_strategies = {}
    
    # Load universe configuration to get tickers
    with open(universe_config, 'r') as f:
        universe = yaml.safe_load(f)
    
    tickers = universe.get('data', {}).get('symbols', ['AAPL', 'NVDA', 'COIN'])
    
    # Look for baseline results
    baseline_patterns = [
        f"{baseline_root}/**/*_grid.csv",
        f"{baseline_root}/**/grid_results.csv",
        f"{baseline_root}/**/*_results.csv"
    ]
    
    for pattern in baseline_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                if len(df) == 0:
                    continue
                
                # Extract ticker from filename
                ticker = None
                for t in tickers:
                    if t.lower() in file_path.lower():
                        ticker = t
                        break
                
                if ticker and ticker not in baseline_strategies:
                    baseline_strategies[ticker] = df
                    logger.info(f"Found baseline strategies for {ticker}: {len(df)} strategies")
                    
            except Exception as e:
                logger.error(f"Error loading baseline file {file_path}: {e}")
    
    return baseline_strategies


def select_top_strategies(df: pd.DataFrame, top_k: int = 10, 
                         min_sharpe: float = 0.0, min_trades: int = 5) -> pd.DataFrame:
    """Select top strategies based on criteria."""
    
    # Filter by minimum criteria
    filtered_df = df.copy()
    
    if 'median_model_sharpe' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['median_model_sharpe'] >= min_sharpe]
    elif 'median_sharpe' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['median_sharpe'] >= min_sharpe]
    
    if 'mean_trades' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['mean_trades'] >= min_trades]
    elif 'median_trades' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['median_trades'] >= min_trades]
    
    # Sort by Sharpe ratio
    sharpe_col = 'median_model_sharpe' if 'median_model_sharpe' in filtered_df.columns else 'median_sharpe'
    if sharpe_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sharpe_col, ascending=False)
    
    # Take top K
    return filtered_df.head(top_k)


def create_ablation_config(base_config: str, family_config: Dict, 
                          output_dir: Path) -> str:
    """Create configuration with specific features dropped."""
    
    # Load base configuration
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add feature filtering
    if 'data' not in config:
        config['data'] = {}
    
    config['data']['feature_filters'] = {
        'drop_prefixes': family_config['drop_prefixes']
    }
    
    # Save modified config
    ablation_config_path = output_dir / f"ablation_{family_config['name']}.yaml"
    with open(ablation_config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(ablation_config_path)


def run_ablation_for_family(family_config: Dict, baseline_strategies: Dict[str, pd.DataFrame],
                           universe_config: str, base_grid_config: str, 
                           output_dir: Path) -> Dict:
    """Run ablation test for a specific feature family."""
    
    family_name = family_config['name']
    logger.info(f"Running ablation test for family: {family_name}")
    
    # Create ablation-specific output directory
    family_output_dir = output_dir / "ablation" / family_name
    family_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create modified configuration
    ablation_config = create_ablation_config(base_grid_config, family_config, family_output_dir)
    
    try:
        # Run universe with modified configuration
        result = subprocess.run([
            'python', '-m', 'ml.runner_universe',
            '--universe-cfg', universe_config,
            '--grid-cfg', ablation_config,
            '--out-dir', str(family_output_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"Ablation test failed for {family_name}: {result.stderr}")
            return {
                'family': family_name,
                'status': 'failed',
                'error': result.stderr,
                'baseline_sharpe': 0.0,
                'ablated_sharpe': 0.0,
                'delta_sharpe': 0.0
            }
        
        # Analyze results
        return analyze_ablation_results(family_name, baseline_strategies, family_output_dir)
        
    except Exception as e:
        logger.error(f"Error running ablation for {family_name}: {e}")
        return {
            'family': family_name,
            'status': 'error',
            'error': str(e),
            'baseline_sharpe': 0.0,
            'ablated_sharpe': 0.0,
            'delta_sharpe': 0.0
        }


def analyze_ablation_results(family_name: str, baseline_strategies: Dict[str, pd.DataFrame],
                           family_output_dir: Path) -> Dict:
    """Analyze ablation results and compare with baseline."""
    
    # Find ablation results
    ablation_files = list(family_output_dir.glob("**/*_grid.csv"))
    
    if not ablation_files:
        return {
            'family': family_name,
            'status': 'no_results',
            'baseline_sharpe': 0.0,
            'ablated_sharpe': 0.0,
            'delta_sharpe': 0.0
        }
    
    # Calculate baseline Sharpe (median across all tickers)
    baseline_sharpes = []
    for ticker, df in baseline_strategies.items():
        if 'median_model_sharpe' in df.columns:
            sharpe = df['median_model_sharpe'].median()
        elif 'median_sharpe' in df.columns:
            sharpe = df['median_sharpe'].median()
        else:
            continue
        
        if not np.isnan(sharpe):
            baseline_sharpes.append(sharpe)
    
    baseline_median_sharpe = np.median(baseline_sharpes) if baseline_sharpes else 0.0
    
    # Calculate ablated Sharpe
    ablated_sharpes = []
    for ablation_file in ablation_files:
        try:
            df = pd.read_csv(ablation_file)
            if 'median_model_sharpe' in df.columns:
                sharpe = df['median_model_sharpe'].median()
            elif 'median_sharpe' in df.columns:
                sharpe = df['median_sharpe'].median()
            else:
                continue
            
            if not np.isnan(sharpe):
                ablated_sharpes.append(sharpe)
        except Exception as e:
            logger.error(f"Error analyzing {ablation_file}: {e}")
    
    ablated_median_sharpe = np.median(ablated_sharpes) if ablated_sharpes else 0.0
    
    # Calculate delta
    delta_sharpe = ablated_median_sharpe - baseline_median_sharpe
    
    return {
        'family': family_name,
        'status': 'completed',
        'baseline_sharpe': baseline_median_sharpe,
        'ablated_sharpe': ablated_median_sharpe,
        'delta_sharpe': delta_sharpe,
        'baseline_count': len(baseline_sharpes),
        'ablated_count': len(ablated_sharpes)
    }


def run_ablation_analysis(universe_config: str, ablation_config: str, 
                         baseline_root: str, base_grid_config: str, 
                         output_dir: str) -> None:
    """Run complete feature ablation analysis."""
    
    # Load configurations
    config = load_ablation_config(ablation_config)
    families = config['families']
    test_params = config.get('test_params', {})
    baseline_selection = config.get('baseline_selection', {})
    
    logger.info(f"Running ablation analysis for {len(families)} feature families")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find baseline strategies
    baseline_strategies = find_baseline_strategies(baseline_root, universe_config)
    
    if not baseline_strategies:
        logger.error("No baseline strategies found")
        return
    
    # Select top strategies for each ticker
    top_k = baseline_selection.get('top_k_strategies', 10)
    min_sharpe = baseline_selection.get('min_sharpe', 0.0)
    min_trades = baseline_selection.get('min_trades', 5)
    
    selected_baselines = {}
    for ticker, df in baseline_strategies.items():
        selected = select_top_strategies(df, top_k, min_sharpe, min_trades)
        selected_baselines[ticker] = selected
        logger.info(f"Selected {len(selected)} top strategies for {ticker}")
    
    # Run ablation tests for each family
    ablation_results = []
    
    for family_config in families:
        result = run_ablation_for_family(
            family_config, selected_baselines, universe_config, 
            base_grid_config, output_path
        )
        ablation_results.append(result)
    
    # Generate summary report
    summary_data = []
    max_delta_threshold = test_params.get('max_delta_threshold', 0.75)
    
    for result in ablation_results:
        summary_data.append({
            'family': result['family'],
            'baseline_median_sharpe': result['baseline_sharpe'],
            'ablated_median_sharpe': result['ablated_sharpe'],
            'delta_sharpe': result['delta_sharpe'],
            'status': result['status']
        })
    
    # Save detailed results
    ablation_summary_path = output_path / "ablation" / "delta_sharpe.csv"
    ablation_summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(ablation_summary_path, index=False)
    
    # Check for significant drops
    significant_drops = []
    for result in ablation_results:
        if result['status'] == 'completed':
            delta_pct = abs(result['delta_sharpe']) / max(result['baseline_sharpe'], 0.01)
            if delta_pct > max_delta_threshold:
                significant_drops.append({
                    'family': result['family'],
                    'delta_pct': delta_pct,
                    'delta_sharpe': result['delta_sharpe']
                })
    
    # Save comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'families_tested': len(families),
        'families_completed': sum(1 for r in ablation_results if r['status'] == 'completed'),
        'significant_drops': significant_drops,
        'max_delta_threshold': max_delta_threshold,
        'results': ablation_results
    }
    
    report_path = output_path / "ablation" / "ablation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Ablation analysis complete!")
    logger.info(f"Families tested: {len(families)}")
    logger.info(f"Families completed: {report['families_completed']}")
    logger.info(f"Significant drops: {len(significant_drops)}")
    logger.info(f"Summary saved to: {ablation_summary_path}")
    logger.info(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run feature ablation analysis')
    parser.add_argument('--universe-cfg', required=True, help='Universe configuration file')
    parser.add_argument('--ablation-cfg', required=True, help='Ablation test configuration file')
    parser.add_argument('--baseline-root', required=True, help='Root directory for baseline results')
    parser.add_argument('--base-grid-cfg', required=True, help='Base grid configuration file')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        run_ablation_analysis(
            args.universe_cfg,
            args.ablation_cfg,
            args.baseline_root,
            args.base_grid_cfg,
            args.out_dir
        )
        
    except Exception as e:
        logger.error(f"Error in ablation analysis: {e}")
        raise


if __name__ == "__main__":
    main()
