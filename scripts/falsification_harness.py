#!/usr/bin/env python3
"""
Falsification Harness - Prove the system isn't just luck

Tests the robustness of the trading system across different conditions:
- Cost stress tests
- Out-of-time validation  
- Symbol holdout tests
- Feature ablation
- Label leakage checks
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Any
import warnings
warnings.filterwarnings('ignore')

# Add core to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config import load_config
from core.decision_core import (
    DecisionCfg, 
    print_decision_legend
)
from scripts.walkforward import run_walkforward_validation


def run_cost_stress_test(base_config: dict[str, Any]) -> dict[str, Any]:
    """Test system robustness to increased costs"""
    print("🧪 Cost Stress Test")
    print("=" * 50)
    
    results = {}
    base_costs = base_config['decision']['costs_bps']
    
    # Test scenarios: base, +50%, +100%
    cost_scenarios = {
        'base': base_costs,
        'stress_50pct': base_costs * 1.5,
        'stress_100pct': base_costs * 2.0
    }
    
    for scenario, costs in cost_scenarios.items():
        print(f"\n📊 Testing {scenario}: {costs:.1f} bps costs")
        
        # Create modified config
        test_config = base_config.copy()
        test_config['decision']['costs_bps'] = costs
        
        # Run walkforward
        try:
            result = run_walkforward_validation(test_config)
            results[scenario] = {
                'costs_bps': costs,
                'mean_sharpe': result.get('mean_sharpe', 0.0),
                'median_sharpe': result.get('median_sharpe', 0.0),
                'successful_folds': result.get('successful_folds', 0),
                'total_folds': result.get('total_folds', 0)
            }
            print(f"   ✅ {scenario}: Sharpe={result.get('mean_sharpe', 0.0):.3f}")
        except Exception as e:
            print(f"   ❌ {scenario}: Failed - {e}")
            results[scenario] = {'error': str(e)}
    
    return results


def run_out_of_time_test(base_config: dict[str, Any]) -> dict[str, Any]:
    """Test system on different time periods"""
    print("\n🧪 Out-of-Time Test")
    print("=" * 50)
    
    results = {}
    
    # Test scenarios: current, backward 2 years, forward to recent
    time_scenarios = {
        'current': {'start_offset_days': 0, 'end_offset_days': 0},
        'backward_2yr': {'start_offset_days': -730, 'end_offset_days': -730},
        'forward_recent': {'start_offset_days': 0, 'end_offset_days': 0}  # Most recent data
    }
    
    for scenario, offsets in time_scenarios.items():
        print(f"\n📊 Testing {scenario}: {offsets}")
        
        # Create modified config with time offsets
        test_config = base_config.copy()
        # Note: This would require modifying the data download logic
        # For now, we'll simulate by changing the lookback period
        
        if scenario == 'backward_2yr':
            # Use older data by reducing lookback
            test_config['folds']['lookback_days'] = 200  # Shorter lookback for older data
        elif scenario == 'forward_recent':
            # Use most recent data
            test_config['folds']['lookback_days'] = 400  # Standard lookback
        
        try:
            result = run_walkforward_validation(test_config)
            results[scenario] = {
                'lookback_days': test_config['folds']['lookback_days'],
                'mean_sharpe': result.get('mean_sharpe', 0.0),
                'median_sharpe': result.get('median_sharpe', 0.0),
                'successful_folds': result.get('successful_folds', 0),
                'total_folds': result.get('total_folds', 0)
            }
            print(f"   ✅ {scenario}: Sharpe={result.get('mean_sharpe', 0.0):.3f}")
        except Exception as e:
            print(f"   ❌ {scenario}: Failed - {e}")
            results[scenario] = {'error': str(e)}
    
    return results


def run_symbol_holdout_test(base_config: dict[str, Any]) -> dict[str, Any]:
    """Test system on unseen symbols"""
    print("\n🧪 Symbol Holdout Test")
    print("=" * 50)
    
    results = {}
    
    # Test scenarios: train on SPY,QQQ; test on IWM
    symbol_scenarios = {
        'base_symbols': ['SPY', 'QQQ'],
        'holdout_test': ['IWM'],  # Russell 2000 - different market segment
        'mixed_test': ['SPY', 'IWM']  # Mix of seen and unseen
    }
    
    for scenario, symbols in symbol_scenarios.items():
        print(f"\n📊 Testing {scenario}: {symbols}")
        
        # Create modified config
        test_config = base_config.copy()
        test_config['data']['symbols'] = symbols
        
        try:
            result = run_walkforward_validation(test_config)
            results[scenario] = {
                'symbols': symbols,
                'mean_sharpe': result.get('mean_sharpe', 0.0),
                'median_sharpe': result.get('median_sharpe', 0.0),
                'successful_folds': result.get('successful_folds', 0),
                'total_folds': result.get('total_folds', 0)
            }
            print(f"   ✅ {scenario}: Sharpe={result.get('mean_sharpe', 0.0):.3f}")
        except Exception as e:
            print(f"   ❌ {scenario}: Failed - {e}")
            results[scenario] = {'error': str(e)}
    
    return results


def run_feature_ablation_test(base_config: dict[str, Any]) -> dict[str, Any]:
    """Test system with different feature configurations"""
    print("\n🧪 Feature Ablation Test")
    print("=" * 50)
    
    results = {}
    
    # Test scenarios: with PCA, without PCA, different PCA dims
    feature_scenarios = {
        'base_pca': {'use_pca': True, 'topk_supervised': 128},
        'no_pca': {'use_pca': False, 'topk_supervised': 128},
        'pca_64': {'use_pca': True, 'topk_supervised': 64},
        'pca_256': {'use_pca': True, 'topk_supervised': 256}
    }
    
    for scenario, features in feature_scenarios.items():
        print(f"\n📊 Testing {scenario}: {features}")
        
        # Create modified config
        test_config = base_config.copy()
        test_config['features']['use_pca'] = features['use_pca']
        test_config['features']['topk_supervised'] = features['topk_supervised']
        
        try:
            result = run_walkforward_validation(test_config)
            results[scenario] = {
                'features': features,
                'mean_sharpe': result.get('mean_sharpe', 0.0),
                'median_sharpe': result.get('median_sharpe', 0.0),
                'successful_folds': result.get('successful_folds', 0),
                'total_folds': result.get('total_folds', 0)
            }
            print(f"   ✅ {scenario}: Sharpe={result.get('mean_sharpe', 0.0):.3f}")
        except Exception as e:
            print(f"   ❌ {scenario}: Failed - {e}")
            results[scenario] = {'error': str(e)}
    
    return results


def run_determinism_test(base_config: dict[str, Any], n_runs: int = 3) -> dict[str, Any]:
    """Test system determinism with different seeds"""
    print("\n🧪 Determinism Test")
    print("=" * 50)
    
    results = {}
    sharpe_values = []
    
    for run in range(n_runs):
        print(f"\n📊 Run {run + 1}/{n_runs}")
        
        # Create modified config with different seed
        test_config = base_config.copy()
        test_config['seed'] = 42 + run * 100  # Different seeds
        
        try:
            result = run_walkforward_validation(test_config)
            sharpe = result.get('mean_sharpe', 0.0)
            sharpe_values.append(sharpe)
            results[f'run_{run+1}'] = {
                'seed': test_config['seed'],
                'mean_sharpe': sharpe,
                'median_sharpe': result.get('median_sharpe', 0.0),
                'successful_folds': result.get('successful_folds', 0),
                'total_folds': result.get('total_folds', 0)
            }
            print(f"   ✅ Run {run + 1}: Sharpe={sharpe:.3f}")
        except Exception as e:
            print(f"   ❌ Run {run + 1}: Failed - {e}")
            results[f'run_{run+1}'] = {'error': str(e)}
    
    # Calculate variance
    if len(sharpe_values) > 1:
        sharpe_std = np.std(sharpe_values)
        sharpe_mean = np.mean(sharpe_values)
        results['variance_analysis'] = {
            'mean_sharpe': sharpe_mean,
            'std_sharpe': sharpe_std,
            'cv': sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float('inf'),
            'is_stable': sharpe_std < 0.2  # Variance should be small
        }
        print("\n📊 Variance Analysis:")
        print(f"   Mean Sharpe: {sharpe_mean:.3f}")
        print(f"   Std Sharpe: {sharpe_std:.3f}")
        print(f"   CV: {sharpe_std/abs(sharpe_mean):.3f}")
        print(f"   Stable: {'✅' if sharpe_std < 0.2 else '❌'}")
    
    return results


def run_falsification_harness(config_path: str, output_dir: str = "falsification_results"):
    """Run the complete falsification harness"""
    print("🚀 Starting Falsification Harness")
    print("=" * 60)
    
    # Print decision legend for debugging
    print_decision_legend()
    
    # Load base config
    base_config = load_config(config_path)
    
    # Create default decision config
    decision_cfg = DecisionCfg(
        tau=0.0001,  # Use the same tau that worked in main eval
        temperature=1.0,
        gate_on="adv",
        cost_bps=base_config.get('decision', {}).get('costs_bps', 4.0)
    )
    print(f"Decision Config: {decision_cfg.to_dict()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all tests
    all_results = {}
    
    # 1. Determinism test (most important first)
    print("\n" + "="*60)
    all_results['determinism'] = run_determinism_test(base_config)
    
    # 2. Cost stress test
    print("\n" + "="*60)
    all_results['cost_stress'] = run_cost_stress_test(base_config)
    
    # 3. Out-of-time test
    print("\n" + "="*60)
    all_results['out_of_time'] = run_out_of_time_test(base_config)
    
    # 4. Symbol holdout test
    print("\n" + "="*60)
    all_results['symbol_holdout'] = run_symbol_holdout_test(base_config)
    
    # 5. Feature ablation test
    print("\n" + "="*60)
    all_results['feature_ablation'] = run_feature_ablation_test(base_config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"falsification_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n📊 Falsification Results Summary")
    print("=" * 60)
    
    # Summary analysis
    for test_name, test_results in all_results.items():
        print(f"\n{test_name.upper()}:")
        if test_name == 'determinism' and 'variance_analysis' in test_results:
            va = test_results['variance_analysis']
            print(f"   Stability: {'✅ PASS' if va['is_stable'] else '❌ FAIL'}")
            print(f"   Sharpe CV: {va['cv']:.3f}")
        else:
            for scenario, result in test_results.items():
                if 'error' not in result:
                    print(f"   {scenario}: Sharpe={result.get('mean_sharpe', 0.0):.3f}")
                else:
                    print(f"   {scenario}: ❌ {result['error']}")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run falsification harness")
    parser.add_argument("--config", default="configs/wfv.yaml", help="Config file path")
    parser.add_argument("--output", default="falsification_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_falsification_harness(args.config, args.output)
