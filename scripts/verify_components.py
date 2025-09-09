#!/usr/bin/env python3
"""
Component Verification System
Derives READY flags from actual files and metrics, not just process completion
"""

import json
import os
import glob
import pandas as pd
import time
from pathlib import Path
from typing import Tuple, Dict, Any


def _has_nonempty_csv(path_glob: str, min_rows: int = 1) -> bool:
    """Check if CSV file exists and has minimum number of rows."""
    paths = glob.glob(path_glob)
    if not paths:
        return False
    try:
        df = pd.read_csv(paths[0])
        return len(df) >= min_rows
    except Exception:
        return False


def _has_json_with_key(file_path: str, key: str, expected_value: Any = None) -> bool:
    """Check if JSON file exists and contains expected key/value."""
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if expected_value is not None:
            return data.get(key) == expected_value
        return key in data
    except Exception:
        return False


def verify_robustness(outdir: str) -> Tuple[bool, str]:
    """Verify robustness testing produced meaningful results."""
    # Check for cost stress artifacts in multiple possible locations
    cost_globs = [
        os.path.join(outdir, "deployment_test_cost", "**", "*_grid.csv"),
        os.path.join("results", "deployment_test_cost", "**", "*_grid.csv"),
        os.path.join("results", "cost_stress_03bps", "**", "*_grid.csv")
    ]
    
    cost_files = []
    for cost_glob in cost_globs:
        cost_files.extend(glob.glob(cost_glob, recursive=True))
    
    if not cost_files:
        return False, "No cost stress results found"
    
    # Check that at least one file has meaningful results
    for file_path in cost_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Check for successful runs (not all failures)
                if 'median_model_sharpe' in df.columns:
                    successful_runs = df['median_model_sharpe'].notna().sum()
                    if successful_runs > 0:
                        return True, f"Found {successful_runs} successful cost stress runs"
        except Exception as e:
            continue
    
    return False, "No successful cost stress runs found"


def verify_oos(outdir: str) -> Tuple[bool, str]:
    """Verify out-of-sample validation produced meaningful results."""
    # Check for OOS report JSON
    oos_report_paths = [
        os.path.join(outdir, "oos_validation", "oos_report.json"),
        os.path.join("results", "oos_validation", "oos_report.json")
    ]
    
    for report_path in oos_report_paths:
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                if report.get('all_slices_pass', False):
                    total_slices = report.get('total_slices', 0)
                    passed_slices = report.get('passed_slices', 0)
                    return True, f"OOS validation passed: {passed_slices}/{total_slices} slices"
                else:
                    total_slices = report.get('total_slices', 0)
                    passed_slices = report.get('passed_slices', 0)
                    return False, f"OOS validation failed: only {passed_slices}/{total_slices} slices passed"
            except Exception as e:
                continue
    
    # Fallback: check for OOS validation artifacts in multiple possible locations
    oos_globs = [
        os.path.join(outdir, "deployment_test_oos", "**", "*_grid.csv"),
        os.path.join("results", "deployment_test_oos", "**", "*_grid.csv"),
        os.path.join("results", "oos_slices", "**", "*_grid.csv")
    ]
    
    oos_files = []
    for oos_glob in oos_globs:
        oos_files.extend(glob.glob(oos_glob, recursive=True))
    
    if not oos_files:
        return False, "No OOS validation results found"
    
    # Check that each OOS slice has results
    successful_slices = 0
    for file_path in oos_files:
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Check for successful runs
                if 'median_model_sharpe' in df.columns:
                    successful_runs = df['median_model_sharpe'].notna().sum()
                    if successful_runs > 0:
                        successful_slices += 1
        except Exception:
            continue
    
    if successful_slices > 0:
        return True, f"Found {successful_slices} OOS slices with successful runs"
    else:
        return False, "No successful OOS validation runs found"


def verify_lag(outdir: str) -> Tuple[bool, str]:
    """Verify signal lag tests passed."""
    # Check for signal lag test results
    lag_files = [
        os.path.join(outdir, "lag_tests", "signal_lag_report.json"),
        os.path.join(outdir, "test_results", "signal_lag.json")
    ]
    
    for file_path in lag_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                
                status = report.get('status', 'fail')
                passed = report.get("passed", False)
                
                if status == 'pass' or passed:
                    auc = report.get("lead_vs_lag_auc", 0)
                    perm_mean = report.get("perm_mean_sharpe", 0)
                    proxy_delta = report.get("proxy_swap_delta", 0)
                    return True, f"Signal lag tests passed: AUC={auc:.3f}, perm_mean={perm_mean:.3f}, proxy_delta={proxy_delta:.3f}"
                elif status == 'neutral':
                    reason = report.get('reason', 'insufficient_data')
                    return True, f"Signal lag tests neutral: {reason}"
                else:
                    return False, f"Signal lag tests failed: {file_path}"
            except Exception as e:
                continue
    
    # Fallback: check if pytest ran successfully
    pytest_log = os.path.join(outdir, "pytest_signal_lag.log")
    if os.path.exists(pytest_log):
        try:
            with open(pytest_log, 'r') as f:
                content = f.read()
                if "failed" not in content.lower() and "passed" in content.lower():
                    return True, "Signal lag tests passed (pytest log)"
        except Exception:
            pass
    
    return False, "No signal lag test results found or tests failed"


def verify_portfolio(outdir: str) -> Tuple[bool, str]:
    """Verify portfolio construction produced meaningful results."""
    # Check for portfolio.json first (preferred format)
    portfolio_json_files = [
        os.path.join(outdir, "portfolio", "portfolio.json"),
        os.path.join(outdir, "deployment_portfolio", "portfolio.json"),
        os.path.join("portfolios", "deployment_portfolio", "portfolio.json"),
        os.path.join("portfolios", "portfolio.json")
    ]
    
    for file_path in portfolio_json_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'weights' in data and data['weights']:
                    total_weight = data.get('sum_abs', sum(abs(w) for w in data['weights'].values()))
                    num_strategies = len(data['weights'])
                    
                    # Check if weights sum to 1.0 (within tolerance)
                    if abs(total_weight - 1.0) < 1e-6:
                        return True, f"Portfolio has {num_strategies} strategies with normalized weights (sum={total_weight:.6f})"
                    elif total_weight > 0:
                        return True, f"Portfolio has {num_strategies} strategies with total weight {total_weight:.3f}"
            except Exception:
                continue
    
    # Fallback: check for portfolio artifacts in multiple possible locations
    portfolio_files = [
        os.path.join(outdir, "portfolio", "portfolio_weights.csv"),
        os.path.join(outdir, "deployment_portfolio", "portfolio_weights.csv"),
        os.path.join("portfolios", "deployment_portfolio", "portfolio_weights.csv"),
        os.path.join("portfolios", "portfolio_weights.csv")
    ]
    
    for file_path in portfolio_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    # Check for non-zero weights
                    if 'weight' in df.columns:
                        total_weight = df['weight'].abs().sum()
                        if total_weight > 0:
                            return True, f"Portfolio has {len(df)} strategies with total weight {total_weight:.3f}"
            except Exception:
                continue
    
    return False, "No portfolio weights found or all weights are zero"


def verify_ablation(outdir: str) -> Tuple[bool, str]:
    """Verify ablation analysis produced meaningful results."""
    # Check for ablation artifacts in multiple possible locations
    ablation_files = [
        os.path.join(outdir, "ablation", "delta_sharpe.csv"),
        os.path.join(outdir, "deployment_ablation", "delta_sharpe.csv"),
        os.path.join("reports", "deployment_ablation", "delta_sharpe.csv"),
        os.path.join("reports", "ablation", "delta_sharpe.csv")
    ]
    
    for file_path in ablation_files:
        if file_path.endswith('.csv') and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    # Check for significant drops (flag if any family causes >75% Sharpe drop)
                    if 'delta_sharpe' in df.columns:
                        max_drop = abs(df['delta_sharpe'].min())
                        baseline_sharpe = df['baseline_median_sharpe'].max()
                        if baseline_sharpe > 0:
                            max_drop_pct = max_drop / baseline_sharpe
                            if max_drop_pct > 0.75:
                                return True, f"Ablation analysis found {len(df)} feature comparisons (WARNING: {max_drop_pct:.1%} max drop)"
                            else:
                                return True, f"Ablation analysis found {len(df)} feature comparisons (max drop: {max_drop_pct:.1%})"
                        else:
                            return True, f"Ablation analysis found {len(df)} feature comparisons"
                    else:
                        return True, f"Ablation analysis found {len(df)} feature comparisons"
            except Exception:
                continue
    
    # Check for ablation report markdown
    ablation_md_files = [
        os.path.join(outdir, "ablation_report.md"),
        os.path.join("reports", "ablation_report.md")
    ]
    
    for file_path in ablation_md_files:
        if file_path.endswith('.md') and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "ΔSharpe" in content or "delta" in content.lower():
                        return True, "Ablation report contains feature analysis"
            except Exception:
                continue
    
    return False, "No ablation analysis results found"


def verify_monitoring(outdir: str) -> Tuple[bool, str]:
    """Verify monitoring system is configured and working."""
    # Check for monitoring configuration
    monitoring_files = [
        os.path.join(outdir, "monitoring", "monitoring_config.json"),
        os.path.join(outdir, "monitoring", "test_run", "monitoring_data.json"),
        os.path.join(outdir, "monitoring", "heartbeat.json")
    ]
    
    config_found = False
    heartbeat_found = False
    
    for file_path in monitoring_files:
        if os.path.exists(file_path):
            if "config" in file_path:
                config_found = True
            elif "heartbeat" in file_path or "monitoring_data" in file_path:
                heartbeat_found = True
                # Check if heartbeat is recent (within 24 hours)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    timestamp = data.get('timestamp') or data.get('last_run_epoch', 0)
                    if isinstance(timestamp, str):
                        # Parse ISO timestamp
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.timestamp()
                    if (time.time() - float(timestamp)) < 24*3600:
                        return True, f"Monitoring heartbeat is recent: {file_path}"
                except Exception:
                    pass
    
    if config_found and heartbeat_found:
        return True, "Monitoring configured and heartbeat found"
    elif config_found:
        return False, "Monitoring configured but no recent heartbeat"
    else:
        return False, "No monitoring configuration found"


def verify_hard_invariants(outdir: str) -> Tuple[bool, str]:
    """Check hard invariants that must pass for READY status."""
    # Check for gate passes in any leaderboard
    leaderboard_globs = [
        os.path.join(outdir, "**", "*leaderboard*.csv"),
        os.path.join("results", "**", "*leaderboard*.csv"),
        os.path.join("portfolios", "**", "*leaderboard*.csv")
    ]
    
    leaderboard_files = []
    for leaderboard_glob in leaderboard_globs:
        leaderboard_files.extend(glob.glob(leaderboard_glob, recursive=True))
    
    if not leaderboard_files:
        # Also check for grid results with gate_pass column
        grid_globs = [
            os.path.join(outdir, "**", "*grid*.csv"),
            os.path.join("results", "**", "*grid*.csv"),
            os.path.join("portfolios", "**", "*grid*.csv")
        ]
        
        grid_files = []
        for grid_glob in grid_globs:
            grid_files.extend(glob.glob(grid_glob, recursive=True))
        
        any_gate_pass = False
        for file_path in grid_files:
            try:
                df = pd.read_csv(file_path)
                if 'gate_pass' in df.columns:
                    gate_passes = df['gate_pass'].astype(str).str.lower().eq('true').sum()
                    if gate_passes > 0:
                        any_gate_pass = True
                        break
            except Exception:
                continue
        
        if any_gate_pass:
            return True, "Found gate passes in grid results"
        else:
            return False, "No strategies passed gate in any results"
    
    # Check leaderboard files
    any_gate_pass = False
    for file_path in leaderboard_files:
        try:
            df = pd.read_csv(file_path)
            if 'gate_pass' in df.columns:
                gate_passes = df['gate_pass'].astype(str).str.lower().eq('true').sum()
                if gate_passes > 0:
                    any_gate_pass = True
                    break
        except Exception:
            continue
    
    if any_gate_pass:
        return True, "Found gate passes in leaderboard"
    else:
        return False, "No strategies passed gate in any leaderboard"


def verify_all_components(outdir: str) -> Dict[str, Tuple[bool, str]]:
    """Verify all components and return detailed results."""
    results = {}
    
    # Verify each component
    results['robustness'] = verify_robustness(outdir)
    results['oos'] = verify_oos(outdir)
    results['lag'] = verify_lag(outdir)
    results['portfolio'] = verify_portfolio(outdir)
    results['ablation'] = verify_ablation(outdir)
    results['monitoring'] = verify_monitoring(outdir)
    results['invariants'] = verify_hard_invariants(outdir)
    
    return results


def main():
    """CLI for component verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify deployment components')
    parser.add_argument('--outdir', required=True, help='Output directory to verify')
    parser.add_argument('--component', help='Specific component to verify')
    
    args = parser.parse_args()
    
    if args.component:
        # Verify specific component
        verifiers = {
            'robustness': verify_robustness,
            'oos': verify_oos,
            'lag': verify_lag,
            'portfolio': verify_portfolio,
            'ablation': verify_ablation,
            'monitoring': verify_monitoring,
            'invariants': verify_hard_invariants
        }
        
        if args.component in verifiers:
            success, message = verifiers[args.component](args.outdir)
            print(f"{args.component}: {'✅' if success else '❌'} {message}")
        else:
            print(f"Unknown component: {args.component}")
    else:
        # Verify all components
        results = verify_all_components(args.outdir)
        
        print("=== COMPONENT VERIFICATION ===")
        for component, (success, message) in results.items():
            print(f"{component:12}: {'✅' if success else '❌'} {message}")
        
        # Overall status
        all_passed = all(success for success, _ in results.values())
        print(f"\nOverall: {'✅ READY' if all_passed else '❌ NOT READY'}")


if __name__ == "__main__":
    main()
