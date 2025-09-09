#!/usr/bin/env python3
"""
Repair leaderboard from cross-sectional run
- Fix runtime_sec=0.0 everywhere
- Clarify ambiguous best_vs_BH column
- Add proper excess vs BH calculations
"""

import pandas as pd
import numpy as np
import sys
import json
import time
from pathlib import Path

def repair_leaderboard(results_dir: Path):
    """Repair leaderboard issues from cross-sectional run"""
    
    lb_path = results_dir / "leaderboard.csv"
    if not lb_path.exists():
        print(f"❌ Leaderboard not found: {lb_path}")
        return False
    
    print(f"🔧 Repairing leaderboard: {lb_path}")
    lb = pd.read_csv(lb_path)
    
    # Fix 1: Handle ambiguous best_vs_BH column
    if "bh_median_sharpe" not in lb.columns:
        # Check if best_vs_BH actually contains BH Sharpe values
        vs_bh_values = lb["best_vs_BH"].values
        model_sharpe_values = lb["best_median_sharpe"].values
        
        # If best_vs_BH values are in [-3,3] range and model Sharpe is large, 
        # it's likely BH Sharpe, not excess
        if (abs(vs_bh_values).max() < 3.0 and model_sharpe_values.max() > 2.0):
            print("📊 Detected best_vs_BH contains BH Sharpe values, not excess")
            lb = lb.rename(columns={"best_vs_BH": "bh_median_sharpe"})
            lb["excess_vs_bh"] = lb["best_median_sharpe"] - lb["bh_median_sharpe"]
        else:
            # best_vs_BH is already excess, just rename for clarity
            lb = lb.rename(columns={"best_vs_BH": "excess_vs_bh"})
            # We can't recover BH Sharpe from this, so set to NaN
            lb["bh_median_sharpe"] = np.nan
    
    # Fix 2: Handle best_vs_rule column
    if "best_vs_rule" in lb.columns:
        lb = lb.rename(columns={"best_vs_rule": "excess_vs_rule"})
        lb["rule_median_sharpe"] = lb["best_median_sharpe"] - lb["excess_vs_rule"]
    
    # Fix 3: Fill runtime_sec if zero with panel runtime from metadata
    if (lb["runtime_sec"] == 0.0).all():
        print("⏱️  All runtime_sec=0.0, attempting to recover from metadata")
        
        # Try to get panel runtime from log file
        log_path = results_dir / "universe_run.log"
        panel_runtime = None
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_content = f.read()
                # Look for "Panel dataset shape" timestamp
                import re
                pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Panel dataset shape'
                matches = re.findall(pattern, log_content)
                if matches:
                    # Estimate runtime from log timestamps
                    start_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Starting cross-sectional universe run'
                    start_matches = re.findall(start_pattern, log_content)
                    if start_matches:
                        from datetime import datetime
                        start_time = datetime.strptime(start_matches[0], '%Y-%m-%d %H:%M:%S')
                        end_time = datetime.strptime(matches[0], '%Y-%m-%d %H:%M:%S')
                        panel_runtime = (end_time - start_time).total_seconds()
                        print(f"📊 Estimated panel runtime: {panel_runtime:.1f} seconds")
        
        if panel_runtime is not None:
            lb["runtime_sec"] = panel_runtime
        else:
            # Fallback: set to NaN to indicate missing data
            lb["runtime_sec"] = np.nan
            print("⚠️  Could not recover runtime, set to NaN")
    
    # Save repaired leaderboard
    lb.to_csv(lb_path, index=False)
    print(f"✅ Leaderboard repaired: {lb_path}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Total symbols: {len(lb)}")
    print(f"  Avg Sharpe: {lb['best_median_sharpe'].mean():.3f}")
    print(f"  Max Sharpe: {lb['best_median_sharpe'].max():.3f}")
    print(f"  Symbols with Sharpe > 4.0: {(lb['best_median_sharpe'] > 4.0).sum()}")
    print(f"  Symbols with Sharpe > 5.0: {(lb['best_median_sharpe'] > 5.0).sum()}")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/repair_leaderboard.py <results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    success = repair_leaderboard(results_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
