#!/usr/bin/env python3
"""
Create run metadata for reproducibility tracking
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
import hashlib


def create_run_metadata(results_dir: str, universe_cfg_path: str, grid_cfg_path: str):
    """Create run metadata for reproducibility"""
    
    # Load configs
    with open(universe_cfg_path, 'r') as f:
        universe_cfg = yaml.safe_load(f)
    
    with open(grid_cfg_path, 'r') as f:
        grid_cfg = yaml.safe_load(f)
    
    # Create metadata
    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "data_contract": {
            "yfinance_auto_adjust": True,  # Explicitly set in runner_grid.py
            "frequency": "daily",
            "timezone": "UTC",
            "missing_data_policy": "forward_fill_then_dropna"
        },
        "configs": {
            "universe": {
                "path": universe_cfg_path,
                "symbols_count": len(universe_cfg.get("universe", [])),
                "market_proxy": universe_cfg.get("market_proxy", "SPY")
            },
            "grid": {
                "path": grid_cfg_path,
                "model_types": [m.get("type", "unknown") for m in grid_cfg.get("models", [])],
                "costs_bps": grid_cfg.get("costs", {}).get("commission_bps", 0) + 
                           grid_cfg.get("costs", {}).get("slippage_bps", 0)
            }
        },
        "gate_policy": {
            "sharpe_threshold": "model_sharpe >= bh_sharpe + 0.10",
            "activity_requirements": {
                "min_trades": ">= 15",
                "turnover_band": "0.05 <= turnover <= 0.30"
            },
            "validation_checks": [
                "best_vs_BH != best_median_sharpe",
                "median_trades and median_turnover populated",
                "realistic_sharpe_ratios < 3.0"
            ]
        }
    }
    
    # Save metadata
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = results_path / "run_meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Run metadata saved to: {metadata_path}")
    return metadata_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create run metadata")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--universe-cfg", required=True, help="Universe config path")
    parser.add_argument("--grid-cfg", required=True, help="Grid config path")
    
    args = parser.parse_args()
    create_run_metadata(args.results_dir, args.universe_cfg, args.grid_cfg)
