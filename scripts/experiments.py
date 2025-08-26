#!/usr/bin/env python3
"""
Aurora Experiment Tracking & Management
=======================================

Simple tools for tracking and comparing training experiments.

Examples:
    # List recent experiments
    python scripts/experiments.py list --limit 10
    
    # Compare specific experiments  
    python scripts/experiments.py compare exp_001 exp_002 exp_003
    
    # Show detailed results
    python scripts/experiments.py show exp_001
    
    # Clean up old experiments
    python scripts/experiments.py cleanup --older-than 30
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import setup_logging

logger = setup_logging(__name__)


class ExperimentTracker:
    """Track and manage training experiments."""
    
    def __init__(self):
        self.experiments_dir = Path("reports/experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self, limit: int = 20, status: Optional[str] = None) -> List[Dict]:
        """List recent experiments."""
        experiments = []
        
        # Scan for experiment files
        for exp_file in self.experiments_dir.glob("*.json"):
            if exp_file.name.startswith("comparison_"):
                continue  # Skip comparison files
                
            try:
                with open(exp_file) as f:
                    exp_data = json.load(f)
                
                # Add file info
                exp_data["file_path"] = str(exp_file)
                exp_data["file_name"] = exp_file.name
                exp_data["created_time"] = datetime.fromtimestamp(exp_file.stat().st_mtime)
                
                # Filter by status if specified
                if status and exp_data.get("status") != status:
                    continue
                    
                experiments.append(exp_data)
                
            except Exception as e:
                logger.warning(f"Could not load experiment {exp_file}: {e}")
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.get("created_time", datetime.min), reverse=True)
        
        return experiments[:limit]
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get specific experiment by ID."""
        # Try exact match first
        exp_file = self.experiments_dir / f"{exp_id}.json"
        if exp_file.exists():
            with open(exp_file) as f:
                return json.load(f)
        
        # Try prefix match
        for exp_file in self.experiments_dir.glob(f"{exp_id}*.json"):
            with open(exp_file) as f:
                return json.load(f)
        
        # Try searching inside files
        for exp_file in self.experiments_dir.glob("*.json"):
            try:
                with open(exp_file) as f:
                    exp_data = json.load(f)
                    if exp_data.get("exp_id") == exp_id:
                        return exp_data
            except:
                continue
                
        return None
    
    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        experiments = []
        
        for exp_id in exp_ids:
            exp_data = self.get_experiment(exp_id)
            if exp_data:
                experiments.append(exp_data)
            else:
                logger.warning(f"Experiment {exp_id} not found")
        
        if not experiments:
            raise ValueError("No valid experiments found")
        
        # Extract comparison data
        comparison_data = []
        for exp in experiments:
            row = {
                "exp_id": exp.get("exp_id", "unknown"),
                "profile": exp.get("profile", "unknown"),
                "model_kind": exp.get("model", {}).get("kind", "unknown"),
                "status": exp.get("status", "unknown")
            }
            
            # Extract metrics
            metrics = exp.get("metrics", {})
            row["ic"] = metrics.get("ic", 0.0)
            row["r2"] = metrics.get("r2", 0.0)
            row["duration_sec"] = metrics.get("duration_sec", 0.0)
            row["rows"] = metrics.get("rows", 0)
            row["cols"] = metrics.get("cols", 0)
            
            # Model parameters
            model_config = exp.get("model", {})
            if model_config.get("kind") == "ridge":
                row["alpha"] = model_config.get("alpha", "N/A")
                row["n_estimators"] = "N/A"
                row["max_depth"] = "N/A"
            elif model_config.get("kind") == "xgboost":
                row["alpha"] = "N/A"
                row["n_estimators"] = model_config.get("n_estimators", "N/A")
                row["max_depth"] = model_config.get("max_depth", "N/A")
            
            # ONNX export info
            onnx_info = exp.get("onnx", {})
            row["onnx_exported"] = onnx_info.get("exported", False)
            row["parity_ok"] = onnx_info.get("parity", {}).get("parity_ok", False)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def show_experiment_details(self, exp_id: str) -> None:
        """Show detailed information about an experiment."""
        exp_data = self.get_experiment(exp_id)
        if not exp_data:
            print(f"Experiment {exp_id} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT DETAILS: {exp_id}")
        print(f"{'='*60}")
        
        # Basic info
        print(f"Profile: {exp_data.get('profile', 'unknown')}")
        print(f"Status: {exp_data.get('status', 'unknown')}")
        print(f"Snapshot: {exp_data.get('snapshot', 'unknown')}")
        
        # Model config
        model = exp_data.get("model", {})
        print(f"\nModel Configuration:")
        print(f"  Kind: {model.get('kind', 'unknown')}")
        for key, value in model.items():
            if key != "kind":
                print(f"  {key}: {value}")
        
        # Metrics
        metrics = exp_data.get("metrics", {})
        print(f"\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # ONNX info
        onnx_info = exp_data.get("onnx", {})
        if onnx_info:
            print(f"\nONNX Export:")
            print(f"  Exported: {onnx_info.get('exported', False)}")
            print(f"  Path: {onnx_info.get('path', 'N/A')}")
            parity = onnx_info.get("parity", {})
            if parity:
                print(f"  Parity OK: {parity.get('parity_ok', False)}")
                print(f"  Max Diff: {parity.get('max_diff', 'N/A')}")
        
        # Features
        features = exp_data.get("features", [])
        if features:
            print(f"\nFeatures ({len(features)}):")
            for feature in features[:10]:  # Show first 10
                print(f"  - {feature}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")
        
        # Grid search info (if available)
        grid_search = exp_data.get("grid_search", {})
        if grid_search and grid_search.get("best_vs_baseline"):
            baseline = grid_search["best_vs_baseline"]
            print(f"\nGrid Search Results:")
            print(f"  Baseline IC: {baseline.get('baseline_ic_mean', 'N/A'):.6f}")
            print(f"  Best IC: {baseline.get('best_ic_mean', 'N/A'):.6f}")
            print(f"  Improvement: {baseline.get('delta_mean', 'N/A'):.6f}")
            print(f"  P-value: {baseline.get('p_value', 'N/A'):.6f}")
    
    def cleanup_experiments(self, older_than_days: int = 30, dry_run: bool = True) -> None:
        """Clean up old experiment files."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        files_to_delete = []
        total_size = 0
        
        for exp_file in self.experiments_dir.glob("*.json"):
            file_time = datetime.fromtimestamp(exp_file.stat().st_mtime)
            if file_time < cutoff_date:
                files_to_delete.append(exp_file)
                total_size += exp_file.stat().st_size
        
        print(f"Found {len(files_to_delete)} files older than {older_than_days} days")
        print(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        if dry_run:
            print("\nFiles to delete (DRY RUN):")
            for file_path in files_to_delete:
                print(f"  {file_path.name}")
            print("\nUse --no-dry-run to actually delete files")
        else:
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path.name}")
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")
            print(f"Cleanup complete: {len(files_to_delete)} files deleted")
    
    def get_best_experiments(self, metric: str = "ic", limit: int = 5) -> List[Dict]:
        """Get top experiments by metric."""
        experiments = self.list_experiments(limit=100)  # Get more for better ranking
        
        # Filter and sort by metric
        valid_experiments = []
        for exp in experiments:
            metrics = exp.get("metrics", {})
            if metric in metrics and exp.get("status") == "completed":
                exp["sort_metric"] = metrics[metric]
                valid_experiments.append(exp)
        
        # Sort by metric (descending)
        valid_experiments.sort(key=lambda x: x["sort_metric"], reverse=True)
        
        return valid_experiments[:limit]


def main():
    parser = argparse.ArgumentParser(description="Aurora Experiment Tracking")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of experiments to show")
    list_parser.add_argument("--status", help="Filter by status")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("exp_ids", nargs="+", help="Experiment IDs to compare")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID to show")
    
    # Best command
    best_parser = subparsers.add_parser("best", help="Show best experiments")
    best_parser.add_argument("--metric", default="ic", help="Metric to rank by")
    best_parser.add_argument("--limit", type=int, default=5, help="Number to show")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old experiments")
    cleanup_parser.add_argument("--older-than", type=int, default=30, help="Days old")
    cleanup_parser.add_argument("--no-dry-run", action="store_true", help="Actually delete files")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tracker = ExperimentTracker()
    
    try:
        if args.command == "list":
            experiments = tracker.list_experiments(args.limit, args.status)
            
            if not experiments:
                print("No experiments found")
                return
            
            print(f"\n{'ID':<15} {'Profile':<20} {'Model':<8} {'IC':<8} {'Quality':<10} {'Status':<10} {'Date':<12}")
            print("-" * 95)
            
            for exp in experiments:
                exp_id = exp.get("exp_id", exp.get("file_name", "unknown"))[:15]
                profile = exp.get("profile", "unknown")[:20]
                model = exp.get("model", {}).get("kind", "unknown")[:8]
                ic = exp.get("metrics", {}).get("ic", 0.0)
                ic_quality = exp.get("metrics", {}).get("ic_quality", "unknown")[:10]
                status = exp.get("status", "unknown")[:10]
                date = exp.get("created_time", datetime.min).strftime("%Y-%m-%d")
                
                print(f"{exp_id:<15} {profile:<20} {model:<8} {ic:<8.4f} {ic_quality:<10} {status:<10} {date:<12}")
        
        elif args.command == "compare":
            df = tracker.compare_experiments(args.exp_ids)
            print("\nExperiment Comparison:")
            print("=" * 100)
            print(df.to_string(index=False))
        
        elif args.command == "show":
            tracker.show_experiment_details(args.exp_id)
        
        elif args.command == "best":
            best_experiments = tracker.get_best_experiments(args.metric, args.limit)
            
            if not best_experiments:
                print(f"No experiments found with metric '{args.metric}'")
                return
            
            print(f"\nTop {args.limit} Experiments by {args.metric.upper()}:")
            print("=" * 70)
            print(f"{'Rank':<4} {'ID':<15} {'Model':<8} {args.metric.upper():<8} {'Profile':<20}")
            print("-" * 70)
            
            for i, exp in enumerate(best_experiments):
                exp_id = exp.get("exp_id", "unknown")[:15]
                model = exp.get("model", {}).get("kind", "unknown")[:8]
                metric_val = exp["sort_metric"]
                profile = exp.get("profile", "unknown")[:20]
                
                print(f"{i+1:<4} {exp_id:<15} {model:<8} {metric_val:<8.4f} {profile:<20}")
        
        elif args.command == "cleanup":
            tracker.cleanup_experiments(args.older_than, dry_run=not args.no_dry_run)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
