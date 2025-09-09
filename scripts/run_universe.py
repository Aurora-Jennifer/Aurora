#!/usr/bin/env python3
"""
Multi-Asset Universe CLI Runner

CLI entrypoint to run grid experiments across a universe of assets.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.runner_universe import run_universe


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-asset universe grid experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classical models across universe
  python scripts/run_universe.py --universe-cfg config/universe.yaml --grid-cfg config/grid.yaml --out-dir results/classical
  
  # Deep learning models across universe  
  python scripts/run_universe.py --universe-cfg config/universe.yaml --grid-cfg config/dl.yaml --out-dir results/dl
  
  # Quick test with subset
  python scripts/run_universe.py --universe-cfg config/universe.yaml --grid-cfg config/grid.yaml --out-dir results/test
        """
    )
    
    parser.add_argument(
        "--universe-cfg", 
        required=True, 
        help="Path to universe configuration YAML (e.g., config/universe.yaml)"
    )
    
    parser.add_argument(
        "--grid-cfg", 
        required=True, 
        help="Path to grid configuration YAML (e.g., config/grid.yaml or config/dl.yaml)"
    )
    
    parser.add_argument(
        "--out-dir", 
        default="universe_results", 
        help="Output directory for results (default: universe_results)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    # OOS validation arguments
    parser.add_argument("--oos-days", type=int, default=252,
                       help="Number of most-recent trading days reserved as out-of-sample.")
    parser.add_argument("--oos-min-train", type=int, default=252,
                       help="Minimum training days required before OOS split; else run aborts.")
    parser.add_argument("--embargo-days", type=int, default=10,
                       help="Bars to embargo around the OOS split to prevent leakage.")
    parser.add_argument("--fast-eval", action="store_true",
                       help="Skip per-ticker backtests; compute IC/Rank-IC + a tiny sanity set.")
    parser.add_argument("--n-jobs", type=int, default=4,
                       help="Number of parallel jobs for evaluation")
    parser.add_argument("--batch-size", type=int, default=25,
                       help="Batch size for parallel processing")
    
    # Date slice arguments for OOS validation
    parser.add_argument("--train-start", type=str, default=None,
                       help="Training period start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, default=None,
                       help="Training period end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, default=None,
                       help="Test period start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default=None,
                       help="Test period end date (YYYY-MM-DD)")
    parser.add_argument("--feature-lookback-days", type=int, default=252,
                       help="Extra days before train-start to compute rolling features")
    
    # Prediction dumping for signal lag tests
    parser.add_argument("--dump-preds", action="store_true",
                       help="Save OOS per-date predictions/edges")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.universe_cfg).exists():
        print(f"❌ Universe config file not found: {args.universe_cfg}")
        sys.exit(1)
    
    if not Path(args.grid_cfg).exists():
        print(f"❌ Grid config file not found: {args.grid_cfg}")
        sys.exit(1)
    
    print(f"🚀 Starting universe run...")
    print(f"  Universe config: {args.universe_cfg}")
    print(f"  Grid config: {args.grid_cfg}")
    print(f"  Output directory: {args.out_dir}")
    
    try:
        # Create date slice dict if any date args provided
        date_slice = {}
        if any([args.train_start, args.train_end, args.test_start, args.test_end]):
            date_slice = {
                'train_start': args.train_start,
                'train_end': args.train_end,
                'test_start': args.test_start,
                'test_end': args.test_end,
                'feature_lookback_days': args.feature_lookback_days
            }
        
        board = run_universe(args.universe_cfg, args.grid_cfg, args.out_dir, 
                           date_slice=date_slice, dump_preds=args.dump_preds,
                           oos_days=args.oos_days, oos_min_train=args.oos_min_train,
                           embargo_days=args.embargo_days, fast_eval=args.fast_eval,
                           n_jobs=args.n_jobs, batch_size=args.batch_size)
        
        print(f"\n🎉 Universe run completed successfully!")
        print(f"📊 Results saved to: {args.out_dir}/")
        print(f"📈 Leaderboard: {args.out_dir}/leaderboard.csv")
        print(f"📝 Log file: {args.out_dir}/universe_run.log")
        
        # Show top performers
        successful = board[board['gate_pass'] == True]
        if len(successful) > 0:
            print(f"\n🏆 Top gate-passing assets:")
            for _, row in successful.head(5).iterrows():
                print(f"  {row['ticker']}: Sharpe={row['best_median_sharpe']:.3f}, "
                      f"vs BH={row.get('excess_vs_bh', 0):.3f}, trades={row['median_trades']:.0f}")
        else:
            print(f"\n⚠️  No assets passed the gate criteria")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Universe run interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Universe run failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
