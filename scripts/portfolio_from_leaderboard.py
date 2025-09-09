#!/usr/bin/env python3
"""
Portfolio Selection from Leaderboard

Simple helper to pick top-K passing tickers for a portfolio run.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


def select_top_k(leaderboard_csv: str, k: int = 5, gate_only: bool = True, 
                min_sharpe: float = None, min_vs_baseline: float = None) -> list:
    """
    Select top-K tickers from leaderboard
    
    Args:
        leaderboard_csv: Path to leaderboard CSV
        k: Number of top tickers to select
        gate_only: Only select tickers that passed the gate
        min_sharpe: Minimum Sharpe ratio threshold
        min_vs_baseline: Minimum improvement over baseline
    
    Returns:
        List of selected ticker symbols
    """
    if not Path(leaderboard_csv).exists():
        raise FileNotFoundError(f"Leaderboard file not found: {leaderboard_csv}")
    
    df = pd.read_csv(leaderboard_csv)
    
    # Filter by gate pass if requested
    if gate_only:
        df = df[df["gate_pass"] == True]
        print(f"Filtered to {len(df)} gate-passing assets")
    
    # Filter by minimum Sharpe if specified
    if min_sharpe is not None:
        df = df[df["best_median_sharpe"] >= min_sharpe]
        print(f"Filtered to {len(df)} assets with Sharpe >= {min_sharpe}")
    
    # Filter by minimum vs baseline if specified
    if min_vs_baseline is not None:
        df = df[df["best_vs_BH"] >= min_vs_baseline]
        print(f"Filtered to {len(df)} assets with vs BH >= {min_vs_baseline}")
    
    if len(df) == 0:
        print("⚠️  No assets meet the selection criteria")
        return []
    
    # Sort by Sharpe ratio and select top K
    df_sorted = df.sort_values("best_median_sharpe", ascending=False)
    selected = df_sorted.head(k)
    
    tickers = list(selected["ticker"])
    
    print(f"\n🎯 Selected top {len(tickers)} assets:")
    for _, row in selected.iterrows():
        print(f"  {row['ticker']}: Sharpe={row['best_median_sharpe']:.3f}, "
              f"vs BH={row['best_vs_BH']:.3f}, vs Rule={row['best_vs_rule']:.3f}")
    
    return tickers


def create_portfolio_config(tickers: list, output_path: str = "portfolio_config.yaml"):
    """
    Create a portfolio configuration file for the selected tickers
    
    Args:
        tickers: List of selected ticker symbols
        output_path: Path to save the portfolio config
    """
    if not tickers:
        print("No tickers to create portfolio config for")
        return
    
    config = {
        "portfolio": {
            "tickers": tickers,
            "weights": "equal",  # Equal weight for now
            "rebalance_frequency": "monthly"
        },
        "data": {
            "start_date": "2023-01-01",
            "end_date": "2024-12-31"
        },
        "costs": {
            "commission_bps": 1,
            "slippage_bps": 2
        }
    }
    
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Portfolio config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select top-K assets from leaderboard for portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select top 5 gate-passing assets
  python scripts/portfolio_from_leaderboard.py --leaderboard results/leaderboard.csv --k 5
  
  # Select top 10 assets (including failed gate)
  python scripts/portfolio_from_leaderboard.py --leaderboard results/leaderboard.csv --k 10 --include-failed
  
  # Select assets with minimum Sharpe of 0.5
  python scripts/portfolio_from_leaderboard.py --leaderboard results/leaderboard.csv --k 5 --min-sharpe 0.5
  
  # Create portfolio config file
  python scripts/portfolio_from_leaderboard.py --leaderboard results/leaderboard.csv --k 5 --create-config
        """
    )
    
    parser.add_argument(
        "--leaderboard", 
        required=True, 
        help="Path to leaderboard CSV file"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of top assets to select (default: 5)"
    )
    
    parser.add_argument(
        "--include-failed", 
        action="store_true", 
        help="Include assets that failed the gate"
    )
    
    parser.add_argument(
        "--min-sharpe", 
        type=float, 
        help="Minimum Sharpe ratio threshold"
    )
    
    parser.add_argument(
        "--min-vs-baseline", 
        type=float, 
        help="Minimum improvement over baseline (vs BH)"
    )
    
    parser.add_argument(
        "--create-config", 
        action="store_true", 
        help="Create a portfolio configuration file"
    )
    
    parser.add_argument(
        "--output", 
        default="portfolio_config.yaml", 
        help="Output path for portfolio config (default: portfolio_config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        # Select top-K assets
        tickers = select_top_k(
            leaderboard_csv=args.leaderboard,
            k=args.k,
            gate_only=not args.include_failed,
            min_sharpe=args.min_sharpe,
            min_vs_baseline=args.min_vs_baseline
        )
        
        if not tickers:
            print("No assets selected")
            return 1
        
        # Print selected tickers as comma-separated list
        print(f"\n📋 Selected tickers: {','.join(tickers)}")
        
        # Create portfolio config if requested
        if args.create_config:
            create_portfolio_config(tickers, args.output)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
