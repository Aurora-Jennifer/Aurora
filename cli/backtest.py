"""
Backtest CLI
Command-line interface for the backtest system
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.backtest import BacktestEngine


def main():
    """Main function for running backtests."""
    parser = argparse.ArgumentParser(description="Comprehensive Backtest System")
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"], help="Trading symbols (default: SPY)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/backtest_config.json",
        help="Configuration file path",
    )
    parser.add_argument("--profile", type=str, help="Profile configuration file path")
    parser.add_argument(
        "--fast", action="store_true", help="Run in fast mode (reduced warmup period)"
    )

    args = parser.parse_args()

    try:
        # Initialize backtest engine
        engine = BacktestEngine(args.config, args.profile)

        # Set fast mode if requested
        if args.fast:
            engine.trading_system.config["fast_mode"] = True

        # Run backtest
        results = engine.run_backtest(
            start_date=args.start, end_date=args.end, symbols=args.symbols
        )

        if results:
            print("✅ Backtest completed successfully!")
        else:
            print("❌ Backtest failed or returned no results")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
