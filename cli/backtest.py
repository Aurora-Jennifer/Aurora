"""
Backtest CLI
Command-line interface for the backtest system
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import json

from core.config_loader import load_config
from core.engine.backtest import BacktestEngine


def main():
    """Main function for running backtests."""
    parser = argparse.ArgumentParser(description="Comprehensive Backtest System")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols (overrides config)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.json",
        help="Base configuration file path (default: config/base.json)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["risk_low", "risk_balanced", "risk_strict"],
        help="Risk profile to apply",
    )
    parser.add_argument("--asset", type=str, help="Asset symbol for asset-specific configuration")
    parser.add_argument(
        "--fast", action="store_true", help="Run in fast mode (reduced warmup period)"
    )
    parser.add_argument("--initial-capital", type=float, help="Initial capital (overrides config)")

    args = parser.parse_args()

    try:
        # Build CLI overrides
        cli_overrides = {}
        if args.symbols:
            cli_overrides["symbols"] = args.symbols
        if args.fast:
            cli_overrides["fast_mode"] = True
        if args.initial_capital:
            cli_overrides["initial_capital"] = args.initial_capital

        # Load configuration with profile and asset overrides
        config = load_config(
            profile=args.profile,
            asset=args.asset,
            cli_overrides=cli_overrides,
            base_config_path=args.config,
        )

        # Save config to temporary file and initialize backtest engine
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2)
            temp_config_path = f.name

        try:
            engine = BacktestEngine(temp_config_path)
        finally:
            # Clean up temp file
            os.unlink(temp_config_path)

        # Run backtest
        results = engine.run_backtest(
            start_date=args.start, end_date=args.end, symbols=config["symbols"]
        )

        if results:
            print("✅ Backtest completed successfully!")
            print(
                f"📊 Configuration: {args.profile or 'base'} profile, {len(config['symbols'])} symbols"
            )
        else:
            print("❌ Backtest failed or returned no results")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
