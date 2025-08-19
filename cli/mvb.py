"""
MVB CLI
Command-line interface for the Minimum Viable Bot
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.mvb_runner import run_mvb
from core.utils import setup_logging


def load_config(config_file: str) -> dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_file) as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def create_default_config() -> dict[str, Any]:
    """Create default configuration"""
    return {
        "initial_capital": 100000,
        "symbols": ["SPY"],
        "strategy": "random",
        "strategy_config": {"signal_strength": 0.3},
        "session_duration": 30,  # minutes
        "bar_interval": 60,  # seconds
        "max_position_pct": 0.1,
        "daily_loss_limit": 0.01,  # 1%
        "heartbeat_interval": 30,  # seconds
        "data_dir": "data",
        "log_file": "results/mvb_session.ndjson",
    }


def main():
    """Main MVB CLI function"""
    parser = argparse.ArgumentParser(description="Minimum Viable Bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "shadow", "paper"],
        default="shadow",
        help="Trading mode",
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], help="Trading symbols")
    parser.add_argument(
        "--strategy",
        choices=["random", "simple_ma", "regime_aware"],
        default="random",
        help="Trading strategy",
    )
    parser.add_argument(
        "--session-duration", type=int, default=30, help="Session duration in minutes"
    )
    parser.add_argument("--initial-capital", type=float, default=100000, help="Initial capital")
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.1,
        help="Maximum position size as percentage of portfolio",
    )
    parser.add_argument(
        "--daily-loss-limit",
        type=float,
        default=0.01,
        help="Daily loss limit as percentage",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no real orders)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()

    # Override config with command line arguments
    config.update(
        {
            "symbols": args.symbols,
            "strategy": args.strategy,
            "session_duration": args.session_duration,
            "initial_capital": args.initial_capital,
            "max_position_pct": args.max_position_pct,
            "daily_loss_limit": args.daily_loss_limit,
            "dry_run": args.dry_run,
        }
    )

    # Add mode-specific settings
    if args.mode == "backtest":
        config.update({"start_date": "2024-01-01", "end_date": "2024-03-31", "data_dir": "data"})
    elif args.mode == "shadow":
        config.update(
            {"log_file": f"results/shadow_{Path(__file__).stem}_{args.session_duration}m.ndjson"}
        )
    elif args.mode == "paper":
        config.update(
            {"log_file": f"results/paper_{Path(__file__).stem}_{args.session_duration}m.ndjson"}
        )

    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info(f"Starting MVB in {args.mode} mode")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    try:
        # Run MVB
        run_mvb(args.mode, config)
        logger.info("MVB completed successfully")

    except KeyboardInterrupt:
        logger.info("MVB interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"MVB failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
