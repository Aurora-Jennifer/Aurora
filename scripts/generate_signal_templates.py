#!/usr/bin/env python3
"""
Walk-Forward Signal Template Generator

This script performs walk-forward analysis to generate signal templates
for use in paper trading and live trading systems.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from core.utils import setup_logging
from scripts.walkforward_framework import (
    LeakageProofPipeline,
    build_feature_table,
    gen_walkforward,
    walkforward_run,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class SignalTemplateGenerator:
    """Generate signal templates using walk-forward analysis."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal template generator.

        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config
        self.symbols = config.get("symbols", ["SPY"])
        self.train_days = config.get("train_days", 252)
        self.test_days = config.get("test_days", 63)
        self.stride_days = config.get("stride_days", 21)
        self.warmup_days = config.get("warmup_days", 100)
        self.min_history_days = config.get("min_history_days", 500)

        # Results storage
        self.templates = {}
        self.performance_metrics = {}
        self.fold_results = {}

    def fetch_historical_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                logger.error(f"No data found for {symbol}")
                return pd.DataFrame()

            # Ensure we have all required columns
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return pd.DataFrame()

            logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def generate_features(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate features from price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Tuple of (features, targets, prices)
        """
        logger.info("Generating features from price data")

        try:
            # Build feature table using the framework
            features, targets, prices = build_feature_table(
                data, warmup_days=self.warmup_days
            )

            logger.info(
                f"Generated {features.shape[1]} features for {len(features)} data points"
            )
            return features, targets, prices

        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return np.array([]), np.array([]), np.array([])

    def run_walkforward_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Run walk-forward analysis for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting walk-forward analysis for {symbol}")

        # Calculate date range - get more historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.min_history_days)

        # For better results, go back even further
        start_date = start_date - timedelta(days=500)  # Get extra historical data

        # Fetch data
        data = self.fetch_historical_data(symbol, str(start_date), str(end_date))

        if data.empty:
            logger.error(f"No data available for {symbol}")
            return {}

        # Generate features
        features, targets, prices = self.generate_features(data)

        if len(features) == 0:
            logger.error(f"No features generated for {symbol}")
            return {}

        # Generate folds
        folds = list(
            gen_walkforward(
                n=len(features),
                train_len=self.train_days,
                test_len=self.test_days,
                stride=self.stride_days,
                warmup=self.warmup_days,
            )
        )

        if not folds:
            logger.error(f"No folds generated for {symbol}")
            return {}

        logger.info(f"Generated {len(folds)} folds for {symbol}")

        # Create pipeline
        pipeline = LeakageProofPipeline(features, targets)

        # Run walk-forward analysis
        try:
            results = walkforward_run(
                pipeline=pipeline, folds=folds, prices=prices, model_seed=42
            )

            logger.info(f"Walk-forward analysis completed for {symbol}")
            return self._process_results(symbol, results, folds, data)

        except Exception as e:
            logger.error(f"Error in walk-forward analysis for {symbol}: {e}")
            return {}

    def _process_results(
        self, symbol: str, results: List, folds: List, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Process walk-forward results into signal templates.

        Args:
            symbol: Trading symbol
            results: Walk-forward results
            folds: Fold information
            data: Original price data

        Returns:
            Processed results dictionary
        """
        logger.info(f"Processing results for {symbol}")

        if not results:
            return {}

        # Extract metrics from each fold
        fold_metrics = []
        signal_patterns = []

        for fold_id, metrics, trades in results:
            fold_metrics.append(
                {
                    "fold_id": fold_id,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "total_return": metrics.get("total_return", 0),
                    "n_trades": metrics.get("n_trades", 0),
                }
            )

            # Extract signal patterns from trades
            if trades:
                signal_patterns.extend(self._extract_signal_patterns(trades))

        # Calculate aggregate metrics
        if fold_metrics:
            avg_sharpe = np.mean([m["sharpe_ratio"] for m in fold_metrics])
            avg_drawdown = np.mean([m["max_drawdown"] for m in fold_metrics])
            avg_win_rate = np.mean([m["win_rate"] for m in fold_metrics])
            total_trades = sum([m["n_trades"] for m in fold_metrics])
        else:
            avg_sharpe = avg_drawdown = avg_win_rate = total_trades = 0

        # Generate signal template
        signal_template = self._generate_signal_template(signal_patterns, fold_metrics)

        return {
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "data_period": {
                "start_date": data.index[0].strftime("%Y-%m-%d"),
                "end_date": data.index[-1].strftime("%Y-%m-%d"),
                "total_days": len(data),
            },
            "walkforward_params": {
                "train_days": self.train_days,
                "test_days": self.test_days,
                "stride_days": self.stride_days,
                "warmup_days": self.warmup_days,
                "total_folds": len(folds),
            },
            "performance_metrics": {
                "avg_sharpe_ratio": float(avg_sharpe),
                "avg_max_drawdown": float(avg_drawdown),
                "avg_win_rate": float(avg_win_rate),
                "total_trades": total_trades,
                "fold_count": len(fold_metrics),
            },
            "signal_template": signal_template,
            "fold_details": fold_metrics,
        }

    def _extract_signal_patterns(self, trades: List) -> List[Dict]:
        """
        Extract signal patterns from trades.

        Args:
            trades: List of trade dictionaries

        Returns:
            List of signal patterns
        """
        patterns = []

        for trade in trades:
            if isinstance(trade, dict) and "side" in trade:
                patterns.append(
                    {
                        "side": trade["side"],
                        "size": trade.get("size", 0),
                        "price": trade.get("price", 0),
                        "timestamp": trade.get("timestamp", 0),
                    }
                )

        return patterns

    def _generate_signal_template(
        self, signal_patterns: List, fold_metrics: List
    ) -> Dict[str, Any]:
        """
        Generate signal template from patterns and metrics.

        Args:
            signal_patterns: List of signal patterns
            fold_metrics: List of fold metrics

        Returns:
            Signal template dictionary
        """
        if not signal_patterns:
            return {"type": "default", "confidence": 0.0}

        # Analyze signal patterns
        buy_signals = [p for p in signal_patterns if p.get("side") == 1]
        sell_signals = [p for p in signal_patterns if p.get("side") == -1]

        # Calculate signal statistics
        total_signals = len(signal_patterns)
        buy_ratio = len(buy_signals) / total_signals if total_signals > 0 else 0.5
        sell_ratio = len(sell_signals) / total_signals if total_signals > 0 else 0.5

        # Calculate average position sizes
        avg_buy_size = (
            np.mean([s.get("size", 0) for s in buy_signals]) if buy_signals else 0
        )
        avg_sell_size = (
            np.mean([s.get("size", 0) for s in sell_signals]) if sell_signals else 0
        )

        # Determine signal type based on performance
        avg_sharpe = (
            np.mean([m["sharpe_ratio"] for m in fold_metrics]) if fold_metrics else 0
        )
        avg_win_rate = (
            np.mean([m["win_rate"] for m in fold_metrics]) if fold_metrics else 0
        )

        # Generate template
        template = {
            "type": "regime_aware_ensemble",
            "confidence": min(1.0, max(0.0, avg_sharpe / 2.0)),  # Normalize confidence
            "signal_ratios": {
                "buy_ratio": float(buy_ratio),
                "sell_ratio": float(sell_ratio),
            },
            "position_sizing": {
                "avg_buy_size": float(avg_buy_size),
                "avg_sell_size": float(avg_sell_size),
                "max_position_size": float(max(avg_buy_size, avg_sell_size) * 1.5),
            },
            "performance_thresholds": {
                "min_sharpe": float(avg_sharpe * 0.8),
                "min_win_rate": float(avg_win_rate * 0.9),
                "max_drawdown": float(
                    np.mean([m["max_drawdown"] for m in fold_metrics]) * 1.2
                )
                if fold_metrics
                else 0.1,
            },
            "regime_weights": {
                "trend_following_weight": 0.6 if avg_sharpe > 0.5 else 0.4,
                "mean_reversion_weight": 0.4 if avg_sharpe > 0.5 else 0.6,
            },
        }

        return template

    def generate_all_templates(self) -> Dict[str, Any]:
        """
        Generate signal templates for all symbols.

        Returns:
            Dictionary with all templates and summary
        """
        logger.info("Starting signal template generation for all symbols")

        all_results = {}
        summary_stats = {
            "total_symbols": len(self.symbols),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "generation_date": datetime.now().isoformat(),
        }

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")

            try:
                result = self.run_walkforward_analysis(symbol)

                if result:
                    all_results[symbol] = result
                    summary_stats["successful_analyses"] += 1
                    logger.info(f"✅ Successfully generated template for {symbol}")
                else:
                    summary_stats["failed_analyses"] += 1
                    logger.warning(f"⚠️ Failed to generate template for {symbol}")

            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                summary_stats["failed_analyses"] += 1

        # Calculate overall statistics
        if all_results:
            all_sharpes = []
            all_win_rates = []

            for result in all_results.values():
                metrics = result.get("performance_metrics", {})
                all_sharpes.append(metrics.get("avg_sharpe_ratio", 0))
                all_win_rates.append(metrics.get("avg_win_rate", 0))

            summary_stats.update(
                {
                    "avg_sharpe_across_symbols": float(np.mean(all_sharpes)),
                    "avg_win_rate_across_symbols": float(np.mean(all_win_rates)),
                    "best_performing_symbol": max(
                        all_results.keys(),
                        key=lambda x: all_results[x]["performance_metrics"][
                            "avg_sharpe_ratio"
                        ],
                    ),
                }
            )

        return {"summary": summary_stats, "templates": all_results}

    def save_templates(
        self, results: Dict[str, Any], output_dir: str = "results/signal_templates"
    ):
        """
        Save signal templates to files.

        Args:
            results: Template generation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = output_path / f"signal_templates_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save individual templates for easy access
        templates_dir = output_path / "individual_templates"
        templates_dir.mkdir(exist_ok=True)

        for symbol, template_data in results.get("templates", {}).items():
            template_file = templates_dir / f"{symbol}_template_{timestamp}.json"
            with open(template_file, "w") as f:
                json.dump(template_data, f, indent=2, default=str)

        # Save summary
        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(results.get("summary", {}), f, indent=2, default=str)

        logger.info(f"Templates saved to {output_path}")
        logger.info(f"Full results: {results_file}")
        logger.info(f"Individual templates: {templates_dir}")
        logger.info(f"Summary: {summary_file}")


def main():
    """Main function to run signal template generation."""
    # Configuration
    config = {
        "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
        "train_days": 126,  # 6 months training (reduced from 252)
        "test_days": 21,  # 1 month testing (reduced from 63)
        "stride_days": 21,  # 1 month stride
        "warmup_days": 50,  # 50 days warmup (reduced from 100)
        "min_history_days": 1000,  # Minimum 1000 days of data (increased from 500)
    }

    print("🚀 Starting Signal Template Generation")
    print("=" * 50)

    # Initialize generator
    generator = SignalTemplateGenerator(config)

    # Generate templates
    results = generator.generate_all_templates()

    # Save results
    generator.save_templates(results)

    # Print summary
    summary = results.get("summary", {})
    print("\n📊 GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total Symbols: {summary.get('total_symbols', 0)}")
    print(f"Successful: {summary.get('successful_analyses', 0)}")
    print(f"Failed: {summary.get('failed_analyses', 0)}")
    print(
        f"Success Rate: {summary.get('successful_analyses', 0) / summary.get('total_symbols', 1) * 100:.1f}%"
    )

    if summary.get("avg_sharpe_across_symbols"):
        print(f"Average Sharpe Ratio: {summary['avg_sharpe_across_symbols']:.3f}")
        print(f"Average Win Rate: {summary['avg_win_rate_across_symbols']:.3f}")
        print(f"Best Performing: {summary.get('best_performing_symbol', 'N/A')}")

    print("\n✅ Signal template generation completed!")
    print("📁 Results saved to: results/signal_templates/")


if __name__ == "__main__":
    main()
