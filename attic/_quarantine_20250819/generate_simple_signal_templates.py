#!/usr/bin/env python3
"""
Simple Signal Template Generator

This script generates signal templates using historical data analysis
without the complex walk-forward framework that has numba issues.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from core.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class SimpleSignalTemplateGenerator:
    """Generate signal templates using simple historical analysis."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the simple signal template generator.

        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config
        self.symbols = config.get("symbols", ["SPY"])
        self.lookback_days = config.get("lookback_days", 252)
        self.min_history_days = config.get("min_history_days", 500)

        # Results storage
        self.templates = {}
        self.performance_metrics = {}

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return pd.DataFrame()

            logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Calculate technical indicators from price data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary of technical indicators
        """
        close = data["Close"]
        data["High"]
        data["Low"]
        volume = data["Volume"]

        indicators = {}

        # Moving averages
        indicators["sma_20"] = close.rolling(window=20).mean()
        indicators["sma_50"] = close.rolling(window=50).mean()
        indicators["sma_200"] = close.rolling(window=200).mean()

        # Price ratios
        indicators["price_to_sma20"] = close / indicators["sma_20"]
        indicators["price_to_sma50"] = close / indicators["sma_50"]
        indicators["sma20_to_sma50"] = indicators["sma_20"] / indicators["sma_50"]

        # Volatility
        returns = close.pct_change()
        indicators["volatility_20"] = returns.rolling(window=20).std()
        indicators["volatility_50"] = returns.rolling(window=50).std()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        indicators["macd"] = ema_12 - ema_26
        indicators["macd_signal"] = indicators["macd"].ewm(span=9).mean()
        indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]

        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        indicators["bb_upper"] = bb_middle + (bb_std * 2)
        indicators["bb_lower"] = bb_middle - (bb_std * 2)
        indicators["bb_position"] = (close - indicators["bb_lower"]) / (
            indicators["bb_upper"] - indicators["bb_lower"]
        )

        # Volume indicators
        indicators["volume_sma"] = volume.rolling(window=20).mean()
        indicators["volume_ratio"] = volume / indicators["volume_sma"]

        return indicators

    def generate_simple_signals(
        self, data: pd.DataFrame, indicators: dict[str, pd.Series]
    ) -> pd.Series:
        """
        Generate simple trading signals based on technical indicators.

        Args:
            data: OHLCV DataFrame
            indicators: Technical indicators dictionary

        Returns:
            Series of trading signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=data.index)

        # Trend following signals
        trend_buy = (
            (indicators["price_to_sma20"] > 1.0)
            & (indicators["sma20_to_sma50"] > 1.0)  # Price above 20-day SMA
            & (indicators["rsi"] < 70)  # 20-day SMA above 50-day SMA
            & (  # RSI not overbought
                indicators["macd"] > indicators["macd_signal"]
            )  # MACD bullish
        )

        trend_sell = (
            (indicators["price_to_sma20"] < 1.0)
            & (indicators["sma20_to_sma50"] < 1.0)  # Price below 20-day SMA
            & (indicators["rsi"] > 30)  # 20-day SMA below 50-day SMA
            & (  # RSI not oversold
                indicators["macd"] < indicators["macd_signal"]
            )  # MACD bearish
        )

        # Mean reversion signals
        mean_rev_buy = (
            (indicators["bb_position"] < 0.2)
            & (indicators["rsi"] < 30)  # Price near lower Bollinger Band
            & (indicators["volume_ratio"] > 1.5)  # RSI oversold  # High volume
        )

        mean_rev_sell = (
            (indicators["bb_position"] > 0.8)
            & (indicators["rsi"] > 70)  # Price near upper Bollinger Band
            & (indicators["volume_ratio"] > 1.5)  # RSI overbought  # High volume
        )

        # Combine signals
        signals[trend_buy | mean_rev_buy] = 1
        signals[trend_sell | mean_rev_sell] = -1

        return signals

    def calculate_performance_metrics(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> dict[str, float]:
        """
        Calculate performance metrics from signals.

        Args:
            data: OHLCV DataFrame
            signals: Trading signals series

        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        price_returns = data["Close"].pct_change()

        # Calculate strategy returns (assuming perfect execution at close)
        strategy_returns = signals.shift(1) * price_returns

        # Remove NaN values
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (signals != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "total_trades": int(total_trades),
        }

    def analyze_symbol(self, symbol: str) -> dict[str, Any]:
        """
        Analyze a symbol and generate signal template.

        Args:
            symbol: Trading symbol

        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing {symbol}")

        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.min_history_days)

        # Fetch data
        data = self.fetch_historical_data(symbol, str(start_date), str(end_date))

        if data.empty:
            logger.error(f"No data available for {symbol}")
            return {}

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)

        # Generate signals
        signals = self.generate_simple_signals(data, indicators)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(data, signals)

        # Generate signal template
        signal_template = self._generate_signal_template(symbol, metrics, indicators)

        return {
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "data_period": {
                "start_date": data.index[0].strftime("%Y-%m-%d"),
                "end_date": data.index[-1].strftime("%Y-%m-%d"),
                "total_days": len(data),
            },
            "performance_metrics": metrics,
            "signal_template": signal_template,
            "indicators_summary": self._summarize_indicators(indicators),
        }

    def _generate_signal_template(
        self, symbol: str, metrics: dict[str, float], indicators: dict[str, pd.Series]
    ) -> dict[str, Any]:
        """
        Generate signal template from metrics and indicators.

        Args:
            symbol: Trading symbol
            metrics: Performance metrics
            indicators: Technical indicators

        Returns:
            Signal template dictionary
        """
        # Analyze recent market conditions
        recent_data = {k: v.tail(50) for k, v in indicators.items()}

        # Determine market regime
        avg_volatility = recent_data["volatility_20"].mean()
        avg_rsi = recent_data["rsi"].mean()
        trend_strength = recent_data["sma20_to_sma50"].mean()

        # Determine signal type based on performance and market conditions
        if metrics["sharpe_ratio"] > 0.5:
            signal_type = "regime_aware_ensemble"
            confidence = min(1.0, metrics["sharpe_ratio"] / 2.0)
        else:
            signal_type = "mean_reversion"
            confidence = 0.3

        # Determine regime weights based on market conditions
        if trend_strength > 1.02:  # Strong uptrend
            trend_weight = 0.7
            mean_rev_weight = 0.3
        elif trend_strength < 0.98:  # Strong downtrend
            trend_weight = 0.3
            mean_rev_weight = 0.7
        else:  # Sideways market
            trend_weight = 0.5
            mean_rev_weight = 0.5

        # Adjust weights based on volatility
        if avg_volatility > 0.02:  # High volatility
            trend_weight *= 0.8
            mean_rev_weight *= 1.2

        # Normalize weights
        total_weight = trend_weight + mean_rev_weight
        trend_weight /= total_weight
        mean_rev_weight /= total_weight

        template = {
            "type": signal_type,
            "confidence": float(confidence),
            "market_regime": {
                "volatility_level": "high" if avg_volatility > 0.02 else "low",
                "trend_strength": "strong" if abs(trend_strength - 1.0) > 0.02 else "weak",
                "rsi_regime": "overbought"
                if avg_rsi > 70
                else "oversold"
                if avg_rsi < 30
                else "neutral",
            },
            "signal_parameters": {
                "rsi_oversold_threshold": 30.0,
                "rsi_overbought_threshold": 70.0,
                "sma_crossover_threshold": 1.0,
                "volume_threshold": 1.5,
                "bb_threshold": 0.2,
            },
            "regime_weights": {
                "trend_following_weight": float(trend_weight),
                "mean_reversion_weight": float(mean_rev_weight),
            },
            "position_sizing": {
                "max_position_size_pct": min(10.0, max(5.0, metrics["sharpe_ratio"] * 10)),
                "base_position_size": 1000.0,
                "volatility_adjustment": True,
            },
            "risk_management": {
                "stop_loss_pct": max(2.0, metrics["max_drawdown"] * 100),
                "take_profit_pct": metrics["sharpe_ratio"] * 5,
                "max_daily_loss_pct": 2.0,
            },
        }

        return template

    def _summarize_indicators(self, indicators: dict[str, pd.Series]) -> dict[str, float]:
        """
        Summarize technical indicators.

        Args:
            indicators: Technical indicators dictionary

        Returns:
            Summary statistics
        """
        summary = {}

        for name, series in indicators.items():
            if not series.empty:
                summary[f"{name}_mean"] = float(series.mean())
                summary[f"{name}_std"] = float(series.std())
                summary[f"{name}_current"] = float(series.iloc[-1])

        return summary

    def generate_all_templates(self) -> dict[str, Any]:
        """
        Generate signal templates for all symbols.

        Returns:
            Dictionary with all templates and summary
        """
        logger.info("Starting simple signal template generation for all symbols")

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
                result = self.analyze_symbol(symbol)

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
                all_sharpes.append(metrics.get("sharpe_ratio", 0))
                all_win_rates.append(metrics.get("win_rate", 0))

            summary_stats.update(
                {
                    "avg_sharpe_across_symbols": float(np.mean(all_sharpes)),
                    "avg_win_rate_across_symbols": float(np.mean(all_win_rates)),
                    "best_performing_symbol": max(
                        all_results.keys(),
                        key=lambda x: all_results[x]["performance_metrics"]["sharpe_ratio"],
                    ),
                }
            )

        return {"summary": summary_stats, "templates": all_results}

    def save_templates(self, results: dict[str, Any], output_dir: str = "results/signal_templates"):
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
        results_file = output_path / f"simple_signal_templates_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save individual templates for easy access
        templates_dir = output_path / "individual_templates"
        templates_dir.mkdir(exist_ok=True)

        for symbol, template_data in results.get("templates", {}).items():
            template_file = templates_dir / f"{symbol}_simple_template_{timestamp}.json"
            with open(template_file, "w") as f:
                json.dump(template_data, f, indent=2, default=str)

        # Save summary
        summary_file = output_path / f"simple_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(results.get("summary", {}), f, indent=2, default=str)

        logger.info(f"Templates saved to {output_path}")
        logger.info(f"Full results: {results_file}")
        logger.info(f"Individual templates: {templates_dir}")
        logger.info(f"Summary: {summary_file}")


def main():
    """Main function to run simple signal template generation."""
    # Configuration
    config = {
        "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
        "lookback_days": 252,
        "min_history_days": 1000,
    }

    print("🚀 Starting Simple Signal Template Generation")
    print("=" * 50)

    # Initialize generator
    generator = SimpleSignalTemplateGenerator(config)

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

    print("\n✅ Simple signal template generation completed!")
    print("📁 Results saved to: results/signal_templates/")


if __name__ == "__main__":
    main()
