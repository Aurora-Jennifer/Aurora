"""
Enhanced Paper Trading System
Integrates regime detection, adaptive features, and regime-aware ensemble strategy
"""

import argparse
import json
import os
import sys
from datetime import date as date_class
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from brokers.data_provider import IBKRDataProvider
from brokers.ibkr_broker import IBKRConfig
from core.enhanced_logging import TradingLogger
from core.feature_reweighter import AdaptiveFeatureEngine, FeatureReweighter
from core.notifications import DiscordConfig, DiscordNotifier
from core.regime_detector import RegimeDetector
from core.utils import ensure_directories

# Removed backtest_engine import - using simplified data management
from strategies.factory import strategy_factory
from strategies.regime_aware_ensemble import (
    RegimeAwareEnsembleParams,
    RegimeAwareEnsembleStrategy,
)


class EnhancedPaperTradingSystem:
    """
    Enhanced paper trading system with regime detection and adaptive features.
    """

    def __init__(
        self,
        config_file: str = "config/enhanced_paper_trading_config.json",
        profile_file: str = None,
    ):
        """
        Initialize enhanced paper trading system.

        Args:
            config_file: Configuration file path
            profile_file: Profile configuration file path (optional)
        """
        self.config_file = config_file
        self.profile_file = profile_file

        # Setup enhanced logging
        self.trading_logger = TradingLogger()
        self.logger = self.trading_logger.main_logger

        # Load config after logging is set up
        self.config = self.load_config()

        # Load profile configuration if provided
        if profile_file and Path(profile_file).exists():
            self.load_profile_config(profile_file)

        # Initialize kill switches
        self.kill_switches = self._initialize_kill_switches()

        # Initialize Discord notifications
        self.discord_notifier = self._setup_discord_notifications()

        # Initialize components
        # Initialize IBKR data provider
        self.use_ibkr = self.config.get("use_ibkr", False)
        if self.use_ibkr:
            self.ibkr_config = IBKRConfig()
            self.data_provider = IBKRDataProvider(
                config=self.ibkr_config, use_cache=True, fallback_to_yfinance=True
            )
            self.logger.info("Initialized IBKR data provider")
        else:
            self.data_provider = None
            self.logger.info("Using yfinance for data (IBKR disabled)")

        # Initialize other components
        self.regime_detector = RegimeDetector(lookback_period=252)
        self.feature_reweighter = FeatureReweighter(
            rolling_window=60, reweight_frequency=20
        )
        self.adaptive_feature_engine = AdaptiveFeatureEngine(self.feature_reweighter)

        # Initialize strategies
        self.strategies = self._initialize_strategies()

        # Paper trading state
        self.capital = self.config.get("initial_capital", 100000)
        self.positions = {}
        self.trade_history = []
        self.daily_returns = []
        self.regime_history = []

        # Performance tracking
        self.performance_metrics = {}

        # Log system startup
        system_info = {
            "initial_capital": self.capital,
            "strategies": list(self.strategies.keys()),
            "symbols": self.config.get("symbols", []),
            "config_file": self.config_file,
        }
        self.trading_logger.log_system_startup(system_info)

        # Send Discord startup notification
        if self.discord_notifier:
            self.discord_notifier.send_startup_notification(system_info)

        self.logger.info(
            f"Initialized EnhancedPaperTradingSystem with ${self.capital:,.0f} capital"
        )

    def _setup_discord_notifications(self) -> Optional[DiscordNotifier]:
        """Setup Discord notifications."""
        try:
            discord_config_path = "config/notifications/discord_config.json"
            if os.path.exists(discord_config_path):
                with open(discord_config_path) as f:
                    discord_config_data = json.load(f)

                if (
                    discord_config_data.get("enabled", False)
                    and discord_config_data.get("webhook_url")
                    != "YOUR_DISCORD_WEBHOOK_URL_HERE"
                ):
                    config = DiscordConfig(
                        webhook_url=discord_config_data["webhook_url"],
                        bot_name=discord_config_data.get("bot_name", "Trading Bot"),
                        bot_avatar=discord_config_data.get(
                            "bot_avatar", "https://cdn.discordapp.com/emojis/📈.png"
                        ),
                        enabled=discord_config_data.get("enabled", True),
                    )
                    return DiscordNotifier(config)
                else:
                    self.logger.info("Discord notifications disabled or not configured")
            else:
                self.logger.info("Discord config file not found")
        except Exception as e:
            self.logger.error(f"Failed to setup Discord notifications: {e}")

        return None

    def load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open(self.config_file) as f:
                config = json.load(f)
            self.logger.info(f"Loaded config from {self.config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def load_profile_config(self, profile_file: str):
        """Load and apply profile configuration."""
        try:
            with open(profile_file) as f:
                profile = json.load(f)

            # Override config with profile settings
            if "risk_params" in profile:
                self.config["risk_params"].update(profile["risk_params"])

            if "execution_params" in profile:
                self.config["execution_params"].update(profile["execution_params"])

            if "symbols" in profile:
                self.config["symbols"] = profile["symbols"]

            if "initial_capital" in profile:
                self.config["initial_capital"] = profile["initial_capital"]
                self.capital = profile["initial_capital"]

            self.logger.info(f"Loaded profile from {profile_file}")

        except Exception as e:
            self.logger.error(f"Failed to load profile: {e}")

    def _initialize_kill_switches(self) -> Dict:
        """Initialize kill switches for risk management."""
        kill_switches = self.config.get("kill_switches", {})

        # Set defaults if not provided
        defaults = {
            "enabled": True,
            "max_daily_loss_pct": 2.0,
            "max_daily_loss_dollars": 2000,
            "max_gross_exposure_pct": 50.0,
            "max_order_rate_per_minute": 1,
            "max_unhandled_exceptions": 1,
        }

        for key, default_value in defaults.items():
            if key not in kill_switches:
                kill_switches[key] = default_value

        # Log kill switch configuration
        self.logger.info("Kill switches configured:")
        for key, value in kill_switches.items():
            self.logger.info(f"  {key}: {value}")

        return kill_switches

    def check_kill_switches(self) -> bool:
        """Check if any kill switches should be triggered."""
        if not self.kill_switches.get("enabled", True):
            return True

        try:
            # Check daily loss percentage
            initial_capital = self.config.get("initial_capital", 100000)
            current_capital = self.capital
            daily_loss_pct = (initial_capital - current_capital) / initial_capital * 100

            if daily_loss_pct > self.kill_switches.get("max_daily_loss_pct", 2.0):
                self.logger.error(
                    f"KILL SWITCH: Daily loss {daily_loss_pct:.2f}% exceeds limit {self.kill_switches.get('max_daily_loss_pct', 2.0)}%"
                )
                return False

            # Check daily loss dollars
            daily_loss_dollars = initial_capital - current_capital
            if daily_loss_dollars > self.kill_switches.get(
                "max_daily_loss_dollars", 2000
            ):
                self.logger.error(
                    f"KILL SWITCH: Daily loss ${daily_loss_dollars:.2f} exceeds limit ${self.kill_switches.get('max_daily_loss_dollars', 2000)}"
                )
                return False

            # Check gross exposure
            total_exposure = sum(abs(pos) for pos in self.positions.values())
            exposure_pct = total_exposure * 100
            if exposure_pct > self.kill_switches.get("max_gross_exposure_pct", 50.0):
                self.logger.error(
                    f"KILL SWITCH: Gross exposure {exposure_pct:.2f}% exceeds limit {self.kill_switches.get('max_gross_exposure_pct', 50.0)}%"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking kill switches: {e}")
            return False

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "initial_capital": 100000,
            "symbols": ["SPY", "QQQ", "IWM"],
            "strategies": ["regime_ensemble", "ensemble", "sma", "momentum"],
            "rebalance_frequency": "daily",
            "max_position_size": 0.2,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "regime_switching": True,
            "feature_adaptation": True,
            "performance_tracking": True,
        }

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize trading strategies."""
        strategies = {}
        strategy_names = self.config.get("strategies", ["regime_ensemble"])

        for strategy_name in strategy_names:
            try:
                if strategy_name == "regime_ensemble":
                    params = RegimeAwareEnsembleParams(
                        combination_method="rolling_ic",
                        confidence_threshold=0.3,
                        use_regime_switching=True,
                        regime_lookback=252,
                    )
                    strategies[strategy_name] = RegimeAwareEnsembleStrategy(params)
                else:
                    strategies[strategy_name] = strategy_factory.create_strategy(
                        strategy_name
                    )

                self.logger.info(f"Initialized strategy: {strategy_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")

        return strategies

    def run_daily_trading(self, date: Optional[date_class] = None):
        """
        Run daily trading cycle.

        Args:
            date: Trading date (defaults to today)
        """
        if date is None:
            date = date_class.today()

        self.logger.info(f"Running daily trading for {date}")

        try:
            # Check kill switches before trading
            if not self.check_kill_switches():
                self.logger.error("KILL SWITCH TRIGGERED - Trading halted")
                return

            # Get market data
            symbols = self.config.get("symbols", ["SPY"])
            data = self._get_market_data(symbols, date)

            if data is None or data.empty:
                self.logger.warning(f"No data available for {date}")
                return

            # Detect regime
            regime_name, confidence, regime_params = self.regime_detector.detect_regime(
                data
            )
            self.regime_history.append(
                {"date": date, "regime": regime_name, "confidence": confidence}
            )

            self.logger.info(
                f"Detected regime: {regime_name} (confidence: {confidence:.2f})"
            )

            # Generate signals for each symbol
            for symbol in symbols:
                symbol_data = (
                    data[data["Symbol"] == symbol] if "Symbol" in data.columns else data
                )

                if symbol_data.empty:
                    continue

                # Generate regime-aware signals
                signals = self._generate_regime_aware_signals(
                    symbol_data, regime_name, regime_params
                )

                # Execute trades
                self._execute_trades(symbol, signals, date, regime_params)

            # Update performance tracking
            self._update_performance_tracking(date)

            # Log daily summary
            self._log_daily_summary(date, regime_name, confidence)

        except Exception as e:
            self.logger.error(f"Error in daily trading for {date}: {e}")

    def _get_market_data(
        self, symbols: List[str], date: date_class
    ) -> Optional[pd.DataFrame]:
        """Get market data for symbols."""
        try:
            # Get data for the last 300 days to ensure enough history
            end_date = date
            start_date = end_date - timedelta(days=300)

            all_data = []
            for symbol in symbols:
                try:
                    if self.use_ibkr and self.data_provider:
                        # Use IBKR data provider
                        data = self.data_provider.get_daily_data(
                            symbol, start_date, end_date
                        )
                        if data is not None and not data.empty:
                            data["Symbol"] = symbol
                            all_data.append(data)
                    else:
                        # Fallback to yfinance
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(start=start_date, end=end_date)

                        if not data.empty:
                            data["Symbol"] = symbol
                            all_data.append(data)

                except Exception as e:
                    self.logger.error(f"Failed to get data for {symbol}: {e}")

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def _generate_regime_aware_signals(
        self, data: pd.DataFrame, regime_name: str, regime_params
    ) -> Dict[str, float]:
        """Generate regime-aware signals for all strategies."""
        signals = {}

        # Generate base features
        base_features = self._generate_base_features(data)

        # Update feature performance
        returns = data["Close"].pct_change().shift(-1)  # Next period returns
        self.feature_reweighter.update_feature_performance(
            base_features, returns, regime_name, data.index[-1]
        )

        # Generate adaptive features
        adaptive_features = self.adaptive_feature_engine.generate_adaptive_features(
            data, regime_name, base_features
        )

        # Generate signals for each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy_name == "regime_ensemble":
                    # Use regime-aware ensemble strategy
                    strategy_signals = strategy.generate_signals(data)
                    signals[strategy_name] = (
                        strategy_signals.iloc[-1] if not strategy_signals.empty else 0.0
                    )
                else:
                    # Use regular strategy
                    strategy_signals = strategy.generate_signals(data)
                    signals[strategy_name] = (
                        strategy_signals.iloc[-1] if not strategy_signals.empty else 0.0
                    )

            except Exception as e:
                self.logger.error(f"Error generating signals for {strategy_name}: {e}")
                signals[strategy_name] = 0.0

        return signals

    def _generate_base_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate base feature set."""
        close = data["Close"]
        high = data["High"] if "High" in data.columns else close
        low = data["Low"] if "Low" in data.columns else close
        volume = (
            data["Volume"]
            if "Volume" in data.columns
            else pd.Series(1.0, index=close.index)
        )

        features = {}

        # Price-based features
        features["returns_1d"] = close.pct_change()
        features["returns_5d"] = close.pct_change(5)
        features["returns_20d"] = close.pct_change(20)

        # Moving averages
        features["ma_20"] = close.rolling(20).mean()
        features["ma_50"] = close.rolling(50).mean()
        features["ma_200"] = close.rolling(200).mean()

        # Volatility
        features["volatility_20"] = close.pct_change().rolling(20).std()
        features["volatility_50"] = close.pct_change().rolling(50).std()

        # Volume
        features["volume_ratio"] = volume / volume.rolling(20).mean()

        # Technical indicators
        features["rsi"] = self._calculate_rsi(close)
        features["bb_position"] = self._calculate_bb_position(close)

        return features

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bb_position(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position."""
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        bb_upper = ma + (std * 2)
        bb_lower = ma - (std * 2)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        return bb_position

    def _execute_trades(
        self, symbol: str, signals: Dict[str, float], date: date_class, regime_params
    ):
        """Execute trades based on signals with position-aware logic."""
        current_price = self._get_current_price(symbol, date)
        if current_price is None:
            return

        # Get current position
        current_position = self.positions.get(symbol, 0.0)

        # Calculate position size based on regime
        position_multiplier = regime_params.position_sizing_multiplier
        max_position_size = (
            self.config.get("risk_params", {}).get("max_weight_per_symbol", 0.25)
            * position_multiplier
        )

        # Get regime-aware ensemble signal
        regime_signal = signals.get("regime_ensemble", 0.0)

        # Calculate target position with position-aware logic
        signal_strength = abs(regime_signal)
        if signal_strength < regime_params.confidence_threshold:
            target_position = 0.0
        else:
            # Calculate target position based on signal direction
            target_position = np.sign(regime_signal) * min(
                signal_strength, max_position_size
            )

            # ENFORCE REDUCE-ONLY LOGIC: No shorting unless explicitly enabled
            if target_position < 0 and current_position <= 0:
                # Cannot sell if we don't have a position
                target_position = 0.0
                self.logger.warning(f"Cannot sell {symbol} - no position to reduce")

            # ENFORCE POSITION LIMITS: Cannot exceed max position size
            if abs(target_position) > max_position_size:
                target_position = np.sign(target_position) * max_position_size

        # Execute trade if position changed significantly
        position_change = target_position - current_position
        if abs(position_change) > 0.01:  # 1% threshold
            # Calculate trade size
            trade_value = position_change * self.capital
            trade_size = trade_value / current_price

            # Validate trade size
            if abs(trade_size) < 0.01:  # Minimum trade size
                return

            # Enhanced trade logging
            trade_data = {
                "symbol": symbol,
                "action": "BUY" if trade_size > 0 else "SELL",
                "size": abs(trade_size),
                "price": current_price,
                "value": abs(trade_value),
                "regime": regime_params.regime_name,
                "confidence": regime_params.confidence_threshold,
                "signal_strength": signal_strength,
                "current_position": current_position,
                "target_position": target_position,
            }

            self.trading_logger.log_trade(trade_data)

            # Send Discord trade notification
            if self.discord_notifier:
                trade_info = {
                    "symbol": symbol,
                    "action": trade_data["action"],
                    "size": trade_data["size"],
                    "price": trade_data["price"],
                    "value": trade_data["value"],
                    "regime": regime_params.regime_name,
                    "confidence": regime_params.confidence_threshold,
                }
                self.discord_notifier.send_trade_notification(trade_info)

            # Record trade
            trade = {
                "date": date,
                "symbol": symbol,
                "action": trade_data["action"],
                "size": trade_data["size"],
                "price": current_price,
                "value": abs(trade_value),
                "regime": regime_params.regime_name,
                "signal_strength": signal_strength,
                "current_position": current_position,
                "target_position": target_position,
            }

            self.trade_history.append(trade)

            # Update position
            self.positions[symbol] = target_position

            # Update capital based on actual trade execution
            self._update_capital_from_trade(trade_value, current_price, trade_size)

    def _update_capital_from_trade(self, trade_value: float, price: float, size: float):
        """Update capital based on actual trade execution."""
        # Calculate transaction costs (fees, slippage)
        fees_bps = self.config.get("execution_params", {}).get("max_slippage_bps", 10)
        fees = abs(trade_value) * (fees_bps / 10000)

        # Update capital: subtract fees from trade value
        self.capital -= fees

        self.logger.info(
            f"Trade executed: ${trade_value:.2f}, Fees: ${fees:.2f}, Capital: ${self.capital:.2f}"
        )

    def _update_performance_tracking(self, date: date_class):
        """Update performance tracking with proper PnL calculation."""
        # Calculate total portfolio value including all positions
        total_value = self.capital  # Start with cash

        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol, date)
            if current_price and position != 0:
                # Calculate position value correctly
                # Position is a fraction of capital, so value = position * capital
                position_value = position * self.capital
                total_value += position_value

                self.logger.debug(
                    f"Position {symbol}: {position:.4f} @ ${current_price:.2f} = ${position_value:.2f}"
                )

        # Calculate daily return based on total value change
        if hasattr(self, "_previous_total_value"):
            daily_return = (
                total_value - self._previous_total_value
            ) / self._previous_total_value
        else:
            # First day - no return
            daily_return = 0.0

        self._previous_total_value = total_value

        # Store daily return
        self.daily_returns.append(
            {
                "date": date,
                "return": daily_return,
                "total_value": total_value,
                "cash": self.capital,
                "positions_value": total_value - self.capital,
            }
        )

        # Don't update capital here - keep it as cash
        # Capital should only change due to trades, not price movements

    def _get_current_price(self, symbol: str, date: date_class) -> Optional[float]:
        """Get current price for symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=date, end=date + timedelta(days=1))
            if not data.empty:
                return data["Close"].iloc[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def _log_daily_summary(self, date: date_class, regime_name: str, confidence: float):
        """Log daily trading summary."""
        # Calculate performance metrics
        if len(self.daily_returns) > 1:
            returns_series = pd.Series([d["return"] for d in self.daily_returns])
            total_return = (
                self.capital / self.config.get("initial_capital", 100000)
            ) - 1
            sharpe_ratio = (
                returns_series.mean() / (returns_series.std() + 1e-6) * np.sqrt(252)
            )
            max_drawdown = (
                returns_series.cumsum() - returns_series.cumsum().expanding().max()
            ).min()
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        # Enhanced performance logging
        performance_data = {
            "total_return": total_return,
            "current_capital": self.capital,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trade_history),
            "regime": regime_name,
            "regime_confidence": confidence,
        }

        self.trading_logger.log_performance(performance_data)

        # Send Discord daily summary
        if self.discord_notifier:
            summary_data = {
                "total_return": total_return,
                "current_capital": self.capital,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_trades": len(self.trade_history),
                "regime": regime_name,
                "regime_confidence": confidence,
            }
            self.discord_notifier.send_daily_summary(summary_data)

        # Log summary
        self.logger.info(f"Daily Summary - {date}:")
        self.logger.info(f"  Regime: {regime_name} (confidence: {confidence:.2f})")
        self.logger.info(f"  Capital: ${self.capital:,.2f}")
        self.logger.info(f"  Total Return: {total_return:.2%}")
        self.logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        self.logger.info(f"  Positions: {len(self.positions)}")

    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        if not self.daily_returns:
            return {}

        returns_series = pd.Series([d["return"] for d in self.daily_returns])
        dates = [d["date"] for d in self.daily_returns]

        # Calculate metrics
        total_return = (self.capital / self.config.get("initial_capital", 100000)) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = (
            returns_series.mean() / (returns_series.std() + 1e-6) * np.sqrt(252)
        )

        # Calculate drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Regime analysis
        regime_stats = {}
        for regime_entry in self.regime_history:
            regime = regime_entry["regime"]
            if regime not in regime_stats:
                regime_stats[regime] = {"count": 0, "avg_confidence": 0.0}
            regime_stats[regime]["count"] += 1
            regime_stats[regime]["avg_confidence"] += regime_entry["confidence"]

        for regime in regime_stats:
            regime_stats[regime]["avg_confidence"] /= regime_stats[regime]["count"]

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trade_history),
            "current_capital": self.capital,
            "regime_stats": regime_stats,
            "feature_performance": self.feature_reweighter.get_regime_performance_summary(
                "chop"
            ),  # Default to chop regime
        }

    def save_results(self, output_dir: str = "results"):
        """Save trading results to files."""
        ensure_directories(output_dir)

        # Save performance report
        report = self.get_performance_report()
        with open(f"{output_dir}/performance_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save trade history
        trade_df = pd.DataFrame(self.trade_history)
        if not trade_df.empty:
            trade_df.to_csv(f"{output_dir}/trade_history.csv", index=False)

        # Save daily returns
        returns_df = pd.DataFrame(self.daily_returns)
        if not returns_df.empty:
            returns_df.to_csv(f"{output_dir}/daily_returns.csv", index=False)

        # Save regime history
        regime_df = pd.DataFrame(self.regime_history)
        if not regime_df.empty:
            regime_df.to_csv(f"{output_dir}/regime_history.csv", index=False)

        self.logger.info(f"Results saved to {output_dir}")


def main():
    """Main function for running enhanced paper trading."""
    parser = argparse.ArgumentParser(description="Enhanced Paper Trading System")
    parser.add_argument("--daily", action="store_true", help="Run daily trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--setup-cron", action="store_true", help="Setup cron job")
    parser.add_argument("--cron", action="store_true", help="Running from cron job")
    parser.add_argument(
        "--config",
        type=str,
        default="config/enhanced_paper_trading_config.json",
        help="Configuration file path",
    )
    parser.add_argument("--profile", help="Profile configuration file path")
    parser.add_argument(
        "--paper", action="store_true", help="Run in paper trading mode"
    )

    args = parser.parse_args()

    try:
        # Initialize system
        system = EnhancedPaperTradingSystem(args.config, args.profile)

        if args.daily or args.cron:
            # Run daily trading
            system.run_daily_trading()
            system.save_results()

            # Send cron notification if running from cron
            if args.cron and system.discord_notifier:
                system.discord_notifier.send_cron_notification(
                    "SUCCESS", "Daily trading completed successfully"
                )

        elif args.backtest:
            # Run backtest (implement if needed)
            print("Backtest functionality not yet implemented")

        elif args.setup_cron:
            # Setup cron job
            cron_command = f"0 9 * * 1-5 cd {os.getcwd()} && python {__file__} --cron"
            print("Add this to your crontab:")
            print(f"{cron_command}")
            print("\nTo edit crontab: crontab -e")

        else:
            # Run daily trading by default
            system.run_daily_trading()
            system.save_results()

    except Exception as e:
        # Send error notification
        if (
            "system" in locals()
            and hasattr(system, "discord_notifier")
            and system.discord_notifier
        ):
            system.discord_notifier.send_error_notification(
                str(e), "Daily trading execution"
            )
        if "system" in locals() and hasattr(system, "trading_logger"):
            system.trading_logger.log_error(str(e), "Daily trading execution", e)
        raise


if __name__ == "__main__":
    main()
