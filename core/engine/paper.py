"""
Core Paper Trading Engine
Handles the main trading logic and state management
"""

import json
import os
from datetime import date as date_class
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from brokers.data_provider import IBKRDataProvider
from brokers.ibkr_broker import IBKRConfig
from core.enhanced_logging import TradingLogger
from core.feature_reweighter import AdaptiveFeatureEngine, FeatureReweighter
from core.notifications import DiscordConfig, DiscordNotifier
from core.regime_detector import RegimeDetector
from core.utils import ensure_directories
from strategies.factory import strategy_factory
from strategies.regime_aware_ensemble import (
    RegimeAwareEnsembleParams,
    RegimeAwareEnsembleStrategy,
)


class PaperTradingEngine:
    """
    Core paper trading engine with regime detection and adaptive features.
    """

    def __init__(
        self,
        config_file: str = "config/enhanced_paper_trading_config.json",
        profile_file: str = None,
    ):
        """
        Initialize paper trading engine.

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

        # Initialize components
        self._initialize_components()

        # Paper trading state
        self.capital = self.config.get("initial_capital", 100000)
        self.positions = {}
        self.trade_history = []
        self.daily_returns = []
        self.regime_history = []

        # Performance tracking
        self.performance_metrics = {}

        # Log system startup
        self._log_startup()

    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize kill switches
        self.kill_switches = self._initialize_kill_switches()

        # Initialize Discord notifications
        self.discord_notifier = self._setup_discord_notifications()

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

    def _log_startup(self):
        """Log system startup information."""
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
            f"Initialized PaperTradingEngine with ${self.capital:,.0f} capital"
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
            "max_drawdown_pct": 10.0,
            "max_position_size_pct": 20.0,
            "max_sector_exposure_pct": 30.0,
        }

        for key, default_value in defaults.items():
            if key not in kill_switches:
                kill_switches[key] = default_value

        return kill_switches

    def check_kill_switches(self) -> bool:
        """Check if any kill switches are triggered."""
        if not self.kill_switches.get("enabled", True):
            return False

        # Calculate daily P&L
        if len(self.daily_returns) > 0:
            daily_pnl = self.daily_returns[-1].get("pnl", 0)
            daily_pnl_pct = (daily_pnl / self.capital) * 100

            # Check daily loss limits
            max_daily_loss_pct = self.kill_switches.get("max_daily_loss_pct", 2.0)
            max_daily_loss_dollars = self.kill_switches.get(
                "max_daily_loss_dollars", 2000
            )

            if daily_pnl_pct < -max_daily_loss_pct:
                self.logger.warning(
                    f"Kill switch triggered: Daily loss {daily_pnl_pct:.2f}% exceeds limit of {max_daily_loss_pct}%"
                )
                return True

            if daily_pnl < -max_daily_loss_dollars:
                self.logger.warning(
                    f"Kill switch triggered: Daily loss ${daily_pnl:.2f} exceeds limit of ${max_daily_loss_dollars}"
                )
                return True

        # Check drawdown
        if len(self.daily_returns) > 0:
            peak_capital = max(
                [r.get("capital", self.capital) for r in self.daily_returns]
            )
            current_capital = self.capital
            drawdown_pct = ((peak_capital - current_capital) / peak_capital) * 100

            max_drawdown_pct = self.kill_switches.get("max_drawdown_pct", 10.0)
            if drawdown_pct > max_drawdown_pct:
                self.logger.warning(
                    f"Kill switch triggered: Drawdown {drawdown_pct:.2f}% exceeds limit of {max_drawdown_pct}%"
                )
                return True

        return False

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "symbols": ["SPY"],
            "initial_capital": 100000,
            "risk_params": {
                "max_position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
            },
            "execution_params": {
                "commission": 0.0,
                "slippage": 0.0,
            },
            "kill_switches": {
                "enabled": True,
                "max_daily_loss_pct": 2.0,
                "max_daily_loss_dollars": 2000,
                "max_drawdown_pct": 10.0,
            },
        }

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize trading strategies."""
        strategies = {}

        # Initialize regime-aware ensemble strategy
        regime_params = RegimeAwareEnsembleParams(
            lookback_period=252,
            regime_confidence_threshold=0.7,
            regime_weights={
                "trend": 0.4,
                "mean_reversion": 0.3,
                "chop": 0.3,
            },
        )

        strategies["regime_aware_ensemble"] = RegimeAwareEnsembleStrategy(
            params=regime_params
        )

        # Initialize other strategies if configured
        strategy_configs = self.config.get("strategies", {})
        for strategy_name, strategy_config in strategy_configs.items():
            if strategy_name != "regime_aware_ensemble":
                try:
                    strategies[strategy_name] = strategy_factory(
                        strategy_name, strategy_config
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize strategy {strategy_name}: {e}"
                    )

        return strategies

    def run_daily_trading(self, date: Optional[date_class] = None):
        """Run daily trading cycle."""
        if date is None:
            date = date_class.today()

        self.logger.info(f"Starting daily trading for {date}")

        # Check kill switches
        if self.check_kill_switches():
            self.logger.warning("Kill switches triggered - stopping trading")
            return

        # Get market data
        symbols = self.config.get("symbols", ["SPY"])
        data = self._get_market_data(symbols, date)

        if data is None or data.empty:
            self.logger.warning("No market data available - skipping trading")
            return

        # Generate regime-aware signals
        signals = self._generate_regime_aware_signals(data, date)

        # Execute trades
        self._execute_trades(signals, date)

        # Update performance tracking
        self._update_performance_tracking(date)

        # Log daily summary
        regime_name = self.regime_detector.get_current_regime_name()
        confidence = self.regime_detector.get_regime_confidence()
        self._log_daily_summary(date, regime_name, confidence)

        self.logger.info(f"Completed daily trading for {date}")

    def _get_market_data(
        self, symbols: List[str], date: date_class
    ) -> Optional[pd.DataFrame]:
        """Get market data for symbols."""
        try:
            if self.use_ibkr and self.data_provider:
                # Use IBKR data provider
                data = {}
                for symbol in symbols:
                    symbol_data = self.data_provider.get_historical_data(
                        symbol, start_date=date, end_date=date, interval="1d"
                    )
                    if symbol_data is not None and not symbol_data.empty:
                        data[symbol] = symbol_data

                if data:
                    # Combine data from multiple symbols
                    combined_data = pd.concat(data.values(), keys=data.keys(), axis=1)
                    return combined_data
            else:
                # Use yfinance as fallback
                data = {}
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(
                            start=date, end=date + pd.Timedelta(days=1), interval="1d"
                        )
                        if not hist.empty:
                            data[symbol] = hist
                    except Exception as e:
                        self.logger.error(f"Failed to get data for {symbol}: {e}")

                if data:
                    # Combine data from multiple symbols
                    combined_data = pd.concat(data.values(), keys=data.keys(), axis=1)
                    return combined_data

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")

        return None

    def _generate_regime_aware_signals(
        self, data: pd.DataFrame, date: date_class
    ) -> Dict[str, float]:
        """Generate regime-aware trading signals."""
        signals = {}

        try:
            # Detect market regime
            regime = self.regime_detector.detect_regime(data)
            regime_name = self.regime_detector.get_current_regime_name()
            confidence = self.regime_detector.get_regime_confidence()

            self.logger.info(
                f"Detected regime: {regime_name} (confidence: {confidence:.2f})"
            )

            # Generate base features
            features = self._generate_base_features(data)

            # Generate signals using regime-aware ensemble
            if "regime_aware_ensemble" in self.strategies:
                strategy = self.strategies["regime_aware_ensemble"]
                signals = strategy.generate_signals(features, regime)

            # Apply adaptive feature reweighting
            if self.adaptive_feature_engine:
                signals = self.adaptive_feature_engine.adapt_signals(signals, features)

        except Exception as e:
            self.logger.error(f"Failed to generate signals: {e}")

        return signals

    def _generate_base_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate base features from market data."""
        features = {}

        try:
            # Extract close prices
            close_cols = [col for col in data.columns if "Close" in col]
            if close_cols:
                close = data[close_cols[0]]  # Use first symbol's close price

                # Calculate technical indicators
                features["rsi"] = self._calculate_rsi(close)
                features["bb_position"] = self._calculate_bb_position(close)
                features["sma_20"] = close.rolling(window=20).mean()
                features["sma_50"] = close.rolling(window=50).mean()
                features["volatility"] = close.pct_change().rolling(window=20).std()

        except Exception as e:
            self.logger.error(f"Failed to generate features: {e}")

        return features

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bb_position(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands position."""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        bb_position = (close - sma) / (2 * std)
        return bb_position

    def _execute_trades(self, signals: Dict[str, float], date: date_class):
        """Execute trades based on signals."""
        if not signals:
            return

        symbols = self.config.get("symbols", ["SPY"])
        risk_params = self.config.get("risk_params", {})
        execution_params = self.config.get("execution_params", {})

        max_position_size = risk_params.get("max_position_size", 0.1)
        commission = execution_params.get("commission", 0.0)
        slippage = execution_params.get("slippage", 0.0)

        for symbol, signal in signals.items():
            if symbol not in symbols:
                continue

            # Get current price
            current_price = self._get_current_price(symbol, date)
            if current_price is None:
                continue

            # Calculate position size
            position_value = self.capital * max_position_size * abs(signal)
            shares = int(position_value / current_price)

            if shares == 0:
                continue

            # Determine trade direction
            if signal > 0:
                action = "BUY"
                # Check if we have enough capital
                trade_value = shares * current_price * (1 + commission + slippage)
                if trade_value > self.capital:
                    shares = int(
                        self.capital / (current_price * (1 + commission + slippage))
                    )
                    if shares == 0:
                        continue
            else:
                action = "SELL"
                # Check if we have enough shares to sell
                current_position = self.positions.get(symbol, 0)
                if current_position < shares:
                    shares = current_position
                    if shares == 0:
                        continue

            # Execute trade
            if shares > 0:
                trade = {
                    "date": date,
                    "symbol": symbol,
                    "action": action,
                    "shares": shares,
                    "price": current_price,
                    "value": shares * current_price,
                    "commission": shares * current_price * commission,
                    "slippage": shares * current_price * slippage,
                    "signal": signal,
                }

                # Update positions and capital
                self._update_capital_from_trade(trade["value"], current_price, shares)

                # Update position
                if action == "BUY":
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                else:
                    self.positions[symbol] = self.positions.get(symbol, 0) - shares

                # Add to trade history
                self.trade_history.append(trade)

                self.logger.info(
                    f"Executed {action} {shares} shares of {symbol} at ${current_price:.2f}"
                )

    def _update_capital_from_trade(self, trade_value: float, price: float, size: float):
        """Update capital after trade execution."""
        # This is a simplified implementation
        # In a real system, you'd track P&L more carefully
        commission = trade_value * 0.001  # Simplified commission
        self.capital -= commission

    def _update_performance_tracking(self, date: date_class):
        """Update performance tracking metrics."""
        # Calculate current portfolio value
        portfolio_value = self.capital
        for symbol, shares in self.positions.items():
            current_price = self._get_current_price(symbol, date)
            if current_price is not None:
                portfolio_value += shares * current_price

        # Calculate daily P&L
        if len(self.daily_returns) > 0:
            prev_value = self.daily_returns[-1].get("portfolio_value", self.capital)
            daily_pnl = portfolio_value - prev_value
        else:
            daily_pnl = 0

        # Record daily return
        daily_return = {
            "date": date,
            "capital": self.capital,
            "portfolio_value": portfolio_value,
            "pnl": daily_pnl,
            "positions": dict(self.positions),
        }

        self.daily_returns.append(daily_return)

        # Record regime
        regime_name = self.regime_detector.get_current_regime_name()
        confidence = self.regime_detector.get_regime_confidence()
        regime_record = {
            "date": date,
            "regime": regime_name,
            "confidence": confidence,
        }
        self.regime_history.append(regime_record)

    def _get_current_price(self, symbol: str, date: date_class) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.use_ibkr and self.data_provider:
                data = self.data_provider.get_historical_data(
                    symbol, start_date=date, end_date=date, interval="1d"
                )
                if data is not None and not data.empty:
                    return data["Close"].iloc[-1]
            else:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=date, end=date + pd.Timedelta(days=1), interval="1d"
                )
                if not hist.empty:
                    return hist["Close"].iloc[-1]
        except Exception as e:
            self.logger.error(f"Failed to get price for {symbol}: {e}")

        return None

    def _log_daily_summary(self, date: date_class, regime_name: str, confidence: float):
        """Log daily trading summary."""
        # Calculate summary metrics
        total_trades = len([t for t in self.trade_history if t["date"] == date])
        total_volume = sum(
            [t["value"] for t in self.trade_history if t["date"] == date]
        )

        # Calculate portfolio metrics
        portfolio_value = self.capital
        for symbol, shares in self.positions.items():
            current_price = self._get_current_price(symbol, date)
            if current_price is not None:
                portfolio_value += shares * current_price

        # Log summary
        summary = {
            "date": date,
            "capital": self.capital,
            "portfolio_value": portfolio_value,
            "total_return": (
                (portfolio_value - self.config.get("initial_capital", 100000))
                / self.config.get("initial_capital", 100000)
            )
            * 100,
            "regime": regime_name,
            "regime_confidence": confidence,
            "trades": total_trades,
            "volume": total_volume,
            "positions": dict(self.positions),
        }

        self.logger.info(
            f"Daily Summary: Portfolio ${portfolio_value:,.0f}, "
            f"Return {summary['total_return']:.2f}%, "
            f"Regime {regime_name} ({confidence:.2f}), "
            f"Trades {total_trades}"
        )

        # Send Discord notification if enabled
        if self.discord_notifier:
            self.discord_notifier.send_daily_summary(summary)

    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        if not self.daily_returns:
            return {}

        # Calculate performance metrics
        initial_capital = self.config.get("initial_capital", 100000)
        final_value = self.daily_returns[-1]["portfolio_value"]
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        # Calculate daily returns
        daily_returns_list = []
        for i, daily in enumerate(self.daily_returns):
            if i == 0:
                daily_return_pct = 0
            else:
                prev_value = self.daily_returns[i - 1]["portfolio_value"]
                daily_return_pct = (
                    (daily["portfolio_value"] - prev_value) / prev_value
                ) * 100
            daily_returns_list.append(daily_return_pct)

        # Calculate statistics
        returns_array = np.array(daily_returns_list)
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        sharpe_ratio = (
            (np.mean(returns_array) * 252) / volatility if volatility > 0 else 0
        )

        # Calculate drawdown
        peak = initial_capital
        max_drawdown = 0
        for daily in self.daily_returns:
            if daily["portfolio_value"] > peak:
                peak = daily["portfolio_value"]
            drawdown = (peak - daily["portfolio_value"]) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "total_trades": len(self.trade_history),
            "win_rate": self._calculate_win_rate(),
            "regime_distribution": self._get_regime_distribution(),
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0

        # Group trades by date and calculate daily P&L
        daily_pnl = {}
        for trade in self.trade_history:
            date = trade["date"]
            if date not in daily_pnl:
                daily_pnl[date] = 0
            # Simplified P&L calculation
            daily_pnl[date] += trade.get("pnl", 0)

        winning_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        total_days = len(daily_pnl)

        return (winning_days / total_days * 100) if total_days > 0 else 0.0

    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of market regimes."""
        if not self.regime_history:
            return {}

        regime_counts = {}
        total_days = len(self.regime_history)

        for record in self.regime_history:
            regime = record["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            regime: (count / total_days * 100)
            for regime, count in regime_counts.items()
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
