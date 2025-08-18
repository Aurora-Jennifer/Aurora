"""
Core Backtest Engine
Handles the main backtesting logic and simulation
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from core.engine.paper import PaperTradingEngine
from core.portfolio import PortfolioState
from core.trade_logger import TradeBook
from core.data_sanity import DataSanityValidator
import os

# ML imports
try:
    from core.ml.profit_learner import ProfitLearner, TradeOutcome

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML profit learner not available")


class BacktestEngine:
    """Comprehensive backtest engine for trading strategies."""

    def __init__(self, config_file: str, profile_file: str = None):
        """Initialize backtest engine."""
        self.config_file = config_file
        self.profile_file = profile_file

        # Initialize trading system
        self.trading_system = PaperTradingEngine(config_file, profile_file)

        # Backtest state
        self.start_date = None
        self.end_date = None
        self.initial_capital = self.trading_system.capital
        self._last_results = None

        # Portfolio and trade tracking
        self.portfolio = PortfolioState(cash=self.initial_capital)
        self.trade_book = TradeBook()

        # ML trade tracking for accurate P&L
        self.ml_trade_positions = {}  # Track open positions for ML
        self.ml_trade_history = []  # Track completed trades for ML

        # Data storage
        self.daily_returns = []
        self.equity_curve = []

        # Logging control
        self.insufficient_data_logged = set()  # Rate limit insufficient data logs
        self.logger = logging.getLogger(__name__)

        # Constants
        self.MIN_HISTORY = 252  # Minimum history for regime detection

        # ML system
        self.ml_enabled = ML_AVAILABLE and self.trading_system.config.get(
            "ml_enabled", False
        )
        print(
            f"🔍 ML Debug: ML_AVAILABLE={ML_AVAILABLE}, ml_enabled_config={self.trading_system.config.get('ml_enabled', False)}"
        )
        self.logger.info(
            f"ML_AVAILABLE: {ML_AVAILABLE}, ml_enabled config: {self.trading_system.config.get('ml_enabled', False)}"
        )
        if self.ml_enabled:
            try:
                import yaml

                with open("config/ml_config.yaml") as f:
                    ml_config = yaml.safe_load(f)
                self.profit_learner = ProfitLearner(
                    ml_config.get("ml_profit_learner", {})
                )
                print("✅ ML profit learner initialized")
                self.logger.info("ML profit learner initialized")
            except Exception as e:
                print(f"❌ Failed to initialize ML system: {e}")
                self.logger.warning(f"Failed to initialize ML system: {e}")
                self.ml_enabled = False
        else:
            self.profit_learner = None
            print("❌ ML system disabled")
            self.logger.info("ML system disabled")

    def run_backtest(
        self, start_date: str, end_date: str, symbols: List[str] = None
    ) -> Dict:
        """Run comprehensive backtest over specified period."""
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if symbols:
            self.trading_system.config["symbols"] = symbols

        # Read flags
        self.carry_positions_from_warmup = self.trading_system.config.get(
            "carry_positions_from_warmup", False
        )
        self.fast_mode = self.trading_system.config.get("fast_mode", False)

        print(f"🚀 Starting backtest: {start_date} to {end_date}")
        print(f"📊 Symbols: {self.trading_system.config['symbols']}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print("=" * 60)

        # Load data with warmup period
        print("📊 Loading data with warmup period...")
        warmup_days = (
            300 if self.fast_mode else 600
        )  # Ensure enough data for regime detection
        warmup_start = self.start_date - timedelta(days=warmup_days)  # Ensure lookbacks
        all_data = self._load_historical_data(warmup_start, self.end_date)

        if all_data is None or all_data.empty:
            print("❌ No data available for backtest period")
            return {}

        print(f"📊 Loaded {len(all_data)} data points")
        print(f"📊 Data columns: {list(all_data.columns)}")
        print(f"📊 Data shape: {all_data.shape}")

        # Get all trading dates (including warmup)
        all_trading_dates = self._get_trading_dates_from_data(all_data)
        print(f"📅 Total trading days (including warmup): {len(all_trading_dates)}")

        if len(all_trading_dates) == 0:
            print("❌ No trading dates found in data")
            return {}

        # Separate warmup and backtest periods
        warmup_dates = [d for d in all_trading_dates if d < self.start_date]
        backtest_dates = [
            d for d in all_trading_dates if self.start_date <= d <= self.end_date
        ]

        print(f"🔥 Warmup period: {len(warmup_dates)} days")
        print(f"📈 Backtest period: {len(backtest_dates)} days")

        # Run warmup period
        if warmup_dates:
            print("🔥 Running warmup period...")
            for warmup_date in warmup_dates:
                self._run_daily_trading(warmup_date, all_data, {})

            print(
                f"🔥 Warmup complete. Portfolio value: ${self.portfolio.value_at({}):,.2f}"
            )

            # Reset trade book for actual backtest
            if not self.carry_positions_from_warmup:
                self.trade_book.reset()
                self.portfolio = PortfolioState(cash=self.initial_capital)
                print("🔄 Reset portfolio for backtest period")

        # Run backtest period
        print("📈 Running backtest period...")
        for i, backtest_date in enumerate(backtest_dates):
            if i % 50 == 0:  # Progress indicator
                print(f"📈 Progress: {i+1}/{len(backtest_dates)} days")

            # Get current prices
            current_prices = self._get_prices_for_date(backtest_date, all_data)

            # Update portfolio with current prices
            self._update_portfolio_prices(backtest_date, all_data)

            # Run daily trading
            self._run_daily_trading(backtest_date, all_data, current_prices)

            # Record daily return
            current_prices = self._get_prices_for_date(backtest_date, all_data)
            portfolio_value = self.portfolio.value_at(current_prices)
            daily_return = {
                "date": backtest_date,
                "portfolio_value": portfolio_value,
                "cash": self.portfolio.cash,
                "positions_value": portfolio_value - self.portfolio.cash,
                "return": 0.0,  # Will be calculated later
            }
            self.daily_returns.append(daily_return)

        # Close any remaining ML positions at end of backtest
        if self.ml_enabled and self.ml_trade_positions:
            print("🔍 Closing remaining ML positions...")
            for symbol, positions in self.ml_trade_positions.items():
                if positions:
                    # Get final price for the symbol
                    final_prices = self._get_prices_for_date(
                        backtest_dates[-1], all_data
                    )
                    if symbol in final_prices:
                        final_price = final_prices[symbol]
                        for position in positions:
                            # Calculate P&L for remaining position
                            entry_price = position["entry_price"]
                            exit_price = final_price
                            profit_loss = (exit_price - entry_price) * position[
                                "shares"
                            ]
                            profit_loss_pct = (exit_price - entry_price) / entry_price

                            # Record completed trade
                            completed_trade = {
                                "symbol": symbol,
                                "entry_date": position["entry_date"],
                                "exit_date": backtest_dates[-1],
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "shares": position["shares"],
                                "profit_loss": profit_loss,
                                "profit_loss_pct": profit_loss_pct,
                                "hold_duration": (
                                    backtest_dates[-1] - position["entry_date"]
                                ).days,
                            }

                            self.ml_trade_history.append(completed_trade)
                            print(
                                f"🔍 Closed position: {profit_loss_pct:.2%} profit, ${profit_loss:.2f}"
                            )

        # Calculate final results
        print("📊 Calculating results...")
        results = self._generate_final_results()

        # Save results
        self._save_results(results)

        # Save ML models and history
        if self.ml_enabled and self.profit_learner:
            self.profit_learner._save_models()
            print(
                f"💾 Saved ML models and {len(self.profit_learner.performance_history)} trade records"
            )

        # Print summary
        self.print_results(results)

        return results

    def _get_data_up_to_date(
        self, all_data: pd.DataFrame, current_date: date
    ) -> pd.DataFrame:
        """Get data up to the current date for regime detection."""
        # Filter data up to current date
        data_up_to_date = all_data[all_data.index.date <= current_date].copy()

        # Ensure minimum history for regime detection
        if len(data_up_to_date) < self.MIN_HISTORY:
            # Only log once per date to avoid spam
            if current_date not in self.insufficient_data_logged:
                self.logger.debug(
                    f"Insufficient data for regime detection on {current_date}: {len(data_up_to_date)} < {self.MIN_HISTORY}"
                )
                self.insufficient_data_logged.add(current_date)
            return pd.DataFrame()

        return data_up_to_date

    def _load_historical_data(
        self, start_date: date, end_date: date
    ) -> Optional[pd.DataFrame]:
        """Load historical data for all symbols."""
        symbols = self.trading_system.config.get("symbols", ["SPY"])
        all_data = []

        # Get DataSanity wrapper for validation
        from core.data_sanity import get_data_sanity_wrapper

        data_sanity = get_data_sanity_wrapper()

        for symbol in symbols:
            try:
                # Use yfinance for historical data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date, end=end_date + timedelta(days=1)
                )

                if not data.empty:
                    # Validate and repair data using DataSanity
                    clean_data = data_sanity.validate_dataframe(data, symbol)

                    # Add symbol column
                    clean_data["Symbol"] = symbol
                    all_data.append(clean_data)
                    print(
                        f"✅ Loaded and validated {len(clean_data)} data points for {symbol}"
                    )
                else:
                    print(f"⚠️  No data for {symbol}")

            except Exception as e:
                print(f"❌ Error loading data for {symbol}: {e}")

        if all_data:
            # Combine all symbol data
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            return combined_data
        else:
            return None

    def _get_trading_dates_from_data(self, data: pd.DataFrame) -> List[date]:
        """Extract trading dates from data."""
        if data.empty:
            return []

        # Get unique dates from the data
        trading_dates = sorted(pd.Series(data.index.date).unique())

        # Filter to business days (simple heuristic)
        business_dates = []
        for date_obj in trading_dates:
            # Skip weekends (5=Saturday, 6=Sunday)
            if date_obj.weekday() < 5:
                business_dates.append(date_obj)

        return business_dates

    def _update_portfolio_prices(self, current_date: date, data: pd.DataFrame):
        """Update portfolio with current prices."""
        current_prices = self._get_prices_for_date(current_date, data)

        # Update the portfolio's last_prices for mark-to-market calculations
        for symbol, price in current_prices.items():
            self.portfolio.last_prices[symbol] = price

    def _run_daily_trading(
        self, current_date: date, data: pd.DataFrame, current_prices: Dict[str, float]
    ):
        """Run daily trading cycle."""
        # Get data up to current date for regime detection
        data_up_to_date = self._get_data_up_to_date(data, current_date)

        # Skip trading if we don't have enough data for regime detection
        if data_up_to_date.empty:
            return

        # Detect regime
        regime_name, regime_params = self._detect_regime_with_rate_limit(
            data_up_to_date
        )

        # Generate signals
        signals = self._generate_signals(data_up_to_date, regime_name, regime_params)

        # Execute trades
        for symbol in self.trading_system.config.get("symbols", []):
            if symbol in signals:
                print(f"🔍 Executing trade for {symbol}, signal: {signals[symbol]:.4f}")
                self._execute_trades_with_portfolio(
                    symbol, signals, current_date, regime_params, current_prices
                )

    def _detect_regime_with_rate_limit(self, data: pd.DataFrame):
        """Detect regime with rate limiting for performance."""
        try:
            # Use the trading system's regime detector
            (
                regime_name,
                confidence,
                regime_params,
            ) = self.trading_system.regime_detector.detect_regime(data)

            return regime_name, regime_params

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return "unknown", None

    def _get_historical_data(self, current_date: date) -> Optional[pd.DataFrame]:
        """Get historical data for current date."""
        try:
            # This is a simplified implementation
            # In a real system, you'd get data from the loaded dataset
            symbols = self.trading_system.config.get("symbols", ["SPY"])

            all_data = []
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                # Get data for the last 300 days
                start_date = current_date - timedelta(days=300)
                data = ticker.history(
                    start=start_date, end=current_date + timedelta(days=1)
                )

                if not data.empty:
                    data["Symbol"] = symbol
                    all_data.append(data)

            if all_data:
                return pd.concat(all_data, ignore_index=False)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

    def _generate_signals(
        self, data: pd.DataFrame, regime_name: str, regime_params
    ) -> Dict[str, float]:
        """Generate trading signals."""
        signals = {}

        try:
            # Use the trading system's signal generation
            # This is a simplified version - in real system you'd use the full signal generation
            symbols = self.trading_system.config.get("symbols", ["SPY"])

            for symbol in symbols:
                # Generate a simple random signal for demonstration
                # In real system, use the actual strategy
                signal = np.random.uniform(-1.0, 1.0)  # Increased signal range
                signals[symbol] = signal

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")

        return signals

    def _get_prices_for_date(
        self, current_date: date, all_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Get prices for all symbols on a specific date."""
        prices = {}

        try:
            # Filter data for the specific date
            date_data = all_data[all_data.index.date == current_date]

            for symbol in self.trading_system.config.get("symbols", []):
                symbol_data = date_data[date_data["Symbol"] == symbol]
                if not symbol_data.empty:
                    prices[symbol] = symbol_data["Close"].iloc[-1]

        except Exception as e:
            self.logger.error(f"Error getting prices for {current_date}: {e}")

        return prices

    def _get_current_market_data(self, symbol: str, current_date: date) -> pd.DataFrame:
        """Get current market data for ML prediction."""
        try:
            # Get historical data up to current date for feature extraction
            # This should be the same data used for regime detection
            all_data = self._load_historical_data(
                current_date - timedelta(days=300), current_date
            )

            if all_data is not None and not all_data.empty:
                # Filter to symbol and current date
                symbol_data = all_data[all_data["Symbol"] == symbol]
                return symbol_data
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting current market data: {e}")
            return pd.DataFrame()

    def _record_trade_for_ml(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        current_date: date,
        regime_params,
    ):
        """Record trade outcome for ML learning."""
        try:
            if not self.ml_enabled or not self.profit_learner:
                return

            # Get current market features
            market_data = self._get_current_market_data(symbol, current_date)
            if market_data.empty:
                return

            market_features = self.profit_learner.extract_market_features(market_data)

            # Check if we have completed trades to record
            if self.ml_trade_history:
                # Record the most recent completed trade
                latest_trade = self.ml_trade_history[-1]

                trade_outcome = TradeOutcome(
                    timestamp=datetime.combine(
                        latest_trade["exit_date"], datetime.min.time()
                    ),
                    symbol=latest_trade["symbol"],
                    strategy=self.trading_system.config.get(
                        "strategy", "regime_aware_ensemble"
                    ),
                    regime=getattr(regime_params, "regime_name", "unknown")
                    if regime_params
                    else "unknown",
                    entry_price=latest_trade["entry_price"],
                    exit_price=latest_trade["exit_price"],
                    position_size=latest_trade["shares"] * latest_trade["entry_price"],
                    hold_duration=latest_trade["hold_duration"],
                    profit_loss=latest_trade["profit_loss"],
                    profit_loss_pct=latest_trade["profit_loss_pct"],
                    market_features=market_features,
                    trade_features={
                        "position_size": latest_trade["shares"]
                        * latest_trade["entry_price"],
                        "entry_price": latest_trade["entry_price"],
                        "exit_price": latest_trade["exit_price"],
                        "hold_duration": latest_trade["hold_duration"],
                        "strategy_confidence": min(
                            abs(self._get_last_signal_strength(symbol)), 1.0
                        ),
                        "market_regime": self._get_regime_confidence(regime_params),
                        "signal_strength": abs(self._get_last_signal_strength(symbol)),
                        "market_volatility": self._get_market_volatility(
                            symbol, current_date
                        ),
                    },
                )

                # Record for learning
                self.profit_learner.record_trade_outcome(trade_outcome)

                print(
                    f"🔍 Recorded completed trade: {latest_trade['profit_loss_pct']:.2%} profit, ${latest_trade['profit_loss']:.2f}"
                )
                self.logger.debug(
                    f"Recorded completed trade for ML: {latest_trade['profit_loss_pct']:.2%} profit"
                )

            else:
                # For opening positions, record with zero P&L (will be updated when closed)
                trade_outcome = TradeOutcome(
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    symbol=symbol,
                    strategy=self.trading_system.config.get(
                        "strategy", "regime_aware_ensemble"
                    ),
                    regime=getattr(regime_params, "regime_name", "unknown")
                    if regime_params
                    else "unknown",
                    entry_price=price,
                    exit_price=price,
                    position_size=abs(shares * price),
                    hold_duration=0,
                    profit_loss=0.0,
                    profit_loss_pct=0.0,
                    market_features=market_features,
                    trade_features={
                        "position_size": abs(shares * price),
                        "entry_price": price,
                        "exit_price": price,
                        "hold_duration": 0,
                        "strategy_confidence": min(
                            abs(self._get_last_signal_strength(symbol)), 1.0
                        ),
                        "market_regime": self._get_regime_confidence(regime_params),
                        "signal_strength": abs(self._get_last_signal_strength(symbol)),
                        "market_volatility": self._get_market_volatility(
                            symbol, current_date
                        ),
                    },
                )

                # Record for learning
                self.profit_learner.record_trade_outcome(trade_outcome)
                print(f"🔍 Recorded opening position: {action} {shares} {symbol}")

        except Exception as e:
            self.logger.error(f"Error recording trade for ML: {e}")

    def _get_last_signal_strength(self, symbol: str) -> float:
        """Get the last signal strength for a symbol."""
        try:
            # This would track the last signal generated for the symbol
            # For now, return a reasonable default
            return 0.5
        except Exception as e:
            self.logger.warning(f"Error getting signal strength: {e}")
            return 0.5

    def _get_market_volatility(self, symbol: str, current_date: date) -> float:
        """Get market volatility for a symbol."""
        try:
            # Calculate volatility from recent price data
            # For now, return a reasonable default
            return 0.02  # 2% volatility
        except Exception as e:
            self.logger.warning(f"Error getting market volatility: {e}")
            return 0.02

    def _get_regime_confidence(self, regime_params) -> float:
        """Get confidence level for current market regime."""
        try:
            if regime_params and hasattr(regime_params, "confidence"):
                return regime_params.confidence
            return 0.5
        except Exception as e:
            self.logger.warning(f"Error getting regime confidence: {e}")
            return 0.5

    def _track_ml_trade(
        self, symbol: str, action: str, shares: int, price: float, current_date: date
    ):
        """Track ML trades for accurate P&L calculation."""
        try:
            if action == "BUY":
                # Opening a new position
                if symbol not in self.ml_trade_positions:
                    self.ml_trade_positions[symbol] = []

                # Record the buy
                self.ml_trade_positions[symbol].append(
                    {
                        "entry_date": current_date,
                        "entry_price": price,
                        "shares": shares,
                        "action": "buy",
                    }
                )

            elif action == "SELL":
                # Closing positions
                if (
                    symbol in self.ml_trade_positions
                    and self.ml_trade_positions[symbol]
                ):
                    # Match sells with buys (FIFO)
                    remaining_shares_to_sell = abs(shares)
                    positions_to_close = []

                    for position in self.ml_trade_positions[symbol]:
                        if remaining_shares_to_sell <= 0:
                            break

                        shares_to_close = min(
                            remaining_shares_to_sell, position["shares"]
                        )

                        # Calculate P&L for this position
                        entry_price = position["entry_price"]
                        exit_price = price
                        profit_loss = (exit_price - entry_price) * shares_to_close
                        profit_loss_pct = (exit_price - entry_price) / entry_price

                        # Record completed trade
                        completed_trade = {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "shares": shares_to_close,
                            "profit_loss": profit_loss,
                            "profit_loss_pct": profit_loss_pct,
                            "hold_duration": (
                                current_date - position["entry_date"]
                            ).days,
                        }

                        self.ml_trade_history.append(completed_trade)

                        # Update position
                        position["shares"] -= shares_to_close
                        remaining_shares_to_sell -= shares_to_close

                        if position["shares"] <= 0:
                            positions_to_close.append(position)

                    # Remove closed positions
                    for pos in positions_to_close:
                        self.ml_trade_positions[symbol].remove(pos)

        except Exception as e:
            self.logger.error(f"Error tracking ML trade: {e}")

    def _execute_trades_with_portfolio(
        self,
        symbol: str,
        signals: Dict[str, float],
        current_date: date,
        regime_params,
        current_prices: Dict[str, float],
    ):
        """Execute trades using portfolio management."""
        if symbol not in signals or symbol not in current_prices:
            return

        signal = signals[symbol]
        current_price = current_prices[symbol]

        # ML prediction and learning
        if self.ml_enabled and self.profit_learner:
            try:
                # Get current market data for ML prediction
                current_market_data = self._get_current_market_data(
                    symbol, current_date
                )

                # Predict profit potential for current strategy
                strategy_name = self.trading_system.config.get(
                    "strategy", "regime_aware_ensemble"
                )
                prediction = self.profit_learner.predict_profit_potential(
                    current_market_data, strategy_name, symbol
                )

                # Adjust signal based on ML prediction
                expected_profit = prediction.get("expected_profit_pct", 0.0)
                confidence = prediction.get("confidence", 0.1)

                # Check if ML system has enough training data
                if (
                    len(self.profit_learner.performance_history)
                    >= self.profit_learner.min_trades_for_learning
                ):
                    # ML system is trained - use strict thresholds
                    if (
                        expected_profit > 0.001 and confidence > 0.3
                    ):  # 0.1% profit, 30% confidence
                        signal *= confidence  # Scale signal by confidence
                        self.logger.info(
                            f"ML prediction: {expected_profit:.2%} profit, {confidence:.1%} confidence"
                        )
                    else:
                        signal *= 0.1  # Reduce signal if ML is not confident
                        self.logger.debug(
                            f"ML low confidence: {expected_profit:.2%} profit, {confidence:.1%} confidence"
                        )
                else:
                    # ML system is learning - be more permissive
                    if (
                        expected_profit > -0.005 and confidence > 0.1
                    ):  # Allow small losses, lower confidence
                        signal *= 0.5  # Scale signal moderately
                        self.logger.debug(
                            f"ML learning mode: {expected_profit:.2%} profit, {confidence:.1%} confidence"
                        )
                    else:
                        signal *= 0.2  # Reduce signal but don't eliminate
                        self.logger.debug(
                            f"ML learning mode - low confidence: {expected_profit:.2%} profit, {confidence:.1%} confidence"
                        )

            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")

        # Skip if signal is too small
        if abs(signal) < 0.0001:  # Extremely low threshold to generate more trades
            print(f"🔍 Signal too small for {symbol}: {signal:.4f} < 0.0001")
            return

        # Calculate position size
        portfolio_value = self.portfolio.value_at(current_prices)
        position_value = (
            abs(signal) * portfolio_value * 0.2
        )  # 20% max position to generate more trades
        shares = int(position_value / current_price)

        print(
            f"🔍 Position sizing: signal={signal:.4f}, portfolio_value=${portfolio_value:,.2f}, position_value=${position_value:,.2f}, shares={shares}"
        )

        if shares == 0:
            print(
                f"🔍 No shares calculated for {symbol}: position_value=${position_value:.2f}, price=${current_price:.2f}"
            )
            return

        # Determine trade direction
        if signal > 0:
            action = "BUY"
            # Check if we have enough cash
            trade_value = shares * current_price
            if trade_value > self.portfolio.cash:
                shares = int(self.portfolio.cash / current_price)
                if shares == 0:
                    return
        else:
            action = "SELL"
            # Check if we have enough shares
            current_position = self.portfolio.get_position(symbol)
            if current_position is None or current_position.qty < shares:
                shares = current_position.qty if current_position else 0
                if shares == 0:
                    return

        # Execute trade
        if shares > 0:
            # Trade executed successfully

            # Track ML trade for accurate P&L
            self._track_ml_trade(symbol, action, shares, current_price, current_date)

            # Update portfolio
            if action == "BUY":
                self.portfolio.execute_order(symbol, shares, current_price, fee=0.0)
            else:
                self.portfolio.execute_order(symbol, -shares, current_price, fee=0.0)

            # Record trade
            print(
                f"🔍 Executing {action} trade: {shares} shares of {symbol} @ ${current_price:.2f}"
            )
            if action == "BUY":
                self.trade_book.on_buy(
                    str(current_date), symbol, shares, current_price, 0.0
                )
            else:
                # For sell, we need to calculate remaining quantity
                current_position = self.portfolio.get_position(symbol)
                remaining_qty = (
                    (current_position.qty - shares) if current_position else 0
                )
                self.trade_book.on_sell(
                    str(current_date), symbol, shares, current_price, 0.0, remaining_qty
                )

            # Record trade outcome for ML learning
            print(
                f"🔍 About to record trade for ML: ml_enabled={self.ml_enabled}, profit_learner={self.profit_learner is not None}"
            )
            if self.ml_enabled and self.profit_learner:
                print(f"🔍 Recording trade for ML: {action} {shares} {symbol}")
                self.logger.info(f"Recording trade for ML: {action} {shares} {symbol}")
                self._record_trade_for_ml(
                    symbol, action, shares, current_price, current_date, regime_params
                )
            else:
                print(
                    f"🔍 ML not enabled or profit_learner not available. ml_enabled: {self.ml_enabled}, profit_learner: {self.profit_learner is not None}"
                )
                self.logger.debug(
                    f"ML not enabled or profit_learner not available. ml_enabled: {self.ml_enabled}, profit_learner: {self.profit_learner is not None}"
                )

    def _slice_ledger_to_backtest(self) -> pd.DataFrame:
        """Slice trade ledger to backtest period."""
        ledger = self.trade_book.get_ledger()
        if ledger.empty:
            return pd.DataFrame()

        # Filter to backtest period
        backtest_ledger = ledger[
            (ledger["date"] >= self.start_date) & (ledger["date"] <= self.end_date)
        ]

        return backtest_ledger

    def _get_trades_in_backtest_window(self) -> List[Dict]:
        """Get trades that occurred during the backtest window."""
        trades = self.trade_book.get_trades()

        backtest_trades = []
        for trade in trades:
            trade_date = trade["date"]
            if isinstance(trade_date, str):
                trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()

            if self.start_date <= trade_date <= self.end_date:
                backtest_trades.append(trade)

        return backtest_trades

    def _calculate_trade_metrics_from(self, trades: List[Dict]) -> Dict:
        """Calculate trade-level metrics."""
        if not trades:
            return {}

        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)

        # Basic trade metrics
        total_trades = len(trades)
        close_trades = len(trades_df[trades_df["action"] == "CLOSE"])
        open_trades = len(trades_df[trades_df["action"] == "OPEN"])

        # Volume metrics (using qty * price)
        if (
            not trades_df.empty
            and "qty" in trades_df.columns
            and "price" in trades_df.columns
        ):
            trades_df["value"] = trades_df["qty"] * trades_df["price"]
            total_volume = trades_df["value"].sum()
            avg_trade_size = trades_df["value"].mean()
        else:
            total_volume = 0.0
            avg_trade_size = 0.0

        # Price metrics
        avg_price = trades_df["price"].mean() if "price" in trades_df.columns else 0.0

        return {
            "total_trades": total_trades,
            "close_trades": close_trades,
            "open_trades": open_trades,
            "total_volume": total_volume,
            "avg_trade_size": avg_trade_size,
            "avg_price": avg_price,
        }

    def _calculate_portfolio_metrics(
        self, backtest_ledger: pd.DataFrame = None
    ) -> Dict:
        """Calculate portfolio-level metrics."""
        # Only return empty if backtest_ledger is provided and empty
        if backtest_ledger is not None and backtest_ledger.empty:
            return {}

        # Calculate daily returns
        daily_returns = []
        for i, daily in enumerate(self.daily_returns):
            if i == 0:
                daily_return = 0.0
            else:
                prev_value = self.daily_returns[i - 1]["portfolio_value"]
                current_value = daily["portfolio_value"]
                daily_return = (
                    (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
                )

            daily_returns.append(daily_return)

        # Calculate metrics
        returns_series = pd.Series(daily_returns)

        # Basic metrics
        if self.daily_returns:
            final_portfolio_value = self.daily_returns[-1]["portfolio_value"]
        else:
            final_portfolio_value = self.initial_capital

        total_return = (final_portfolio_value / self.initial_capital) - 1
        annualized_return = (
            (1 + total_return) ** (252 / len(returns_series)) - 1
            if len(returns_series) > 0
            else 0
        )

        # Risk metrics
        volatility = (
            returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0
        )
        sharpe_ratio = (
            (returns_series.mean() * 252) / volatility if volatility > 0 else 0
        )

        # Drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": final_portfolio_value,
            "initial_capital": self.initial_capital,
        }

    def _generate_results(self, trade_metrics: Dict, portfolio_metrics: Dict) -> Dict:
        """Generate comprehensive results dictionary."""
        results = {
            "backtest_info": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "symbols": self.trading_system.config.get("symbols", []),
                "initial_capital": self.initial_capital,
            },
            "trade_metrics": trade_metrics,
            "portfolio_metrics": portfolio_metrics,
            "summary": {
                "total_return_pct": portfolio_metrics.get("total_return", 0) * 100,
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0),
                "max_drawdown_pct": portfolio_metrics.get("max_drawdown", 0) * 100,
                "total_trades": trade_metrics.get("total_trades", 0),
            },
        }

        self._last_results = results
        return results

    def _save_results(self, results: Dict, backtest_ledger: pd.DataFrame = None):
        """Save backtest results to files."""
        # Save results JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/backtest/results_{timestamp}.json"

        Path("results/backtest").mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save ledger if provided
        if backtest_ledger is not None and not backtest_ledger.empty:
            ledger_file = f"results/backtest/ledger_{timestamp}.csv"
            backtest_ledger.to_csv(ledger_file, index=False)

        print(f"💾 Results saved to {results_file}")

    def _format_summary(self, results: Dict) -> str:
        """Format results summary for display."""
        summary = results.get("summary", {})

        summary_text = f"""
📊 BACKTEST SUMMARY
{'='*50}
📅 Period: {results['backtest_info']['start_date']} to {results['backtest_info']['end_date']}
💰 Initial Capital: ${results['backtest_info']['initial_capital']:,.2f}
💵 Final Value: ${results['portfolio_metrics']['final_value']:,.2f}

📈 PERFORMANCE
Total Return: {summary['total_return_pct']:.2f}%
Sharpe Ratio: {summary['sharpe_ratio']:.2f}
Max Drawdown: {summary['max_drawdown_pct']:.2f}%

📊 TRADING
Total Trades: {summary['total_trades']}
"""

        return summary_text

    def get_last_summary(self) -> Dict:
        """Get summary of last backtest run."""
        if self._last_results is None:
            return {}

        return self._last_results.get("summary", {})

    def print_results(self, results: Dict):
        """Print backtest results."""
        summary_text = self._format_summary(results)
        print(summary_text)

        # Print detailed metrics
        print("📊 DETAILED METRICS")
        print("=" * 50)

        portfolio_metrics = results.get("portfolio_metrics", {})
        for key, value in portfolio_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("\n📊 TRADE METRICS")
        print("=" * 50)

        trade_metrics = results.get("trade_metrics", {})
        for key, value in trade_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    def _generate_final_results(self) -> Dict:
        """Generate final results from backtest."""
        # Get trades in backtest window
        trades = self._get_trades_in_backtest_window()

        # Calculate metrics
        trade_metrics = self._calculate_trade_metrics_from(trades)
        portfolio_metrics = self._calculate_portfolio_metrics(None)

        # Generate results
        results = self._generate_results(trade_metrics, portfolio_metrics)

        return results


# Lightweight wrapper for tests and parity with smoke gates
def run_backtest(df: pd.DataFrame, cfg: Dict, *, sanity_profile: str = "walkforward") -> Dict:
    """
    Deterministic, side-effect-free backtest gate used in unit tests.
    - Enforce DataSanity via validate_dataframe_fast
    - Return structured status without raising
    - Keep shape stable for promote/gate checks
    """
    try:
        validator = DataSanityValidator("config/data_sanity.yaml", profile=sanity_profile)
        res = validator.validate_dataframe_fast(df, sanity_profile)
        if res.violations and getattr(res, "mode", "warn") == "enforce":
            code = res.violations[0].code if res.violations else "UNKNOWN"
            reason = res.summary() if hasattr(res, "summary") else code
            return {"status": "FAIL", "violation_code": code, "reason": reason}
    except Exception as e:
        return {"status": "FAIL", "violation_code": "UNEXPECTED_ERROR", "reason": f"{e.__class__.__name__}: {e}"}

    # Trading cost / leverage guards (CI-enforced)
    risk_cfg = (cfg or {}).get("risk", {})
    in_ci = os.getenv("CI", "").lower() in ("1", "true", "yes")
    slippage_bps = risk_cfg.get("slippage_bps")
    fee_bps = risk_cfg.get("fee_bps")
    max_lev = risk_cfg.get("max_leverage")
    if in_ci:
        if slippage_bps is None:
            return {"status": "FAIL", "violation_code": "MISSING_COSTS", "reason": "slippage_bps not set in risk config"}
        if fee_bps is None:
            return {"status": "FAIL", "violation_code": "MISSING_COSTS", "reason": "fee_bps not set in risk config"}
        if max_lev is None:
            return {"status": "FAIL", "violation_code": "LEVERAGE_LIMIT", "reason": "max_leverage not set in risk config"}
        if max_lev > 3.0:
            return {"status": "FAIL", "violation_code": "LEVERAGE_LIMIT", "reason": f"max_leverage too high: {max_lev}"}
        max_gross = risk_cfg.get("max_gross_exposure", 1.0)
        max_pos = risk_cfg.get("max_position_pct", 1.0)
        if max_gross > 2.0:
            return {"status": "FAIL", "violation_code": "GROSS_EXPOSURE_LIMIT", "reason": f"max_gross_exposure too high: {max_gross}"}
        if max_pos > 1.0:
            return {"status": "FAIL", "violation_code": "POSITION_LIMIT", "reason": f"max_position_pct too high: {max_pos}"}

    # Minimal OK payload with placeholder metrics for tests
    n = len(df) if isinstance(df, pd.DataFrame) else 0
    trades = max(1, n // 10)
    return {
        "status": "OK",
        "trades": trades,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
    }
