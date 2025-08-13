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

        # Data storage
        self.daily_returns = []
        self.equity_curve = []

        # Logging control
        self.insufficient_data_logged = set()  # Rate limit insufficient data logs
        self.logger = logging.getLogger(__name__)

        # Constants
        self.MIN_HISTORY = (
            252  # Minimum trading days for warmup (matches regime detector)
        )

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
        warmup_days = 200 if self.fast_mode else 600
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
                f"🔥 Warmup complete. Portfolio value: ${self.portfolio.total_value:,.2f}"
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
            daily_return = {
                "date": backtest_date,
                "portfolio_value": self.portfolio.total_value,
                "cash": self.portfolio.cash,
                "positions_value": self.portfolio.total_value - self.portfolio.cash,
                "return": 0.0,  # Will be calculated later
            }
            self.daily_returns.append(daily_return)

        # Calculate final results
        print("📊 Calculating results...")
        results = self._generate_final_results()

        # Save results
        self._save_results(results)

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
            if current_date not in self.insufficient_data_logged:
                self.logger.warning(
                    f"Insufficient data for {current_date}: {len(data_up_to_date)} < {self.MIN_HISTORY}"
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

        for symbol in symbols:
            try:
                # Use yfinance for historical data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date, end=end_date + timedelta(days=1)
                )

                if not data.empty:
                    # Add symbol column
                    data["Symbol"] = symbol
                    all_data.append(data)
                    print(f"✅ Loaded {len(data)} data points for {symbol}")
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
        trading_dates = sorted(data.index.date.unique())

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

        for symbol, position in self.portfolio.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                position.market_value = position.quantity * position.current_price

    def _run_daily_trading(
        self, current_date: date, data: pd.DataFrame, current_prices: Dict[str, float]
    ):
        """Run daily trading cycle."""
        # Get data up to current date for regime detection
        data_up_to_date = self._get_data_up_to_date(data, current_date)

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
                self._execute_trades_with_portfolio(
                    symbol, signals, current_date, regime_params, current_prices
                )

    def _detect_regime_with_rate_limit(self, data: pd.DataFrame):
        """Detect regime with rate limiting for performance."""
        try:
            # Use the trading system's regime detector
            regime = self.trading_system.regime_detector.detect_regime(data)
            regime_name = self.trading_system.regime_detector.get_current_regime_name()

            # Create regime params object
            class RegimeParams:
                def __init__(self, name, confidence):
                    self.regime_name = name
                    self.confidence_threshold = confidence
                    self.position_sizing_multiplier = 1.0

            regime_params = RegimeParams(regime_name, 0.7)

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
                signal = np.random.uniform(-0.5, 0.5)
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

        # Skip if signal is too small
        if abs(signal) < 0.1:
            return

        # Calculate position size
        position_value = (
            abs(signal) * self.portfolio.total_value * 0.1
        )  # 10% max position
        shares = int(position_value / current_price)

        if shares == 0:
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
            if current_position is None or current_position.quantity < shares:
                shares = current_position.quantity if current_position else 0
                if shares == 0:
                    return

        # Execute trade
        if shares > 0:
            trade = {
                "date": current_date,
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "price": current_price,
                "value": shares * current_price,
                "regime": regime_params.regime_name if regime_params else "unknown",
            }

            # Update portfolio
            if action == "BUY":
                self.portfolio.buy(symbol, shares, current_price)
            else:
                self.portfolio.sell(symbol, shares, current_price)

            # Record trade
            self.trade_book.record_trade(trade)

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
        buy_trades = len(trades_df[trades_df["action"] == "BUY"])
        sell_trades = len(trades_df[trades_df["action"] == "SELL"])

        # Volume metrics
        total_volume = trades_df["value"].sum()
        avg_trade_size = trades_df["value"].mean()

        # Price metrics
        avg_price = trades_df["price"].mean()

        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_volume": total_volume,
            "avg_trade_size": avg_trade_size,
            "avg_price": avg_price,
        }

    def _calculate_portfolio_metrics(self, backtest_ledger: pd.DataFrame) -> Dict:
        """Calculate portfolio-level metrics."""
        if backtest_ledger.empty:
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
        total_return = (self.portfolio.total_value / self.initial_capital) - 1
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
            "final_value": self.portfolio.total_value,
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
        portfolio_metrics = self._calculate_portfolio_metrics(pd.DataFrame())

        # Generate results
        results = self._generate_results(trade_metrics, portfolio_metrics)

        return results
