#!/usr/bin/env python3
"""
Trading system guardrails to prevent invalid operations.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TradingGuardrails:
    """Guardrails to prevent invalid trading operations."""

    def __init__(self, config: dict):
        self.config = config
        self.max_trades_per_symbol = 1  # One order per symbol per bar
        self.cooldown_minutes = 30  # Cooldown on reversals
        self.last_trade_time = {}  # Track last trade time per symbol
        self.trade_count = {}  # Track trades per symbol per day

    def validate_trade(self, symbol: str, action: str, size: float, price: float) -> bool:
        """Validate if a trade should be executed."""

        # Check if we have enough data
        if not self._has_sufficient_data(symbol):
            logger.warning(f"Insufficient data for {symbol}")
            return False

        # Check trade frequency limits
        if not self._check_trade_frequency(symbol):
            logger.warning(f"Trade frequency limit exceeded for {symbol}")
            return False

        # Check position limits
        if not self._check_position_limits(symbol, action, size):
            logger.warning(f"Position limit exceeded for {symbol}")
            return False

        # Check price sanity
        if not self._check_price_sanity(symbol, price):
            logger.warning(f"Price sanity check failed for {symbol}: {price}")
            return False

        # Check for reversal cooldown
        if not self._check_reversal_cooldown(symbol, action):
            logger.warning(f"Reversal cooldown active for {symbol}")
            return False

        return True

    def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if we have sufficient data for the symbol."""
        # This would check if we have enough historical data
        # For now, return True
        return True

    def _check_trade_frequency(self, symbol: str) -> bool:
        """Check if trade frequency limits are respected."""
        today = datetime.now().date()
        key = f"{symbol}_{today}"

        if key not in self.trade_count:
            self.trade_count[key] = 0

        return not self.trade_count[key] >= self.max_trades_per_symbol

    def _check_position_limits(self, symbol: str, action: str, size: float) -> bool:
        """Check if position limits are respected."""
        max_weight = self.config.get("max_weight_per_symbol", 0.25)
        initial_capital = self.config.get("initial_capital", 100000)
        initial_capital * max_weight

        # For now, just check if size is reasonable
        if size <= 0:
            return False

        # Check if position value exceeds limits
        # This would need current price to be accurate
        return True

    def _check_price_sanity(self, symbol: str, price: float) -> bool:
        """Check if price is within reasonable bounds."""
        if price <= 0:
            return False

        # Add symbol-specific price bounds
        price_bounds = {
            "SPY": (100, 1000),
            "AAPL": (50, 500),
            "NVDA": (100, 1000),
            "GOOGL": (50, 500),
            "MSFT": (50, 500),
        }

        if symbol in price_bounds:
            min_price, max_price = price_bounds[symbol]
            if price < min_price or price > max_price:
                return False

        return True

    def _check_reversal_cooldown(self, symbol: str, action: str) -> bool:
        """Check if reversal cooldown is active."""
        if symbol not in self.last_trade_time:
            return True

        last_time = self.last_trade_time[symbol]
        return not datetime.now() - last_time < timedelta(minutes=self.cooldown_minutes)

    def record_trade(self, symbol: str, action: str, size: float, price: float):
        """Record a trade for guardrail tracking."""
        today = datetime.now().date()
        key = f"{symbol}_{today}"

        # Update trade count
        if key not in self.trade_count:
            self.trade_count[key] = 0
        self.trade_count[key] += 1

        # Update last trade time
        self.last_trade_time[symbol] = datetime.now()

        logger.info(f"Trade recorded: {symbol} {action} {size} @ {price}")


class ConfigValidator:
    """Validate configuration files."""

    @staticmethod
    def validate_trading_config(config: dict) -> bool:
        """Validate trading configuration."""
        required_fields = ["initial_capital", "symbols"]

        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False

        # Check risk_params
        if "risk_params" not in config:
            logger.error("Missing risk_params section")
            return False

        risk_params = config["risk_params"]
        required_risk_fields = [
            "max_weight_per_symbol",
            "max_drawdown",
            "max_daily_loss",
        ]

        for field in required_risk_fields:
            if field not in risk_params:
                logger.error(f"Missing required risk field: {field}")
                return False

        # Validate numeric ranges
        if config["initial_capital"] <= 0:
            logger.error("Initial capital must be positive")
            return False

        if not (0 < risk_params["max_weight_per_symbol"] <= 1):
            logger.error("max_weight_per_symbol must be between 0 and 1")
            return False

        if not (0 < risk_params["max_drawdown"] <= 1):
            logger.error("max_drawdown must be between 0 and 1")
            return False

        return True

    @staticmethod
    def validate_ibkr_config(config: dict) -> bool:
        """Validate IBKR configuration."""
        if "ibkr_config" not in config:
            logger.error("Missing IBKR configuration")
            return False

        ibkr_config = config["ibkr_config"]
        required_fields = ["host", "port", "client_id"]

        for field in required_fields:
            if field not in ibkr_config:
                logger.error(f"Missing IBKR field: {field}")
                return False

        return True


class SystemHealthChecker:
    """Check system health before trading."""

    @staticmethod
    def check_ibkr_connection() -> bool:
        """Check if IBKR connection is available."""
        try:
            from brokers.ibkr_broker import IBKRBroker, IBKRConfig

            config = IBKRConfig()
            broker = IBKRBroker(config=config, auto_connect=False)

            if broker.connect():
                broker.disconnect()
                return True
            logger.warning("IBKR connection failed")
            return False

        except Exception as e:
            logger.error(f"IBKR health check failed: {e}")
            return False

    @staticmethod
    def check_data_providers() -> bool:
        """Check if data providers are working."""
        try:
            import yfinance as yf

            # Test yfinance
            ticker = yf.Ticker("SPY")
            data = ticker.history(period="1d")

            if len(data) == 0:
                logger.warning("yfinance data provider not working")
                return False

            return True

        except Exception as e:
            logger.error(f"Data provider health check failed: {e}")
            return False

    @staticmethod
    def check_file_permissions() -> bool:
        """Check if required directories are writable."""
        required_dirs = ["logs", "results", "data"]

        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except Exception as e:
                    logger.error(f"Cannot create directory {dir_name}: {e}")
                    return False

            # Check if writable
            test_file = os.path.join(dir_name, "test_write.tmp")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                logger.error(f"Directory {dir_name} not writable: {e}")
                return False

        return True


def main():
    """Run guardrail checks."""
    print("🔒 Trading System Guardrails")
    print("=" * 50)

    # Load configuration
    try:
        with open("config/enhanced_paper_trading_config.json") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1

    # Validate configuration
    validator = ConfigValidator()
    if not validator.validate_trading_config(config):
        print("❌ Trading configuration validation failed")
        return 1

    if not validator.validate_ibkr_config(config):
        print("❌ IBKR configuration validation failed")
        return 1

    print("✅ Configuration validation passed")

    # Check system health
    health_checker = SystemHealthChecker()

    if not health_checker.check_file_permissions():
        print("❌ File permissions check failed")
        return 1

    print("✅ File permissions check passed")

    if not health_checker.check_data_providers():
        print("❌ Data providers check failed")
        return 1

    print("✅ Data providers check passed")

    # IBKR connection is optional for paper trading
    if config.get("use_ibkr", False):
        if not health_checker.check_ibkr_connection():
            print("⚠️  IBKR connection check failed (optional for paper trading)")
        else:
            print("✅ IBKR connection check passed")

    print("\n🎉 All guardrail checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
