"""
Minimum Viable Bot Runner
Main orchestrator for the MVB system
"""

import logging
import time
from datetime import datetime
from typing import Any

from core.factory import make_mvb_components
from core.strategy import create_strategy


class MVBRunner:
    """Main MVB runner that orchestrates all components"""

    def __init__(self, mode: str, config: dict[str, Any]):
        self.mode = mode
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create components
        self.components = make_mvb_components(mode, config)

        # Create strategy
        strategy_type = config.get("strategy", "random")
        self.strategy = create_strategy(strategy_type, config.get("strategy_config", {}))

        # Trading state
        self.trading_enabled = True
        self.session_start = datetime.now()
        self.session_duration = config.get("session_duration", 30)  # minutes
        self.heartbeat_interval = config.get("heartbeat_interval", 30)  # seconds
        self.last_heartbeat = datetime.now()

        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.reject_count = 0
        self.total_orders = 0

    def run(self):
        """Main run loop"""
        self.logger.info(f"Starting MVB in {self.mode} mode")
        self.logger.info(f"Session duration: {self.session_duration} minutes")

        # Log startup event
        self.components["telemetry"].log_event(
            "startup",
            {
                "mode": self.mode,
                "config": self.config,
                "session_start": self.session_start.isoformat(),
            },
        )

        try:
            while self.trading_enabled and not self._should_stop():
                # Check market hours
                if not self.components["clock"].is_market_open():
                    self.logger.info("Market closed, waiting...")
                    time.sleep(60)
                    continue

                # Check data feed
                if not self.components["feed"].is_connected():
                    self.logger.warning("Data feed disconnected")
                    self._handle_data_gap()
                    continue

                # Get market data
                symbols = self.config.get("symbols", ["SPY"])
                market_data = self.components["feed"].get_latest_data(symbols)

                if not market_data:
                    self.logger.warning("No market data received")
                    continue

                # Create context
                ctx = self._create_context(market_data)

                # Generate signals
                signals = self.strategy.on_bar(ctx, market_data)

                # Process signals
                self._process_signals(signals, market_data)

                # Update portfolio
                self._update_portfolio(market_data)

                # Check risk limits
                if not self._check_risk_limits():
                    self.logger.warning("Risk limits exceeded, stopping trading")
                    self.trading_enabled = False
                    break

                # Heartbeat
                self._heartbeat()

                # Sleep
                time.sleep(self.config.get("bar_interval", 60))  # seconds

        except KeyboardInterrupt:
            self.logger.info("Received interrupt, shutting down gracefully")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.components["telemetry"].log_event("error", {"error": str(e)})
        finally:
            self._shutdown()

    def _should_stop(self) -> bool:
        """Check if we should stop the session"""
        elapsed = datetime.now() - self.session_start
        return elapsed.total_seconds() > (self.session_duration * 60)

    def _create_context(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Create context for strategy"""
        positions = self.components["broker"].get_positions()
        portfolio_value = self.components["portfolio"].get_total_value(
            {symbol: data["price"] for symbol, data in market_data.items()}
        )

        return {
            "positions": positions,
            "portfolio_value": portfolio_value,
            "cash": self.components["portfolio"].cash,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "mode": self.mode,
        }

    def _process_signals(self, signals: dict[str, float], market_data: dict[str, Any]):
        """Process trading signals"""
        for symbol, signal in signals.items():
            if abs(signal) < 0.1:  # Ignore small signals
                continue

            # Calculate position size
            price = market_data[symbol]["price"]
            portfolio_value = self.components["portfolio"].get_total_value(
                {symbol: price for symbol in market_data}
            )

            # Position sizing
            max_position_pct = self.config.get("max_position_pct", 0.1)
            position_value = abs(signal) * portfolio_value * max_position_pct
            quantity = int(position_value / price)

            if quantity == 0:
                continue

            # Create order
            order = {
                "symbol": symbol,
                "side": "BUY" if signal > 0 else "SELL",
                "quantity": quantity,
                "price": price,
                "order_type": "MARKET",
            }

            # Apply risk checks
            order, reason = self.components["risk"].validate_trade(
                order,
                self.components["portfolio"].cash,
                self.components["broker"].get_positions(),
            )

            if reason:
                self.logger.warning(f"Order rejected: {reason}")
                self.reject_count += 1
                self.components["telemetry"].log_event(
                    "order_rejected",
                    {"symbol": symbol, "reason": reason, "order": order},
                )
                continue

            # Submit order
            try:
                order_id = self.components["broker"].submit_order(order)
                self.total_orders += 1

                self.logger.info(f"Order submitted: {order}")
                self.components["telemetry"].log_event(
                    "order_submitted", {"order_id": order_id, "order": order}
                )

                # Update portfolio
                self.components["portfolio"].update_position(symbol, quantity, price, order["side"])

            except Exception as e:
                self.logger.error(f"Error submitting order: {e}")
                self.components["telemetry"].log_event(
                    "order_error", {"error": str(e), "order": order}
                )

    def _update_portfolio(self, market_data: dict[str, Any]):
        """Update portfolio with current market data"""
        # Calculate current P&L
        current_value = self.components["portfolio"].get_total_value(
            {symbol: data["price"] for symbol, data in market_data.items()}
        )
        initial_capital = self.config.get("initial_capital", 100000)

        # Update daily P&L (simplified)
        self.daily_pnl = current_value - initial_capital

        # Log portfolio snapshot
        self.components["telemetry"].log_event(
            "portfolio_snapshot",
            {
                "total_value": current_value,
                "cash": self.components["portfolio"].cash,
                "positions": self.components["broker"].get_positions(),
                "daily_pnl": self.daily_pnl,
            },
        )

    def _check_risk_limits(self) -> bool:
        """Check risk limits and kill switches"""
        # Check daily loss limit
        daily_loss_limit = self.config.get("daily_loss_limit", 0.01)  # 1%
        initial_capital = self.config.get("initial_capital", 100000)

        if self.daily_pnl < -(initial_capital * daily_loss_limit):
            self.logger.error(f"Daily loss limit exceeded: {self.daily_pnl}")
            self.components["telemetry"].log_event(
                "risk_limit_exceeded",
                {"limit": "daily_loss", "daily_pnl": self.daily_pnl},
            )
            return False

        # Check reject rate
        if self.total_orders > 0:
            reject_rate = self.reject_count / self.total_orders
            if reject_rate > 0.5:  # 50% reject rate
                self.logger.error(f"Reject rate too high: {reject_rate:.2%}")
                self.components["telemetry"].log_event(
                    "risk_limit_exceeded",
                    {"limit": "reject_rate", "reject_rate": reject_rate},
                )
                return False

        # Check heartbeat
        if (datetime.now() - self.last_heartbeat).total_seconds() > 60:
            self.logger.error("Heartbeat missed")
            self.components["telemetry"].log_event(
                "risk_limit_exceeded", {"limit": "heartbeat_missed"}
            )
            return False

        return True

    def _handle_data_gap(self):
        """Handle data feed gaps"""
        self.logger.warning("Data gap detected, halting new orders")
        self.components["telemetry"].log_event(
            "data_gap", {"timestamp": datetime.now().isoformat()}
        )

        # Wait for data to resume
        time.sleep(30)

    def _heartbeat(self):
        """Send heartbeat"""
        self.last_heartbeat = datetime.now()
        self.components["telemetry"].log_event(
            "heartbeat", {"timestamp": self.last_heartbeat.isoformat()}
        )

    def _shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down MVB")

        # Final reconciliation
        if self.components["portfolio"].reconcile():
            self.logger.info("Portfolio reconciliation passed")
        else:
            self.logger.error("Portfolio reconciliation failed")

        # Log shutdown event
        session_duration = datetime.now() - self.session_start
        self.components["telemetry"].log_event(
            "shutdown",
            {
                "session_duration": session_duration.total_seconds(),
                "total_orders": self.total_orders,
                "reject_count": self.reject_count,
                "final_pnl": self.daily_pnl,
            },
        )

        self.logger.info(f"Session completed: {session_duration}")
        self.logger.info(f"Total orders: {self.total_orders}")
        self.logger.info(f"Reject rate: {self.reject_count / max(self.total_orders, 1):.2%}")
        self.logger.info(f"Final P&L: ${self.daily_pnl:.2f}")


def run_mvb(mode: str, config: dict[str, Any]):
    """Convenience function to run MVB"""
    runner = MVBRunner(mode, config)
    runner.run()
