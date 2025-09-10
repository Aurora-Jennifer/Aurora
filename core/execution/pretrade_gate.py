"""
Pre-trade Safety Gate

Comprehensive safety checks before any order submission.
Implements all oh-shit guardrails to prevent bad trades.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum

try:
    # Optional Alpaca data client imports for price sanity
    from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest  # type: ignore
except Exception:  # pragma: no cover
    StockHistoricalDataClient = None  # type: ignore
    StockLatestQuoteRequest = None  # type: ignore
    StockLatestTradeRequest = None  # type: ignore

logger = logging.getLogger(__name__)


def _is_reducer(side: str, current_pos: int) -> bool:
    """Determine if an order is a reducer (closing position) vs opener (new exposure)."""
    # Selling a long or buying to cover a short reduces exposure
    return (side == "sell" and current_pos > 0) or (side == "buy" and current_pos < 0)


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def as_utc(dt):
    """Convert datetime to UTC timezone."""
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        # naive → assume it was intended as UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class GateResult(Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


@dataclass
class GateDecision:
    result: GateResult
    reason: str
    shares: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class PreTradeGate:
    """
    Pre-trade safety gate that runs before any order submission.
    
    Implements comprehensive safety checks:
    - Idempotency/duplicates
    - Signal freshness
    - Market state
    - Price sanity
    - Position constraints
    - Sizing caps
    - Order throttling
    """
    
    def __init__(self, config: Dict[str, Any], trading_client=None, data_client=None):
        self.config = config
        self.risk_config = config.get('risk_management', {})
        
        # Optional clients for richer checks
        self.trading_client = trading_client
        self.data_client = data_client
        
        # Track daily order counts
        self.daily_orders = {}  # {symbol: count}
        self.daily_total = 0
        self.last_reset_date = datetime.now().date()
    
    def _reset_daily_counts_if_needed(self):
        """Reset daily order counts if it's a new day."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_orders = {}
            self.daily_total = 0
            self.last_reset_date = today
            logger.info("Reset daily order counts for new day")

    def _get_price_mid(self, symbol: str, ref_price: Optional[float]) -> Optional[float]:
        """Best-effort: NBBO mid -> last trade -> ref_price."""
        # Try NBBO mid via data client
        try:
            if self.data_client and StockLatestQuoteRequest:
                q = self.data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol=symbol))
                bid = float(getattr(q, 'bid_price', 0) or 0)
                ask = float(getattr(q, 'ask_price', 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
        except Exception as e:
            logger.debug(f"Quote fetch failed for {symbol}: {e}")
        # Try last trade via data client
        try:
            if self.data_client and StockLatestTradeRequest:
                t = self.data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol=symbol))
                price = float(getattr(t, 'price', 0) or 0)
                if price > 0:
                    return price
        except Exception as e:
            logger.debug(f"Trade fetch failed for {symbol}: {e}")
        # Fallback
        return ref_price
    
    def check_order(
        self,
        signal_id: str,
        symbol: str,
        side: str,
        qty_target: int,
        px_ref: float,
        current_position: int,
        broker_client,
        signal_timestamp: Optional[datetime] = None
    ) -> GateDecision:
        """
        Run comprehensive pre-trade safety checks.
        """
        now = utc_now()
        
        # Debug: log gate inputs
        reducer = _is_reducer(side, current_position)
        intent = "reduce" if reducer else "open"
        logger.info(f"GateIn {symbol} qty={qty_target} ref_px={px_ref:.2f} side={side} current_pos={current_position} intent={intent}")
        
        # Guardrail: check for zero quantity before doing anything
        if not isinstance(qty_target, int) or qty_target == 0:
            return GateDecision(
                result=GateResult.REJECT,
                reason="SIZE_ZERO_BEFORE_GATE",
                metadata={"qty_target": qty_target, "symbol": symbol}
            )
        
        # Reset daily counts if needed
        self._reset_daily_counts_if_needed()
        
        try:
            # Get account and market data
            account = broker_client.get_account()
            clock = broker_client.get_clock()
            
            # A. Idempotency / duplicates check
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                
                req = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol],
                    limit=200
                )
                open_orders = broker_client.get_orders(filter=req)
                if any(order.client_order_id == signal_id for order in open_orders):
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="DUPLICATE_ORDER",
                        metadata={"signal_id": signal_id, "symbol": symbol}
                    )
            except Exception as e:
                logger.warning(f"Could not check for duplicate orders: {e}")
            
            # B. Signal freshness check
            if signal_timestamp:
                stale_secs = self.risk_config.get('stale_signal_secs', 120)
                sig_ts = as_utc(signal_timestamp)
                age_seconds = (now - sig_ts).total_seconds()
                
                if age_seconds > stale_secs:
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="STALE_SIGNAL",
                        metadata={
                            "signal_age_secs": age_seconds,
                            "stale_threshold_secs": stale_secs
                        }
                    )
            
            # C. Market state check
            allow_extended_hours = self.risk_config.get('allow_extended_hours', False)
            if not (clock.is_open or allow_extended_hours or (clock.next_open and clock.next_open - now < timedelta(minutes=1))):
                metadata = {
                    "is_open": clock.is_open,
                    "next_open": clock.next_open.isoformat() if clock.next_open else None,
                    "extended_hours_allowed": allow_extended_hours
                }
                try:
                    from zoneinfo import ZoneInfo
                    local_tz = ZoneInfo("America/Chicago")
                    if clock.next_open:
                        metadata["next_open_ct"] = clock.next_open.astimezone(local_tz).isoformat()
                except Exception:
                    pass
                return GateDecision(
                    result=GateResult.REJECT,
                    reason="MARKET_CLOSED",
                    metadata=metadata
                )
            
            # D. Price sanity check (reject fat-finger)
            last_price = self._get_price_mid(symbol, ref_price=px_ref)
            if not last_price:
                return GateDecision(
                    result=GateResult.REJECT,
                    reason="PRICE_UNKNOWN",
                    metadata={"symbol": symbol, "ref_price": px_ref}
                )
            
            if px_ref and last_price:
                max_slip_pct = self.risk_config.get('max_slip_pct', 0.5)
                price_deviation = abs(last_price - px_ref) / px_ref * 100
                if price_deviation > max_slip_pct:
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="PRICE_OOB",
                        metadata={
                            "last_price": last_price,
                            "ref_price": px_ref,
                            "deviation_pct": price_deviation,
                            "max_deviation_pct": max_slip_pct
                        }
                    )
            
            # E. Side/position rules
            allow_shorts = self.risk_config.get('allow_shorts', False)
            if side == "sell":
                if current_position <= 0 and not allow_shorts:
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="NO_POSITION_LONG_ONLY",
                        metadata={
                            "current_position": current_position,
                            "allow_shorts": allow_shorts
                        }
                    )
            
            # F. Sizing caps (split by intent: reducer vs opener)
            eff_price = last_price  # already validated above
            notional = abs(qty_target) * eff_price  # qty_target is now the actual order quantity
            reducer = _is_reducer(side, current_position)
            
            # Min notional (split by intent)
            if reducer:
                min_notional = self.risk_config.get('min_notional_reducer', 0)
                if notional < min_notional:
                    allow_dust_close = self.risk_config.get('allow_dust_close', True)
                    if not (allow_dust_close and abs(qty_target) >= abs(current_position)):
                        return GateDecision(
                            result=GateResult.REJECT,
                            reason="TOO_SMALL",
                            metadata={
                                "notional": notional,
                                "min_notional": min_notional,
                                "intent": "reduce"
                            }
                        )
            else:
                min_notional = self.risk_config.get('min_notional_opener', 10)
                if notional < min_notional:
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="TOO_SMALL",
                        metadata={
                            "notional": notional,
                            "min_notional": min_notional,
                            "intent": "open"
                        }
                    )
            
            # Max notional (split by intent) - CLIP instead of REJECT
            if reducer:
                max_notional = self.risk_config.get('max_notional_reducer', None)  # None = unlimited
            else:
                max_notional = self.risk_config.get('max_notional_opener', 600)
            
            if max_notional is not None and notional > max_notional:
                # Clip to max notional instead of rejecting
                lot_size = self.risk_config.get('lot_size', 5)
                max_shares = int(max_notional // last_price)
                clipped_shares = (max_shares // lot_size) * lot_size
                
                if clipped_shares <= 0:
                    return GateDecision(
                        result=GateResult.REJECT,
                        reason="TOO_LARGE_CANNOT_CLIP",
                        metadata={
                            "notional": notional,
                            "max_notional": max_notional,
                            "intent": "reduce" if reducer else "open"
                        }
                    )
                
                # Apply direction
                clipped_qty = clipped_shares if qty_target > 0 else -clipped_shares
                clipped_notional = abs(clipped_qty) * last_price
                
                logger.info(f"GateIn {symbol}: CLIPPED from {qty_target} to {clipped_qty} (${notional:.2f} → ${clipped_notional:.2f})")
                
                # Sanity assertions to prevent constructor errors
                assert isinstance(clipped_qty, int), f"clipped_qty must be int, got {type(clipped_qty)}"
                assert clipped_qty != 0, "clipped_qty cannot be zero"
                
                return GateDecision(
                    result=GateResult.APPROVE,
                    reason="CLIPPED_PER_TRADE",
                    shares=clipped_qty,
                    metadata={
                        "original_qty": qty_target,
                        "original_notional": notional,
                        "clipped_qty": clipped_qty,
                        "clipped_notional": clipped_notional,
                        "max_notional": max_notional,
                        "intent": "reduce" if reducer else "open",
                        "clip_ratio": abs(clipped_qty) / abs(qty_target),
                        "price": last_price
                    }
                )
            
            # Safety: reducers must not increase position magnitude
            if reducer and abs(qty_target) > abs(current_position):
                return GateDecision(
                    result=GateResult.REJECT,
                    reason="TOO_LARGE",
                    metadata={
                        "qty": qty_target,
                        "current_pos": current_position,
                        "intent": "reduce_over"
                    }
                )
            
            # Per-symbol cap check (skip for reducers since they reduce exposure)
            if not reducer:
                max_pos_pct = self.risk_config.get('max_pos_pct', 0.05)
                equity = float(account.equity)
                sym_cap = max_pos_pct * equity
                
                # Compute post-trade position correctly
                if side == "sell":
                    new_pos = current_position - qty_target
                else:
                    new_pos = current_position + qty_target
                
                new_position_value = abs(new_pos) * eff_price
                if new_position_value > sym_cap:
                    max_shares = int(sym_cap / eff_price) if eff_price > 0 else 0
                    if abs(qty_target) > max_shares:
                        return GateDecision(
                            result=GateResult.REJECT,
                            reason="SYMBOL_CAP",
                            metadata={
                                "new_position_value": new_position_value,
                                "symbol_cap": sym_cap,
                                "max_pos_pct": max_pos_pct,
                                "intent": "open"
                            }
                        )
            
            # Note: Daily order throttling is now handled by the risk manager
            # before orders reach the pre-trade gate
            
            # Round to lot size (simplified - assumes 1 share lots)
            final_shares = int(qty_target)
            
            # Sanity assertions to prevent constructor errors
            assert isinstance(final_shares, int), f"final_shares must be int, got {type(final_shares)}"
            assert final_shares != 0, "final_shares cannot be zero"
            
            # Note: Daily counters are now tracked by the risk manager
            
            return GateDecision(
                result=GateResult.APPROVE,
                reason="APPROVED",
                shares=final_shares,
                metadata={
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "side": side,
                    "order_quantity": final_shares,
                    "current_position": current_position,
                    "new_position": current_position + final_shares,
                    "notional": abs(final_shares) * (eff_price or 0.0),
                    "last_price": last_price,
                    "ref_price": px_ref,
                    "intent": intent
                }
            )
        
        except Exception as e:
            logger.error(f"Error in pre-trade gate: {e}")
            return GateDecision(
                result=GateResult.REJECT,
                reason="GATE_ERROR",
                metadata={"error": str(e)}
            )
    
    def log_decision(self, decision: GateDecision, signal_id: str, symbol: str):
        """Log the gate decision in structured JSON format."""
        log_entry = {
            "ts": datetime.now().isoformat(),
            "signal_id": signal_id,
            "symbol": symbol,
            "result": decision.result.value,
            "reason": decision.reason,
            "shares": decision.shares,
            "metadata": decision.metadata or {}
        }
        
        if decision.result == GateResult.APPROVE:
            logger.info(f"PRETRADE_GATE_APPROVE: {log_entry}")
        else:
            logger.warning(f"PRETRADE_GATE_REJECT: {log_entry}")
