"""
Position Sizing Engine

Converts trading signals into appropriate position sizes based on signal strength,
portfolio value, and risk parameters. Implements various position sizing strategies.
"""

import logging
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .order_types import Order, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class SizeDecision:
    qty: int            # signed shares
    ref_price: float
    notional: float     # abs(qty) * ref_price
    reason: str | None = None  # for skipped cases


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing engine."""
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_total_exposure: float = 0.8  # 80% of portfolio total
    min_trade_size: float = 100.0  # $100 minimum trade
    max_trade_size: float = 10000.0  # $10,000 maximum trade
    signal_threshold: float = 0.1  # Minimum signal strength to trade
    volatility_adjustment: bool = True  # Adjust for asset volatility
    portfolio_heat: float = 0.02  # 2% portfolio heat per trade
    # New: hard cap applied by sizer (gate remains backstop)
    order_notional_cap: float = 500.0
    # New: minimum shares per order (post-cap)
    min_shares: int = 1
    # New: capital utilization factor to scale positions
    capital_utilization_factor: float = 1.0
    # Professional-grade position management
    position_cap: float = 15000.0  # Hard risk cap per position
    buffer_band_pct: float = 0.05  # 5% buffer zone around target
    rebalance_trigger: str = "signal_change"  # signal_change, periodic, threshold_breach
    order_lot_size: int = 5  # Round to nearest N shares
    min_rebalance_threshold: float = 0.02  # 2% minimum deviation to trigger rebalance


class PositionSizer:
    """
    Calculates appropriate position sizes based on signals and risk parameters.
    
    Responsibilities:
    - Convert signal strength to position size
    - Apply risk limits and constraints
    - Consider portfolio exposure and diversification
    - Handle minimum/maximum trade sizes
    - Adjust for asset volatility
    """
    
    def __init__(self, config: PositionSizingConfig):
        """
        Initialize Position Sizer.
        
        Args:
            config: Position sizing configuration
        """
        self.config = config
        logger.info(f"PositionSizer initialized with config: {config}")
    
    def calculate_position_size(
        self,
        signal: float,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict],
        volatility: Optional[float] = None
    ) -> Tuple[int, Dict[str, float]]:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal: Signal strength (-1 to 1)
            symbol: Trading symbol
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_positions: Current positions {symbol: {quantity, value, ...}}
            volatility: Asset volatility (optional, for volatility adjustment)
            
        Returns:
            Tuple of (quantity, sizing_metadata)
        """
        logger.debug(f"Calculating position size for {symbol}: signal={signal:.3f}, price=${current_price:.2f}")
        
        # Validate inputs
        if abs(signal) < self.config.signal_threshold:
            logger.debug(f"Signal {signal:.3f} below threshold {self.config.signal_threshold}")
            return 0, {"reason": "signal_below_threshold", "signal": signal, "threshold": self.config.signal_threshold}
        
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative")
            return 0, {"reason": "invalid_portfolio_value", "portfolio_value": portfolio_value}
        
        if current_price <= 0:
            logger.warning(f"Invalid price for {symbol}: {current_price}")
            return 0, {"reason": "invalid_price", "price": current_price}
        
        # Calculate base position size
        base_size = self._calculate_base_size(signal, portfolio_value, volatility)
        
        # Apply risk constraints
        constrained_size = self._apply_risk_constraints(
            base_size, symbol, current_price, portfolio_value, current_positions
        )
        
        # Convert to shares
        quantity = self._convert_to_shares(constrained_size, current_price)

        # Enforce sizer hard cap in shares
        try:
            max_shares_cap = int(self.config.order_notional_cap // current_price)
        except Exception:
            max_shares_cap = 0
        if max_shares_cap <= 0:
            logger.debug(f"Order notional cap too small for {symbol}: cap=${self.config.order_notional_cap:.2f}, price=${current_price:.2f}")
            return 0, {"reason": "cap_zero_shares", "cap": self.config.order_notional_cap, "price": current_price}
        if quantity > max_shares_cap:
            logger.debug(f"Clamping quantity for {symbol} from {quantity} to cap {max_shares_cap}")
            quantity = max_shares_cap

        # Enforce minimum shares
        if quantity < max(self.config.min_shares, 1):
            logger.debug(f"Quantity {quantity} below min_shares {self.config.min_shares} for {symbol}")
            return 0, {"reason": "below_min_shares", "quantity": quantity, "min_shares": self.config.min_shares}

        # Final validation
        if quantity == 0:
            logger.debug(f"Position size calculation resulted in zero quantity for {symbol}")
            return 0, {"reason": "zero_quantity", "base_size": base_size, "constrained_size": constrained_size}

        # Prepare metadata
        metadata = {
            "signal": signal,
            "base_size": base_size,
            "constrained_size": constrained_size,
            "quantity": quantity,
            "position_value": quantity * current_price,
            "position_pct": (quantity * current_price) / portfolio_value,
            "volatility_adjustment": volatility or 1.0,
            "order_notional_cap": self.config.order_notional_cap
        }
        
        logger.info(f"Position size for {symbol}: {quantity} shares (${quantity * current_price:.2f}, {metadata['position_pct']:.1%})")
        return quantity, metadata
    
    def _calculate_base_size(self, signal: float, portfolio_value: float, volatility: Optional[float]) -> float:
        """
        Calculate base position size from signal strength.
        
        Args:
            signal: Signal strength (-1 to 1)
            portfolio_value: Total portfolio value
            volatility: Asset volatility
            
        Returns:
            Base position size in dollars
        """
        # Use signal strength as position size multiplier
        signal_multiplier = abs(signal)
        
        # Base position size as percentage of portfolio
        base_pct = self.config.max_position_size * signal_multiplier
        
        # Apply volatility adjustment if enabled
        if self.config.volatility_adjustment and volatility:
            # Higher volatility = smaller position size
            volatility_multiplier = 1.0 / (1.0 + volatility)
            base_pct *= volatility_multiplier
        
        # Convert to dollar amount
        base_size = portfolio_value * base_pct
        
        return base_size
    
    def _apply_risk_constraints(
        self,
        base_size: float,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict]
    ) -> float:
        """
        Apply risk constraints to position size.
        
        Args:
            base_size: Base position size in dollars
            symbol: Trading symbol
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_positions: Current positions
            
        Returns:
            Constrained position size in dollars
        """
        constrained_size = base_size
        
        # 1. Maximum position size constraint
        max_position_value = portfolio_value * self.config.max_position_size
        constrained_size = min(constrained_size, max_position_value)
        
        # 2. Minimum trade size constraint
        if constrained_size < self.config.min_trade_size:
            logger.debug(f"Position size {constrained_size:.2f} below minimum {self.config.min_trade_size}")
            return 0.0
        
        # 3. Maximum trade size constraint
        constrained_size = min(constrained_size, self.config.max_trade_size)
        
        # 4. Total exposure constraint
        current_exposure = self._calculate_current_exposure(current_positions, portfolio_value)
        max_additional_exposure = (self.config.max_total_exposure - current_exposure) * portfolio_value
        
        if max_additional_exposure <= 0:
            logger.debug(f"Portfolio already at maximum exposure: {current_exposure:.1%}")
            return 0.0
        
        constrained_size = min(constrained_size, max_additional_exposure)
        
        # 5. Portfolio heat constraint
        max_heat_value = portfolio_value * self.config.portfolio_heat
        constrained_size = min(constrained_size, max_heat_value)
        
        return constrained_size
    
    def _calculate_current_exposure(self, current_positions: Dict[str, Dict], portfolio_value: float) -> float:
        """
        Calculate current portfolio exposure.
        
        Args:
            current_positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Current exposure as percentage of portfolio
        """
        if not current_positions or portfolio_value <= 0:
            return 0.0
        
        total_exposure = sum(pos.get('value', 0) for pos in current_positions.values())
        return total_exposure / portfolio_value
    
    def compute_size(self, symbol: str, target_weight: float, price: float,
                     portfolio_value: float, min_notional: float,
                     max_order_notional: float, capital_utilization_factor: float = 1.0) -> Optional[SizeDecision]:
        """
        Compute position size with price cap skip + 1-share bump logic.
        
        Args:
            symbol: Trading symbol
            target_weight: Target position weight (-1.0 to 1.0)
            price: Current price per share
            portfolio_value: Total portfolio value
            min_notional: Minimum order notional
            max_order_notional: Maximum order notional (cap)
            
        Returns:
            SizeDecision if tradeable, None if skipped
        """
        # Hard skip: ticket too expensive for cap (don't ping gate)
        if price > max_order_notional:
            return None  # interpreted as SKIP_PRICE_GT_CAP upstream

        intended_val = abs(target_weight) * portfolio_value * capital_utilization_factor
        capped_val = min(intended_val, max_order_notional)

        # primary qty from capped dollars
        qty = int(capped_val // price)

        # 1-share bump if economically non-trivial
        if qty == 0 and intended_val >= min_notional:
            # only if we can afford 1 share under the cap
            if price <= max_order_notional:
                qty = 1

        if qty == 0:
            logger.info(f"SizeDecision: {symbol} qty=0, intended_val={intended_val:.2f}, min_notional={min_notional}, price={price:.2f}")
            return None  # interpreted as SKIP_SIZE_ZERO upstream

        notional = float(qty * price)
        # signed direction
        if target_weight < 0:
            qty = -qty
        # Note: qty here is TARGET position, not order delta
        logger.info(f"SizeDecision: {symbol} qty={qty:+d} (TARGET), notional={notional:.2f}, intended_val={intended_val:.2f}")
        return SizeDecision(qty=qty, ref_price=price, notional=abs(notional))

    def compute_size_with_buffer(self, symbol: str, target_weight: float, price: float,
                                portfolio_value: float, current_position: int,
                                min_notional: float, capital_utilization_factor: float = 1.0) -> Optional[SizeDecision]:
        """
        Professional-grade position sizing with buffer zones and proper rebalancing triggers.
        
        FIXED: Proper signed value handling to prevent pushing past caps.
        
        Args:
            symbol: Trading symbol
            target_weight: Target position weight (-1.0 to 1.0)
            price: Current price per share
            portfolio_value: Total portfolio value
            current_position: Current position in shares
            min_notional: Minimum order notional
            capital_utilization_factor: Factor to scale intended position values
            
        Returns:
            SizeDecision with order delta (not target position), None if no action needed
        """
        # Calculate signal target (what the model wants) - KEEP SIGN
        intended_val = target_weight * portfolio_value * capital_utilization_factor  # SIGNED
        cap_val = self.config.position_cap
        
        # 1) Clip the SIGNED target by cap
        cap_target_val = max(-cap_val, min(cap_val, intended_val))  # SIGNED
        
        # 2) Band edges (signed, around cap_target)
        buffer_band = self.config.buffer_band_pct
        lower_bound = cap_target_val * (1 - buffer_band)  # SIGNED
        upper_bound = cap_target_val * (1 + buffer_band)  # SIGNED
        
        # Current position value (signed)
        current_pos_val = current_position * price
        
        # Cap-satisfied guard: if model wants beyond cap but we're already at/within band around cap → NOOP
        if intended_val > cap_val and current_pos_val >= lower_bound:
            logger.debug(f"{symbol}: At cap satisfied (long) - current=${current_pos_val:.2f} >= lower=${lower_bound:.2f}")
            return None
        if intended_val < -cap_val and current_pos_val <= upper_bound:
            logger.debug(f"{symbol}: At cap satisfied (short) - current=${current_pos_val:.2f} <= upper=${upper_bound:.2f}")
            return None
        
        # Check if within buffer zone
        if lower_bound <= current_pos_val <= upper_bound:
            logger.debug(f"{symbol}: Current ${current_pos_val:.2f} within buffer zone [${lower_bound:.2f}, ${upper_bound:.2f}], no action")
            return None
        
        # 3) Move to nearest band edge (keep sign) - but never exceed cap
        if current_pos_val < lower_bound:
            new_target_val = lower_bound
            logger.debug(f"{symbol}: Current ${current_pos_val:.2f} < lower bound ${lower_bound:.2f}, moving to ${new_target_val:.2f}")
        else:  # current_pos_val > upper_bound
            new_target_val = upper_bound
            logger.debug(f"{symbol}: Current ${current_pos_val:.2f} > upper bound ${upper_bound:.2f}, moving to ${new_target_val:.2f}")
        
        # CRITICAL: Ensure we never exceed the cap
        if abs(new_target_val) > cap_val:
            new_target_val = cap_val if new_target_val > 0 else -cap_val
            logger.debug(f"{symbol}: Clipped to cap: ${new_target_val:.2f}")
        
        # Check minimum rebalance threshold
        rebalance_amount = abs(new_target_val - current_pos_val)
        min_rebalance_val = portfolio_value * self.config.min_rebalance_threshold
        
        if rebalance_amount < min_rebalance_val:
            logger.debug(f"{symbol}: Rebalance amount ${rebalance_amount:.2f} below threshold ${min_rebalance_val:.2f}, no action")
            return None
        
        # 4) Convert to SIGNED shares and round to lots
        signed_target_shares = round(new_target_val / price / self.config.order_lot_size) * self.config.order_lot_size  # SIGNED
        delta_shares = int(signed_target_shares - current_position)  # SIGNED
        
        # SANITY ASSERTIONS (catch sign bugs forever)
        assert delta_shares == signed_target_shares - current_position, f"Delta calculation error: {delta_shares} != {signed_target_shares} - {current_position}"
        
        # Cap-satisfied assertions
        if intended_val < -cap_val and current_pos_val <= upper_bound:
            assert delta_shares >= 0, f"cap-satisfied short attempted to sell more: delta={delta_shares}"
        if intended_val > cap_val and current_pos_val >= lower_bound:
            assert delta_shares <= 0, f"cap-satisfied long attempted to buy more: delta={delta_shares}"
        
        # Final validation
        if delta_shares == 0:
            logger.debug(f"{symbol}: Order delta is zero after rounding, no action")
            return None
        
        # Check minimum notional
        order_notional = abs(delta_shares) * price
        if order_notional < min_notional:
            logger.debug(f"{symbol}: Order notional ${order_notional:.2f} below minimum ${min_notional:.2f}")
            return None
        
        # Derive side from signed delta
        side = "buy" if delta_shares > 0 else "sell"
        qty = abs(delta_shares)
        
        # Create SizeDecision with order delta (not target position)
        logger.info(f"SizeDecision: {symbol} order_delta={delta_shares:+d}, current={current_position:+d}, "
                   f"target_shares={signed_target_shares:.0f}, side={side}, qty={qty}, notional=${order_notional:.2f}, "
                   f"signal_target=${intended_val:.2f}, cap_target=${cap_target_val:.2f}, "
                   f"buffer=[${lower_bound:.2f}, ${upper_bound:.2f}]")
        
        return SizeDecision(
            qty=delta_shares,  # This is the SIGNED ORDER DELTA
            ref_price=price,
            notional=order_notional,
            reason=f"buffer_rebalance_{side}"
        )

    def _convert_to_shares(self, position_value: float, current_price: float) -> int:
        """
        Convert position value to number of shares.
        
        Args:
            position_value: Position value in dollars
            current_price: Current market price
            
        Returns:
            Number of shares (rounded down)
        """
        if position_value <= 0 or current_price <= 0:
            return 0
        
        shares = position_value / current_price
        return int(math.floor(shares))
    
    def validate_position(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        portfolio_value: float,
        current_positions: Dict[str, Dict]
    ) -> Tuple[bool, str]:
        """
        Validate a position against risk limits.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_positions: Current positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if quantity == 0:
            return True, "zero_quantity"
        
        position_value = quantity * current_price
        
        # Check minimum trade size
        if position_value < self.config.min_trade_size:
            return False, f"position_value {position_value:.2f} below minimum {self.config.min_trade_size}"
        
        # Check maximum trade size
        if position_value > self.config.max_trade_size:
            return False, f"position_value {position_value:.2f} above maximum {self.config.max_trade_size}"
        
        # Check maximum position size
        position_pct = position_value / portfolio_value
        if position_pct > self.config.max_position_size:
            return False, f"position_pct {position_pct:.1%} above maximum {self.config.max_position_size:.1%}"
        
        # Check total exposure
        current_exposure = self._calculate_current_exposure(current_positions, portfolio_value)
        new_exposure = current_exposure + position_pct
        if new_exposure > self.config.max_total_exposure:
            return False, f"total_exposure {new_exposure:.1%} would exceed maximum {self.config.max_total_exposure:.1%}"
        
        return True, "valid"
    
    def calculate_rebalancing_orders(
        self,
        current_positions: Dict[str, Dict],
        target_weights: Dict[str, float],
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Calculate rebalancing orders to achieve target weights.
        
        Args:
            current_positions: Current positions {symbol: {quantity, value, ...}}
            target_weights: Target weights {symbol: weight}
            portfolio_value: Total portfolio value
            current_prices: Current prices {symbol: price}
            
        Returns:
            Dictionary of rebalancing quantities {symbol: quantity}
        """
        rebalancing_orders = {}
        
        # Calculate target values
        target_values = {symbol: weight * portfolio_value for symbol, weight in target_weights.items()}
        
        # Calculate current values
        current_values = {}
        for symbol, pos in current_positions.items():
            current_values[symbol] = pos.get('value', 0)
        
        # Calculate rebalancing amounts
        for symbol in set(list(current_positions.keys()) + list(target_weights.keys())):
            current_value = current_values.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            
            rebalance_amount = target_value - current_value
            
            if abs(rebalance_amount) >= self.config.min_trade_size:
                current_price = current_prices.get(symbol, 0)
                if current_price > 0:
                    quantity = self._convert_to_shares(abs(rebalance_amount), current_price)
                    if quantity > 0:
                        # Sign indicates direction (positive = buy, negative = sell)
                        rebalancing_orders[symbol] = int(math.copysign(quantity, rebalance_amount))
        
        logger.info(f"Calculated {len(rebalancing_orders)} rebalancing orders")
        return rebalancing_orders
    
    def get_position_sizing_summary(self) -> Dict[str, float]:
        """
        Get summary of position sizing configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "max_position_size": self.config.max_position_size,
            "max_total_exposure": self.config.max_total_exposure,
            "min_trade_size": self.config.min_trade_size,
            "max_trade_size": self.config.max_trade_size,
            "signal_threshold": self.config.signal_threshold,
            "portfolio_heat": self.config.portfolio_heat,
            "volatility_adjustment": self.config.volatility_adjustment,
            "order_notional_cap": self.config.order_notional_cap,
            "min_shares": self.config.min_shares
        }
