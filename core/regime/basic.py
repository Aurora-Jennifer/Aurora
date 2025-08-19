"""
Basic Regime Extractor
Simple regime detection using technical indicators.
"""

import numpy as np

from ..composer.contracts import MarketState, RegimeExtractor, RegimeFeatures


class BasicRegimeExtractor(RegimeExtractor):
    """
    Basic regime extractor using simple technical indicators.

    Identifies market regimes: trend, chop, volatile, unknown
    """

    def __init__(
        self,
        lookback: int = 20,
        trend_threshold: float = 0.3,
        chop_threshold: float = 0.7,
        volatility_threshold: float = 0.02,
        adx_threshold: float = 25.0,
    ):
        """
        Initialize basic regime extractor.

        Args:
            lookback: Lookback period for calculations
            trend_threshold: Threshold for trend strength
            chop_threshold: Threshold for choppiness
            volatility_threshold: Threshold for volatility
            adx_threshold: Threshold for ADX (trend strength)
        """
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.chop_threshold = chop_threshold
        self.volatility_threshold = volatility_threshold
        self.adx_threshold = adx_threshold

    def extract(self, market_state: MarketState) -> RegimeFeatures:
        """
        Extract regime features from market state.

        Args:
            market_state: Current market state

        Returns:
            RegimeFeatures with extracted characteristics
        """
        prices = market_state.prices

        if len(prices) < self.lookback + 1:  # Need at least lookback + 1 elements
            return RegimeFeatures(
                trend_strength=0.0,
                choppiness=0.0,
                volatility=0.0,
                momentum=0.0,
                regime_type="unknown",
            )

        # Calculate basic features
        returns = np.diff(np.log(prices))
        volatility = 0.0 if len(returns) < self.lookback else np.std(returns[-self.lookback:])

        # Trend strength (linear regression slope)
        trend_strength = self._calculate_trend_strength(prices)

        # Choppiness (ADX-like measure)
        choppiness = self._calculate_choppiness(prices)

        # Momentum
        momentum = self._calculate_momentum(prices)

        # Determine regime type
        regime_type = self._classify_regime(trend_strength, choppiness, volatility)

        return RegimeFeatures(
            trend_strength=trend_strength,
            choppiness=choppiness,
            volatility=volatility,
            momentum=momentum,
            regime_type=regime_type,
        )

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression."""
        if len(prices) < self.lookback:
            return 0.0

        x = np.arange(self.lookback)
        y = prices[-self.lookback :]

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Normalize by mean price
        mean_price = np.mean(y)
        if mean_price == 0:
            return 0.0

        # Use tanh to bound between -1 and 1
        trend_strength = np.tanh(slope / mean_price)

        return trend_strength

    def _calculate_choppiness(self, prices: np.ndarray) -> float:
        """Calculate choppiness (ADX-like measure)."""
        if len(prices) < self.lookback + 1:  # Need at least lookback + 1 elements for diff
            return 0.0

        # Additional bounds check
        if len(prices) <= self.lookback:
            return 0.0

        # Calculate high-low range
        high_low_range = np.max(prices[-self.lookback :]) - np.min(prices[-self.lookback :])

        # Calculate path length (sum of absolute price changes)
        price_changes = np.diff(prices[-self.lookback :])
        path_length = np.sum(np.abs(price_changes))

        # Choppiness = 1 - (range / path_length)
        if path_length == 0:
            return 0.0

        choppiness = 1.0 - (high_low_range / path_length)

        return np.clip(choppiness, 0.0, 1.0)

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum (price change over lookback period)."""
        if len(prices) < self.lookback + 1:  # Need at least lookback + 1 elements
            return 0.0

        # Use safe indexing with bounds checking
        if len(prices) <= self.lookback:
            return 0.0

        # Import safety functions
        from ..utils import _last, _safe_len

        if _safe_len(prices) < self.lookback + 1:
            return 0.0

        start_price = prices[-self.lookback] if _safe_len(prices) > self.lookback else None
        end_price = _last(prices)

        if start_price == 0:
            return 0.0

        momentum = (end_price - start_price) / start_price

        # Use tanh to bound between -1 and 1
        return np.tanh(momentum)

    def _classify_regime(self, trend_strength: float, choppiness: float, volatility: float) -> str:
        """
        Classify market regime based on features.

        Args:
            trend_strength: Trend strength (-1 to 1)
            choppiness: Choppiness measure (0 to 1)
            volatility: Volatility measure

        Returns:
            Regime type string
        """
        # Strong trend
        if abs(trend_strength) > self.trend_threshold:
            return "trend"

        # High choppiness (sideways market)
        elif choppiness > self.chop_threshold:
            return "chop"

        # High volatility
        elif volatility > self.volatility_threshold:
            return "volatile"

        # Default
        else:
            return "unknown"

    @property
    def name(self) -> str:
        return "basic_kpis"


class AdvancedRegimeExtractor(RegimeExtractor):
    """
    Advanced regime extractor with more sophisticated indicators.

    Uses multiple timeframes and advanced technical indicators.
    """

    def __init__(
        self,
        short_lookback: int = 10,
        medium_lookback: int = 20,
        long_lookback: int = 50,
        trend_threshold: float = 0.3,
        chop_threshold: float = 0.7,
        volatility_threshold: float = 0.02,
    ):
        """
        Initialize advanced regime extractor.

        Args:
            short_lookback: Short-term lookback
            medium_lookback: Medium-term lookback
            long_lookback: Long-term lookback
            trend_threshold: Threshold for trend strength
            chop_threshold: Threshold for choppiness
            volatility_threshold: Threshold for volatility
        """
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.trend_threshold = trend_threshold
        self.chop_threshold = chop_threshold
        self.volatility_threshold = volatility_threshold

    def extract(self, market_state: MarketState) -> RegimeFeatures:
        """
        Extract regime features using multiple timeframes.

        Args:
            market_state: Current market state

        Returns:
            RegimeFeatures with extracted characteristics
        """
        prices = market_state.prices

        if len(prices) < self.long_lookback:
            return RegimeFeatures(
                trend_strength=0.0,
                choppiness=0.0,
                volatility=0.0,
                momentum=0.0,
                regime_type="unknown",
            )

        # Multi-timeframe analysis
        short_trend = self._calculate_trend_strength(prices, self.short_lookback)
        medium_trend = self._calculate_trend_strength(prices, self.medium_lookback)
        long_trend = self._calculate_trend_strength(prices, self.long_lookback)

        # Weighted trend strength (more weight on longer timeframes)
        trend_strength = 0.2 * short_trend + 0.3 * medium_trend + 0.5 * long_trend

        # Multi-timeframe choppiness
        short_chop = self._calculate_choppiness(prices, self.short_lookback)
        medium_chop = self._calculate_choppiness(prices, self.medium_lookback)
        choppiness = 0.4 * short_chop + 0.6 * medium_chop

        # Volatility across timeframes
        returns = np.diff(np.log(prices))
        short_vol = np.std(returns[-self.short_lookback :])
        medium_vol = np.std(returns[-self.medium_lookback :])
        volatility = 0.3 * short_vol + 0.7 * medium_vol

        # Momentum (price change over medium term)
        momentum = self._calculate_momentum(prices, self.medium_lookback)

        # Regime classification with more sophisticated logic
        regime_type = self._classify_regime_advanced(
            trend_strength, choppiness, volatility, short_trend, long_trend
        )

        return RegimeFeatures(
            trend_strength=trend_strength,
            choppiness=choppiness,
            volatility=volatility,
            momentum=momentum,
            regime_type=regime_type,
        )

    def _calculate_trend_strength(self, prices: np.ndarray, lookback: int) -> float:
        """Calculate trend strength for specific lookback."""
        if len(prices) < lookback:
            return 0.0

        x = np.arange(lookback)
        y = prices[-lookback:]

        slope = np.polyfit(x, y, 1)[0]
        mean_price = np.mean(y)

        if mean_price == 0:
            return 0.0

        return np.tanh(slope / mean_price)

    def _calculate_choppiness(self, prices: np.ndarray, lookback: int) -> float:
        """Calculate choppiness for specific lookback."""
        if len(prices) < lookback:
            return 0.0

        high_low_range = np.max(prices[-lookback:]) - np.min(prices[-lookback:])
        price_changes = np.diff(prices[-lookback:])
        path_length = np.sum(np.abs(price_changes))

        if path_length == 0:
            return 0.0

        choppiness = 1.0 - (high_low_range / path_length)
        return np.clip(choppiness, 0.0, 1.0)

    def _calculate_momentum(self, prices: np.ndarray, lookback: int) -> float:
        """Calculate momentum for specific lookback."""
        if len(prices) < lookback:
            return 0.0

        # Import safety functions
        from ..utils import _last, _safe_len

        if _safe_len(prices) < lookback + 1:
            return 0.0

        start_price = prices[-lookback] if _safe_len(prices) > lookback else None
        end_price = _last(prices)

        if start_price == 0:
            return 0.0

        momentum = (end_price - start_price) / start_price
        return np.tanh(momentum)

    def _classify_regime_advanced(
        self,
        trend_strength: float,
        choppiness: float,
        volatility: float,
        short_trend: float,
        long_trend: float,
    ) -> str:
        """
        Advanced regime classification using multiple features.

        Args:
            trend_strength: Weighted trend strength
            choppiness: Weighted choppiness
            volatility: Weighted volatility
            short_trend: Short-term trend
            long_trend: Long-term trend

        Returns:
            Regime type string
        """
        # Strong trend with alignment across timeframes
        if abs(trend_strength) > self.trend_threshold and np.sign(short_trend) == np.sign(
            long_trend
        ):
            return "trend"

        # High choppiness with low trend strength
        elif choppiness > self.chop_threshold and abs(trend_strength) < self.trend_threshold * 0.5:
            return "chop"

        # High volatility
        elif volatility > self.volatility_threshold:
            return "volatile"

        # Weak trend (trending but not strong)
        elif abs(trend_strength) > self.trend_threshold * 0.5:
            return "weak_trend"

        # Default
        else:
            return "unknown"

    @property
    def name(self) -> str:
        return "advanced_kpis"


# Factory function for creating regime extractors
def create_regime_extractor(extractor_type: str, **kwargs) -> RegimeExtractor:
    """
    Factory function to create regime extractors.

    Args:
        extractor_type: Type of extractor ('basic' or 'advanced')
        **kwargs: Additional parameters

    Returns:
        RegimeExtractor instance
    """
    if extractor_type == "basic":
        return BasicRegimeExtractor(**kwargs)
    elif extractor_type == "advanced":
        return AdvancedRegimeExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown regime extractor type: {extractor_type}")
