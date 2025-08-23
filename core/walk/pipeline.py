import logging

import numpy as np

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, X: np.ndarray, y: np.ndarray, strategy_type: str = "regime_ensemble"):
        self.X = X
        self.y = y
        self.mu = None
        self.sd = None
        self.model = None
        self.strategy_type = strategy_type
        self.feature_names = [
            "ret1",
            "ma20",
            "vol20",
            "zscore20",
        ]  # Default feature names
        self.regime_detector = None
        self.signal_combiner = None

    def fit_transforms(self, idx):
        mu = self.X[idx].mean(axis=0)
        sd = self.X[idx].std(axis=0)
        sd[sd == 0] = 1.0
        self.mu, self.sd = mu, sd

    def transform(self, idx):
        return (self.X[idx] - self.mu) / self.sd

    def fit_model(self, Xtr, ytr, warm=None):
        """Fit the signal model using regime-aware ensemble logic."""
        # Store training data for regime detection
        self.X_train = Xtr
        self.y_train = ytr

        # Initialize regime detector if needed
        if self.regime_detector is None:
            try:
                from core.regime_detector import RegimeDetector

                self.regime_detector = RegimeDetector(lookback_period=252)
            except ImportError:
                logger.warning("RegimeDetector not available, using fallback")
                self.regime_detector = None

        # Store model parameters
        self.model = {
            "confidence_threshold": 0.3,
            "trend_weight": 0.6,
            "mean_reversion_weight": 0.4,
            "volatility_threshold": 0.5,
            "signal_threshold": 0.01,  # Lower threshold to allow more trades
        }
        return self.model

    def predict(self, Xte):
        """Generate regime-aware ensemble signals."""
        if self.model is None:
            raise ValueError("Model not fitted")

        # Extract features (assuming standard order: ret1, ma20, vol20, zscore20)
        returns = Xte[:, 0]  # ret1
        vol20 = Xte[:, 2]  # vol20
        zscore = Xte[:, 3]  # zscore20

        # Calculate additional features
        price_ma_ratio = 1.0 + returns  # Approximate price/MA ratio
        volatility_ratio = np.clip(vol20, 1e-8, None)

        # Regime detection (simplified)
        regime = self._detect_regime(returns, volatility_ratio)

        # Store regime info for metadata
        self.current_regime = regime

        # Generate regime-specific signals
        if regime == "trend":
            signals = self._trend_following_signals(returns, price_ma_ratio, zscore)
        elif regime == "chop":
            signals = self._mean_reversion_signals(zscore, price_ma_ratio, volatility_ratio)
        elif regime == "volatile":
            signals = self._volatile_regime_signals(returns, zscore, volatility_ratio)
        else:
            signals = self._ensemble_signals(returns, zscore, price_ma_ratio, volatility_ratio)

        # Apply confidence threshold
        threshold = self.model["signal_threshold"]
        signals = np.where(np.abs(signals) < threshold, 0, np.sign(signals))

        return signals.astype(np.int8)

    def _detect_regime(self, returns: np.ndarray, volatility: np.ndarray) -> str:
        """Detect market regime using simplified logic."""
        if len(returns) < 20:
            return "unknown"

        # Calculate regime indicators
        trend_strength = np.abs(np.mean(returns[-20:]))
        volatility_level = np.mean(volatility[-20:])
        price_consistency = np.std(returns[-20:])

        # Regime classification
        if volatility_level > 0.02:  # High volatility
            return "volatile"
        if trend_strength > 0.005 and price_consistency < 0.01:  # Strong trend
            return "trend"
        if price_consistency < 0.008:  # Low volatility, choppy
            return "chop"
        return "unknown"

    def _trend_following_signals(
        self, returns: np.ndarray, price_ma_ratio: np.ndarray, zscore: np.ndarray
    ) -> np.ndarray:
        """Generate trend-following signals."""
        # Momentum-based signals
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        trend_signal = np.where(
            price_ma_ratio > 1.01, 0.5, np.where(price_ma_ratio < 0.99, -0.5, 0)
        )

        # Combine with momentum
        signals = trend_signal + 0.3 * momentum
        return np.clip(signals, -1, 1)

    def _mean_reversion_signals(
        self, zscore: np.ndarray, price_ma_ratio: np.ndarray, volatility: np.ndarray
    ) -> np.ndarray:
        """Generate mean reversion signals."""
        # Z-score based mean reversion
        zscore_signal = np.where(zscore > 1.0, -0.5, np.where(zscore < -1.0, 0.5, 0))

        # Price/MA ratio reversion
        ma_signal = np.where(price_ma_ratio > 1.02, -0.3, np.where(price_ma_ratio < 0.98, 0.3, 0))

        # Volatility adjustment
        vol_adj = np.clip(1.0 / (volatility * 100), 0.5, 2.0)

        signals = (zscore_signal + ma_signal) * vol_adj
        return np.clip(signals, -1, 1)

    def _volatile_regime_signals(
        self, returns: np.ndarray, zscore: np.ndarray, volatility: np.ndarray
    ) -> np.ndarray:
        """Generate signals for volatile regime."""
        # Reduce position sizes in volatile periods
        vol_scale = np.clip(1.0 / (volatility * 200), 0.2, 1.0)

        # Use shorter-term signals
        short_signal = np.where(np.abs(zscore) > 1.5, np.sign(zscore) * 0.3, 0)

        signals = short_signal * vol_scale
        return np.clip(signals, -1, 1)

    def _ensemble_signals(
        self,
        returns: np.ndarray,
        zscore: np.ndarray,
        price_ma_ratio: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        """Generate ensemble signals combining multiple approaches."""
        trend_weight = self.model["trend_weight"]
        mean_rev_weight = self.model["mean_reversion_weight"]

        trend_sig = self._trend_following_signals(returns, price_ma_ratio, zscore)
        mean_rev_sig = self._mean_reversion_signals(zscore, price_ma_ratio, volatility)

        signals = trend_weight * trend_sig + mean_rev_weight * mean_rev_sig
        return np.clip(signals, -1, 1)
