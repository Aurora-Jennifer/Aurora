"""
Ensemble Strategy
Uses feature engineering and signal combination for sophisticated trading signals
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from features.ensemble import SignalCombiner
from features.feature_engine import FeatureEngine
from strategies.base import BaseStrategy, StrategyParams

logger = logging.getLogger(__name__)


@dataclass
class EnsembleStrategyParams(StrategyParams):
    """Parameters for ensemble strategy."""

    # Feature selection
    trend_features: List[str] = None
    mean_reversion_features: List[str] = None
    volatility_features: List[str] = None
    volume_features: List[str] = None

    # Signal combination
    combination_method: str = "rolling_ic"  # "rolling_ic", "sharpe", "ridge", "voting"
    weight_window: int = 60
    min_obs: int = 30

    # Regime switching
    use_regime_switching: bool = False
    regime_method: str = "volatility"  # "volatility", "adx"

    # Risk management
    confidence_threshold: float = 0.3
    max_position_size: float = 1.0

    def __post_init__(self):
        """Set default feature lists if not provided."""
        if self.trend_features is None:
            self.trend_features = [
                "SMA_Crossover_20_180",
                "EMA_Crossover_20_180",
                "MACD_Crossover",
                "RSI_Momentum",
                "Donchian_Breakout_20",
            ]

        if self.mean_reversion_features is None:
            self.mean_reversion_features = [
                "RSI_2_Extreme",
                "RSI_3_Extreme",
                "ZScore_20",
                "BB_Touch_Upper",
                "BB_Touch_Lower",
            ]

        if self.volatility_features is None:
            self.volatility_features = [
                "ATR_Pct",
                "Realized_Vol_20",
                "ADX_Trend",
                "Vol_High",
                "Vol_Low",
            ]

        if self.volume_features is None:
            self.volume_features = [
                "OBV_Slope",
                "Volume_ZScore",
                "MFI_Overbought",
                "MFI_Oversold",
            ]


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy using feature engineering and signal combination.

    This strategy:
    1. Generates comprehensive features from price data
    2. Selects relevant features based on market regime
    3. Combines features using various methods (IC, Sharpe, Ridge, Voting)
    4. Applies confidence-based position sizing
    5. Supports regime switching between trend and mean reversion
    """

    def __init__(self, params: EnsembleStrategyParams):
        """
        Initialize ensemble strategy.

        Args:
            params: Strategy parameters
        """
        super().__init__(params)
        self.params = params

        # Initialize feature engine
        self.feature_engine = FeatureEngine()

        # Initialize signal combiner
        self.signal_combiner = None

        # Feature cache
        self.features = {}

        logger.info(
            f"Initialized EnsembleStrategy with {len(params.trend_features)} trend features"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using ensemble approach.

        Args:
            data: OHLCV data

        Returns:
            Trading signals (-1, 0, 1)
        """
        logger.info("Generating ensemble signals...")

        # Generate all features
        self.features = self.feature_engine.generate_all_features(data)

        # Initialize signal combiner with price data
        self.signal_combiner = SignalCombiner(data["Close"])

        # Add features to combiner based on regime
        if self.params.use_regime_switching:
            signals = self._generate_regime_switched_signals(data)
        else:
            signals = self._generate_combined_signals()

        # Apply confidence threshold
        if self.params.confidence_threshold > 0:
            signals = self._apply_confidence_filter(signals)

        # Apply position size limits
        signals = signals.clip(
            -self.params.max_position_size, self.params.max_position_size
        )

        logger.info(
            f"Generated ensemble signals with mean absolute value: {abs(signals).mean():.3f}"
        )

        return signals

    def _generate_combined_signals(self) -> pd.Series:
        """Generate signals by combining all selected features."""
        # Add all selected features to combiner
        all_features = (
            self.params.trend_features
            + self.params.mean_reversion_features
            + self.params.volatility_features
            + self.params.volume_features
        )

        for feature_name in all_features:
            if feature_name in self.features:
                self.signal_combiner.add_feature(
                    feature_name, self.features[feature_name]
                )
            else:
                logger.warning(
                    f"Feature {feature_name} not found in generated features"
                )

        # Compute weights and combine
        self.signal_combiner.compute_weights(
            method=self.params.combination_method,
            window=self.params.weight_window,
            min_obs=self.params.min_obs,
        )

        combined_signal, confidence = self.signal_combiner.combine()

        # Store confidence for later use
        self.confidence = confidence

        return combined_signal

    def _generate_regime_switched_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with regime switching."""
        # Detect market regime
        regime = self._detect_regime(data)

        # Initialize combined signal
        combined_signal = pd.Series(0, index=data.index)
        self.confidence = pd.Series(0, index=data.index)

        # Generate signals for each regime
        for regime_name in ["trend", "mean_reversion", "choppy"]:
            regime_mask = regime == regime_name

            if regime_mask.sum() > 0:
                # Select features for this regime
                if regime_name == "trend":
                    regime_features = self.params.trend_features
                elif regime_name == "mean_reversion":
                    regime_features = self.params.mean_reversion_features
                else:  # choppy
                    regime_features = self.params.volatility_features

                # Create regime-specific combiner
                regime_combiner = SignalCombiner(data["Close"])

                # Add regime-specific features
                for feature_name in regime_features:
                    if feature_name in self.features:
                        regime_combiner.add_feature(
                            feature_name, self.features[feature_name]
                        )

                if regime_combiner.features:
                    # Compute weights and combine for this regime
                    regime_combiner.compute_weights(
                        method=self.params.combination_method,
                        window=self.params.weight_window,
                        min_obs=self.params.min_obs,
                    )

                    regime_signal, regime_confidence = regime_combiner.combine()

                    # Apply regime signals with proper dtype handling
                    combined_signal = combined_signal.astype(float)
                    regime_signal = regime_signal.astype(float)
                    combined_signal[regime_mask] = regime_signal[regime_mask]

                    self.confidence = self.confidence.astype(float)
                    regime_confidence = regime_confidence.astype(float)
                    self.confidence[regime_mask] = regime_confidence[regime_mask]

        return combined_signal

    def _detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime."""
        if self.params.regime_method == "volatility":
            returns = data["Close"].pct_change()
            vol = returns.rolling(20).std()

            regime = pd.Series("choppy", index=data.index)
            regime[vol > vol.rolling(100).quantile(0.8)] = "trend"
            regime[vol < vol.rolling(100).quantile(0.2)] = "mean_reversion"

        elif self.params.regime_method == "adx":
            adx = self.features.get("ADX", pd.Series(0, index=data.index))

            regime = pd.Series("choppy", index=data.index)
            regime[adx > 25] = "trend"
            regime[adx < 15] = "mean_reversion"

        else:
            # Default to choppy
            regime = pd.Series("choppy", index=data.index)

        return regime

    def _apply_confidence_filter(self, signals: pd.Series) -> pd.Series:
        """Apply confidence threshold to filter signals."""
        if not hasattr(self, "confidence"):
            logger.warning("No confidence scores available, skipping confidence filter")
            return signals

        # Zero out signals below confidence threshold
        low_confidence = self.confidence < self.params.confidence_threshold
        filtered_signals = signals.copy()
        filtered_signals[low_confidence] = 0

        logger.info(
            f"Applied confidence filter: {low_confidence.sum()} signals filtered out"
        )

        return filtered_signals

    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the strategy."""
        return {
            "combination_method": "rolling_ic",
            "use_regime_switching": False,
            "confidence_threshold": 0.3,
            "max_position_size": 1.0,
            "weight_window": 60,
            "min_obs": 30,
        }

    def get_param_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization."""
        return {
            "combination_method": ["rolling_ic", "sharpe", "ridge", "voting"],
            "confidence_threshold": [0.0, 0.2, 0.3, 0.4, 0.5],
            "max_position_size": [0.5, 0.75, 1.0, 1.25, 1.5],
            "weight_window": [30, 60, 90, 120],
            "min_obs": [20, 30, 40, 50],
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from signal combiner."""
        if self.signal_combiner is None:
            return pd.DataFrame()

        return self.signal_combiner.get_feature_summary()

    def get_regime_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get summary of regime detection."""
        if not self.params.use_regime_switching:
            return pd.DataFrame()

        regime = self._detect_regime(data)
        regime_counts = regime.value_counts()

        summary_data = []
        for regime_name, count in regime_counts.items():
            summary_data.append(
                {
                    "Regime": regime_name,
                    "Count": count,
                    "Percentage": count / len(regime) * 100,
                }
            )

        return pd.DataFrame(summary_data)

    def plot_ensemble_analysis(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """Plot ensemble analysis including feature weights and confidence."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Feature importance
            if self.signal_combiner:
                importance_df = self.get_feature_importance()
                if not importance_df.empty:
                    importance_df.plot(
                        x="Feature", y="Weight", kind="bar", ax=axes[0, 0]
                    )
                    axes[0, 0].set_title("Feature Weights")
                    axes[0, 0].tick_params(axis="x", rotation=45)

            # Plot 2: Confidence over time
            if hasattr(self, "confidence"):
                self.confidence.plot(ax=axes[0, 1], alpha=0.7)
                axes[0, 1].set_title("Confidence Score Over Time")
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Regime detection
            if self.params.use_regime_switching:
                regime = self._detect_regime(data)
                regime.value_counts().plot(kind="pie", ax=axes[1, 0])
                axes[1, 0].set_title("Market Regime Distribution")

            # Plot 4: Signal distribution
            if hasattr(self, "signals"):
                self.signals.hist(bins=50, ax=axes[1, 1], alpha=0.7)
                axes[1, 1].set_title("Signal Distribution")
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Ensemble analysis plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib not available, skipping ensemble analysis plot")


# Example usage
if __name__ == "__main__":
    import yfinance as yf

    # Fetch sample data
    data = yf.download("SPY", start="2020-01-01", end="2025-01-01")

    # Create ensemble strategy
    params = EnsembleStrategyParams(
        combination_method="rolling_ic",
        use_regime_switching=True,
        confidence_threshold=0.3,
    )

    strategy = EnsembleStrategy(params)

    # Generate signals
    signals = strategy.generate_signals(data)

    # Print results
    print("Ensemble Strategy Results:")
    print(f"Signal mean: {signals.mean():.3f}")
    print(f"Signal std: {signals.std():.3f}")
    print(f"Non-zero signals: {(signals != 0).sum()}")

    # Print feature importance
    print("\nFeature Importance:")
    print(strategy.get_feature_importance())

    # Print regime summary
    if params.use_regime_switching:
        print("\nRegime Summary:")
        print(strategy.get_regime_summary(data))
