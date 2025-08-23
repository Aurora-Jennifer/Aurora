"""
Signal Ensemble Module
Combines multiple feature signals into a single trading signal with confidence scoring
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Combines multiple feature signals into a single trading signal with confidence scoring.

    Supports multiple combination methods:
    - rolling_ic: Information Coefficient based weighting
    - sharpe: Sharpe ratio based weighting
    - ridge: Ridge regression based combination
    - voting: Simple voting/averaging
    """

    def __init__(self, price_data: pd.Series):
        """
        Initialize SignalCombiner with price data.

        Args:
            price_data: Price series with DateTimeIndex (e.g., Close prices)
        """
        self.price_data = price_data
        self.features: dict[str, pd.Series] = {}
        self.weights: dict[str, float] = {}
        from core.utils import calculate_returns

        self.returns = calculate_returns(price_data, shift=-1)  # Next period returns

        logger.info(f"Initialized SignalCombiner with {len(price_data)} price observations")

    def add_feature(self, name: str, signal: pd.Series) -> None:
        """
        Add a feature signal to the ensemble.

        Args:
            name: Feature name
            signal: Signal series aligned with price index, values in [-1, +1] or real-valued
        """
        # Align signal with price data
        aligned_signal = signal.reindex(self.price_data.index).fillna(0)

        # Validate signal values
        if not np.isfinite(aligned_signal).all():
            logger.warning(f"Feature {name} contains non-finite values, filling with 0")
            aligned_signal = aligned_signal.replace([np.inf, -np.inf], 0).fillna(0)

        self.features[name] = aligned_signal
        logger.info(f"Added feature '{name}' with {len(aligned_signal)} observations")

    def compute_weights(
        self,
        method: str = "rolling_ic",
        window: int = 60,
        min_obs: int = 30,
        regularization: float = 1e-6,
    ) -> dict[str, float]:
        """
        Compute feature weights based on recent performance.

        Args:
            method: Weighting method ("rolling_ic", "sharpe", "ridge", "voting")
            window: Rolling window for performance calculation
            min_obs: Minimum observations required for weight calculation
            regularization: Regularization parameter for ridge regression

        Returns:
            Dict mapping feature names to weights
        """
        if not self.features:
            logger.warning("No features added, returning equal weights")
            return {}

        if method == "rolling_ic":
            weights = self._compute_ic_weights(window, min_obs)
        elif method == "sharpe":
            weights = self._compute_sharpe_weights(window, min_obs)
        elif method == "ridge":
            weights = self._compute_ridge_weights(window, min_obs, regularization)
        elif method == "voting":
            weights = self._compute_voting_weights()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize weights to sum to 1
        total_weight = sum(abs(w) for w in weights.values())
        normalized_weights = {name: w / total_weight for name, w in weights.items()} if total_weight > 0 else {name: 1.0 / len(weights) for name in weights}

        self.weights = normalized_weights
        logger.info(f"Computed weights using {method}: {normalized_weights}")
        return normalized_weights

    def _compute_ic_weights(self, window: int, min_obs: int) -> dict[str, float]:
        """Compute weights based on Information Coefficient."""
        weights = {}

        for name, signal in self.features.items():
            # Calculate rolling IC (correlation with next period returns)
            ic_series = signal.rolling(window).corr(self.returns)

            # Calculate mean IC and IC volatility
            mean_ic = ic_series.mean()
            ic_vol = ic_series.std()

            # Weight = mean_ic / ic_vol (with regularization)
            weight = mean_ic / (ic_vol + 1e-06) if ic_vol > 0 and not pd.isna(mean_ic) and not pd.isna(ic_vol) else 0.0

            weights[name] = weight

        return weights

    def _compute_sharpe_weights(self, window: int, min_obs: int) -> dict[str, float]:
        """Compute weights based on Sharpe ratio of feature returns."""
        weights = {}

        for name, signal in self.features.items():
            # Calculate feature returns (signal * next period return)
            feature_returns = signal * self.returns

            # Calculate rolling Sharpe ratio
            rolling_mean = feature_returns.rolling(window).mean()
            rolling_std = feature_returns.rolling(window).std()

            # Final Sharpe ratio
            sharpe = rolling_mean.iloc[-1] / (rolling_std.iloc[-1] + 1e-6)

            if not pd.isna(sharpe):
                weights[name] = sharpe
            else:
                weights[name] = 0.0

        return weights

    def _compute_ridge_weights(
        self, window: int, min_obs: int, regularization: float
    ) -> dict[str, float]:
        """Compute weights using Ridge regression."""
        if len(self.features) < 2:
            logger.warning("Need at least 2 features for ridge regression, using equal weights")
            return {name: 1.0 / len(self.features) for name in self.features}

        # Prepare feature matrix
        feature_matrix = pd.DataFrame(self.features)

        # Remove rows with missing data
        valid_data = feature_matrix.dropna()
        valid_returns = self.returns.reindex(valid_data.index)

        if len(valid_data) < min_obs:
            logger.warning(
                f"Insufficient data for ridge regression ({len(valid_data)} < {min_obs})"
            )
            return {name: 1.0 / len(self.features) for name in self.features}

        # Fit ridge regression
        try:
            ridge = Ridge(alpha=regularization, fit_intercept=False)
            ridge.fit(valid_data, valid_returns)

            weights = dict(zip(self.features.keys(), ridge.coef_, strict=False))
        except Exception as e:
            logger.error(f"Ridge regression failed: {e}")
            weights = {name: 1.0 / len(self.features) for name in self.features}

        return weights

    def _compute_voting_weights(self) -> dict[str, float]:
        """Compute equal weights for simple voting."""
        return {name: 1.0 / len(self.features) for name in self.features}

    def combine(self, normalize: bool = True, cap: float = 1.0) -> tuple[pd.Series, pd.Series]:
        """
        Combine features into a single signal with confidence score.

        Args:
            normalize: Whether to normalize the combined signal
            cap: Maximum absolute value for the combined signal

        Returns:
            Tuple of (combined_signal, confidence_score)
        """
        if not self.features:
            logger.warning("No features to combine")
            empty_series = pd.Series(0, index=self.price_data.index)
            return empty_series, empty_series

        if not self.weights:
            logger.warning("No weights computed, using equal weights")
            self.compute_weights(method="voting")

        # Combine signals using weights
        combined_signal = pd.Series(0, index=self.price_data.index)

        for name, signal in self.features.items():
            weight = self.weights.get(name, 0.0)
            combined_signal += weight * signal

        # Normalize if requested
        if normalize:
            signal_std = combined_signal.rolling(20).std()
            combined_signal = combined_signal / (signal_std + 1e-6)

        # Cap the signal
        combined_signal = combined_signal.clip(-cap, cap)

        # Calculate confidence score
        confidence = self._calculate_confidence(combined_signal)

        logger.info(
            f"Combined {len(self.features)} features into signal with mean confidence {confidence.mean():.3f}"
        )

        return combined_signal, confidence

    def _calculate_confidence(self, combined_signal: pd.Series) -> pd.Series:
        """
        Calculate confidence score based on signal strength and feature agreement.

        Returns:
            Confidence score between 0 and 1
        """
        # Signal strength component (absolute z-score)
        signal_std = combined_signal.rolling(20).std()
        signal_strength = np.abs(combined_signal) / (signal_std + 1e-6)
        signal_strength = signal_strength.clip(0, 3) / 3  # Normalize to [0, 1]

        # Feature agreement component
        agreement = pd.Series(0, index=self.price_data.index)

        if len(self.features) > 1:
            # Calculate percentage of features with same sign as combined signal
            feature_signs = pd.DataFrame(
                {name: np.sign(signal) for name, signal in self.features.items()}
            )
            combined_sign = np.sign(combined_signal)

            # Agreement = percentage of features with same sign
            agreement = (feature_signs == combined_sign.values.reshape(-1, 1)).mean(axis=1)

        # Combine components (average of signal strength and agreement)
        return (signal_strength + agreement) / 2


    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics for all features."""
        if not self.features:
            return pd.DataFrame()

        summary_data = []
        for name, signal in self.features.items():
            weight = self.weights.get(name, 0.0)

            # Calculate feature performance metrics
            feature_returns = signal * self.returns
            sharpe = feature_returns.mean() / (feature_returns.std() + 1e-6)
            hit_rate = (feature_returns > 0).mean()

            summary_data.append(
                {
                    "Feature": name,
                    "Weight": weight,
                    "Sharpe": sharpe,
                    "Hit_Rate": hit_rate,
                    "Mean_Signal": signal.mean(),
                    "Signal_Std": signal.std(),
                    "Non_Zero_Pct": (signal != 0).mean(),
                }
            )

        return pd.DataFrame(summary_data)

    def plot_ensemble(self, save_path: str | None = None) -> None:
        """Plot feature weights and confidence over time."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot feature weights
            if self.weights:
                features = list(self.weights.keys())
                weights = list(self.weights.values())
                ax1.bar(features, weights)
                ax1.set_title("Feature Weights")
                ax1.set_ylabel("Weight")
                ax1.tick_params(axis="x", rotation=45)

            # Plot confidence over time
            if self.features:
                combined_signal, confidence = self.combine()
                confidence.plot(ax=ax2, alpha=0.7)
                ax2.set_title("Confidence Score Over Time")
                ax2.set_ylabel("Confidence")
                ax2.set_xlabel("Date")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Ensemble plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib not available, skipping plot")


class RegimeSwitcher:
    """
    Switches between different feature sets based on market regime.
    """

    def __init__(self, price_data: pd.Series):
        self.price_data = price_data
        self.regime_indicators = {}
        self.regime_labels = pd.Series(index=price_data.index)

    def add_regime_indicator(
        self, name: str, indicator: pd.Series, threshold: float, regime_type: str
    ) -> None:
        """
        Add a regime indicator.

        Args:
            name: Indicator name
            indicator: Indicator series
            threshold: Threshold for regime classification
            regime_type: "trend" or "mean_reversion"
        """
        self.regime_indicators[name] = {
            "indicator": indicator,
            "threshold": threshold,
            "type": regime_type,
        }

    def detect_regime(self, method: str = "adx") -> pd.Series:
        """
        Detect market regime.

        Args:
            method: Regime detection method ("adx", "volatility", "hmm")

        Returns:
            Regime labels: "trend", "mean_reversion", or "choppy"
        """
        if method == "adx":
            return self._detect_adx_regime()
        if method == "volatility":
            return self._detect_volatility_regime()
        raise ValueError(f"Unknown regime detection method: {method}")

    def _detect_adx_regime(self) -> pd.Series:
        """Detect regime using ADX indicator."""
        # This would require ADX calculation
        # For now, return a simple regime based on volatility
        from core.utils import calculate_returns

        returns = calculate_returns(self.price_data, shift=0)
        vol = returns.rolling(20).std()

        # Simple regime classification
        regime = pd.Series("choppy", index=self.price_data.index)
        regime[vol > vol.quantile(0.7)] = "trend"
        regime[vol < vol.quantile(0.3)] = "mean_reversion"

        return regime

    def _detect_volatility_regime(self) -> pd.Series:
        """Detect regime using volatility bands."""
        from core.utils import calculate_returns

        returns = calculate_returns(self.price_data, shift=0)
        vol = returns.rolling(20).std()

        # Volatility-based regime classification
        regime = pd.Series("choppy", index=self.price_data.index)
        regime[vol > vol.quantile(0.8)] = "trend"
        regime[vol < vol.quantile(0.2)] = "mean_reversion"

        return regime


# Example usage and demo
if __name__ == "__main__":
    # Demo: Combine 3 features on SPY
    import yfinance as yf

    from core.data_sanity import get_data_sanity_wrapper

    # Fetch SPY data
    raw_spy = yf.download("SPY", start="2020-01-01", end="2025-01-01")

    # Validate and repair data using DataSanity
    data_sanity = get_data_sanity_wrapper()
    spy = data_sanity.validate_dataframe(raw_spy, "SPY")
    price_data = spy["Close"]

    # Create SignalCombiner
    combiner = SignalCombiner(price_data)

    # Add some example features
    # 1. SMA crossover
    sma_fast = price_data.rolling(20).mean()
    sma_slow = price_data.rolling(180).mean()
    sma_signal = ((sma_fast > sma_slow) * 2 - 1).fillna(0)
    combiner.add_feature("SMA_Crossover", sma_signal)

    # 2. RSI momentum
    delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_signal = ((rsi > 50) * 2 - 1).fillna(0)
    combiner.add_feature("RSI_Momentum", rsi_signal)

    # 3. Donchian breakout
    high_20 = price_data.rolling(20).max()
    low_20 = price_data.rolling(20).min()
    donchian_signal = (
        (price_data > high_20.shift(1)) * 1 + (price_data < low_20.shift(1)) * -1
    ).fillna(0)
    combiner.add_feature("Donchian_Breakout", donchian_signal)

    # Compute weights and combine
    combiner.compute_weights(method="rolling_ic", window=60)
    combined_signal, confidence = combiner.combine()

    # Print last 5 rows
    print("\nLast 5 rows of combined signal & confidence:")
    result_df = pd.DataFrame({"Combined_Signal": combined_signal, "Confidence": confidence})
    print(result_df.tail())

    # Print feature summary
    print("\nFeature Summary:")
    print(combiner.get_feature_summary())
