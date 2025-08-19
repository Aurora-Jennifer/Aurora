"""
Profit-Based Machine Learning System
Learns from actual trade outcomes to predict profitable market conditions
"""

import logging
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.ml.warm_start import WarmStartManager
from core.regime_detector import RegimeDetector
from experiments.persistence import FeaturePersistenceAnalyzer, RunMetadata

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Represents the outcome of a trade for learning."""

    timestamp: datetime
    symbol: str
    strategy: str
    regime: str
    entry_price: float
    exit_price: float
    position_size: float
    hold_duration: int  # days
    profit_loss: float
    profit_loss_pct: float
    market_features: dict[str, float]
    trade_features: dict[str, float]


@dataclass
class MarketState:
    """Current market state for prediction."""

    timestamp: datetime
    symbol: str
    regime: str
    volatility: float
    trend_strength: float
    momentum: float
    mean_reversion: float
    volume_profile: float
    price_level: float
    technical_features: dict[str, float]
    market_features: dict[str, float]


class ProfitLearner:
    """
    Machine learning system that learns from actual trade profits.

    Features:
    - Learns which market conditions lead to profitable trades
    - Predicts expected profit for different strategies
    - Adapts to changing market conditions
    - Uses ensemble of models for robustness
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the profit learner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.regime_detector = RegimeDetector(lookback_period=252)

        # Model storage
        self.models = {}
        self.feature_scalers = {}
        self.performance_history = []

        # Learning parameters
        self.min_trades_for_learning = config.get("min_trades_for_learning", 50)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.model_update_frequency = config.get("model_update_frequency", 10)

        # Unified model support
        self.unified_model = config.get("unified_model", False)
        self.assets = config.get("symbols", ["SPY"])
        self.cross_asset_features = config.get("unified_training", {}).get(
            "cross_asset_features", False
        )

        # State tracking
        self.trade_count = 0
        self.last_model_update = datetime.now()

        # Storage paths
        self.state_dir = Path("state/ml_profit_learner")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Load existing models if available
        self._load_models()

        # Initialize persistence and warm-start systems
        self.persistence_analyzer = FeaturePersistenceAnalyzer()
        self.warm_start_manager = WarmStartManager()
        self.current_run_id = None

        logger.info("Initialized ProfitLearner")

    def _load_models(self):
        """Load existing trained models and performance history."""
        try:
            # Load models
            model_file = self.state_dir / "models.pkl"
            if model_file.exists():
                with open(model_file, "rb") as f:
                    self.models = pickle.load(f)
                logger.info(f"Loaded {len(self.models)} existing models")

            # Load performance history
            history_file = self.state_dir / "performance_history.pkl"
            if history_file.exists():
                with open(history_file, "rb") as f:
                    self.performance_history = pickle.load(f)
                self.trade_count = len(self.performance_history)
                logger.info(f"Loaded {len(self.performance_history)} existing trade records")
            else:
                self.performance_history = []
                self.trade_count = 0

        except Exception as e:
            logger.warning(f"Could not load existing models and history: {e}")
            self.models = {}
            self.performance_history = []
            self.trade_count = 0

    def start_new_run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        model_type: str = "ridge",
        seed: int = 42,
    ) -> str:
        """Start a new ML run and generate run ID."""
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}"
        logger.info(f"Started new ML run: {self.current_run_id}")
        return self.current_run_id

    def log_feature_importance(self, regime: str = None):
        """Log feature importance for the current run."""
        try:
            if not self.current_run_id:
                logger.warning("No active run to log feature importance")
                return

            # Get current model coefficients
            if not self.models:
                logger.warning("No models available for feature importance logging")
                return

            # Use the first available model for feature importance
            model_key = list(self.models.keys())[0]
            model = self.models[model_key]

            if not hasattr(model, "coef_") or model.coef_ is None:
                logger.warning("Model has no coefficients for feature importance")
                return

            # Extract feature names and coefficients
            feature_names = self._get_feature_names()
            coefficients = dict(zip(feature_names, model.coef_, strict=False))

            # Calculate feature importance (absolute coefficients)
            feature_importances = {name: abs(coeff) for name, coeff in coefficients.items()}

            # Calculate performance metrics
            if self.performance_history:
                recent_trades = self.performance_history[-10:]  # Last 10 trades
                avg_profit = np.mean(
                    [
                        trade.profit_loss_pct
                        if hasattr(trade, "profit_loss_pct")
                        else trade.get("profit_loss_pct", 0.0)
                        for trade in recent_trades
                    ]
                )
                win_rate = np.mean(
                    [
                        1
                        if (
                            trade.profit_loss_pct
                            if hasattr(trade, "profit_loss_pct")
                            else trade.get("profit_loss_pct", 0.0)
                        )
                        > 0
                        else 0
                        for trade in recent_trades
                    ]
                )
                profits = [
                    trade.profit_loss_pct
                    if hasattr(trade, "profit_loss_pct")
                    else trade.get("profit_loss_pct", 0.0)
                    for trade in recent_trades
                ]
                sharpe_ratio = np.mean(profits) / (np.std(profits) + 1e-8)
            else:
                avg_profit = 0.0
                win_rate = 0.0
                sharpe_ratio = 0.0

            # Create metadata
            metadata = RunMetadata(
                run_id=self.current_run_id,
                timestamp=datetime.now().isoformat(),
                ticker="SPY",  # Default, should be passed from backtest
                start_date=datetime.now().strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                model_type="ridge",  # Default model type
                seed=42,  # Default seed
                total_trades=len(self.performance_history),
                avg_profit=avg_profit,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
            )

            # Log feature importance
            self.persistence_analyzer.log_feature_importance(
                run_id=self.current_run_id,
                feature_importances=feature_importances,
                coefficients=coefficients,
                metadata=metadata,
                regime=regime,
            )

            logger.info(f"Logged feature importance for run {self.current_run_id}")

        except Exception as e:
            logger.error(f"Error logging feature importance: {e}")

    def _get_feature_names(self) -> list[str]:
        """Get feature names for the current model."""
        # Default feature names - should be updated based on actual features used
        return [
            "z_score",
            "rsi",
            "price_position",
            "sma_ratio",
            "returns_5d",
            "volatility",
            "momentum",
            "mean_reversion",
            "volume_profile",
            "trend_strength",
            "regime_confidence",
            "signal_strength",
            "market_volatility",
            "position_size",
            "entry_price",
            "exit_price",
            "hold_duration",
            "strategy_confidence",
        ]

    def _save_models(self):
        """Save trained models and performance history."""
        try:
            # Save models
            model_file = self.state_dir / "models.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(self.models, f)

            # Save performance history
            history_file = self.state_dir / "performance_history.pkl"
            with open(history_file, "wb") as f:
                pickle.dump(self.performance_history, f)

            # Log feature importance if we have an active run
            if self.current_run_id:
                self.log_feature_importance()

            logger.info(
                f"Saved {len(self.models)} models and {len(self.performance_history)} trade records"
            )
        except Exception as e:
            logger.error(f"Could not save models and history: {e}")

    def extract_market_features(
        self, market_data: pd.DataFrame, symbol: str = None
    ) -> dict[str, float]:
        """Extract comprehensive market features for learning."""
        try:
            if len(market_data) < 50:
                return {}

            # Basic price features
            close = market_data["Close"]
            high = market_data["High"]
            low = market_data["Low"]
            volume = market_data["Volume"]

            # Returns and volatility
            returns = close.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]

            # Technical indicators
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            rsi = self._calculate_rsi(close, 14)

            # Volume features
            volume_ma = volume.rolling(20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_ma if volume_ma > 0 else 1.0

            # Price level features
            price_position = (close.iloc[-1] - low.rolling(20).min().iloc[-1]) / (
                high.rolling(20).max().iloc[-1] - low.rolling(20).min().iloc[-1]
            )

            # Momentum features
            momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            momentum_20 = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

            # Mean reversion features
            z_score = (close.iloc[-1] - sma_20) / (close.rolling(20).std().iloc[-1])

            features = {
                "volatility": float(volatility),
                "rsi": float(rsi),
                "sma_ratio": float(sma_20 / sma_50) if sma_50 > 0 else 1.0,
                "volume_ratio": float(volume_ratio),
                "price_position": float(price_position),
                "momentum_5": float(momentum_5),
                "momentum_20": float(momentum_20),
                "z_score": float(z_score),
                "returns_1d": float(returns.iloc[-1]) if len(returns) > 0 else 0.0,
                "returns_5d": float(returns.tail(5).sum()) if len(returns) >= 5 else 0.0,
            }

            # Add cross-asset features if unified model is enabled
            if self.unified_model and self.cross_asset_features and symbol:
                cross_asset_features = self._extract_cross_asset_features(symbol, market_data)
                features.update(cross_asset_features)

            return features

        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return {}

    def _extract_cross_asset_features(
        self, symbol: str, market_data: pd.DataFrame
    ) -> dict[str, float]:
        """Extract cross-asset features for unified model training."""
        try:
            features = {}

            # Asset-specific features
            returns_20d = (
                market_data["Close"].pct_change(20).iloc[-1] if len(market_data) >= 20 else 0
            )
            features[f"{symbol}_relative_strength"] = returns_20d
            features[f"{symbol}_volatility"] = (
                market_data["Close"].pct_change().rolling(20).std().iloc[-1]
                if len(market_data) >= 20
                else 0
            )

            # Market regime features (would need regime detector)
            features["market_regime_bull"] = 0.5
            features["market_regime_bear"] = 0.3
            features["market_regime_sideways"] = 0.2

            # Sector rotation features (for stocks)
            if symbol in ["SPY", "AAPL", "TSLA", "GOOG"]:
                features["sector_tech"] = 1.0 if symbol in ["AAPL", "TSLA", "GOOG"] else 0.0
                features["sector_broad"] = 1.0 if symbol == "SPY" else 0.0
            else:
                features["sector_crypto"] = 1.0 if symbol == "BTC-USD" else 0.0

            # Asset correlation features (placeholder - would need all assets)
            features["correlation_spy"] = 0.5 if symbol != "SPY" else 1.0
            features["correlation_tech"] = 0.7 if symbol in ["AAPL", "TSLA", "GOOG"] else 0.3

            return features

        except Exception as e:
            logger.error(f"Error extracting cross-asset features: {e}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0

    def extract_trade_features(self, trade_data: dict[str, Any]) -> dict[str, float]:
        """Extract features specific to the trade."""
        try:
            features = {
                "position_size": float(trade_data.get("position_size", 0.0)),
                "entry_price": float(trade_data.get("entry_price", 0.0)),
                "hold_duration": float(trade_data.get("hold_duration", 1)),
                "strategy_confidence": float(trade_data.get("confidence", 0.5)),
                "market_regime": float(trade_data.get("regime_confidence", 0.5)),
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting trade features: {e}")
            return {}

    def record_trade_outcome(self, trade_outcome: TradeOutcome):
        """
        Record a trade outcome for learning.

        Args:
            trade_outcome: Complete trade outcome data
        """
        try:
            # Store the trade outcome
            self.performance_history.append(trade_outcome)
            self.trade_count += 1

            # Update models periodically
            if self.trade_count % self.model_update_frequency == 0:
                self._update_models()

            logger.info(f"Recorded trade outcome: {trade_outcome.profit_loss_pct:.2%} profit")

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    def predict_profit_potential(
        self, market_data: pd.DataFrame, strategy: str, symbol: str = None
    ) -> dict[str, float]:
        """
        Predict profit potential for a strategy in current market conditions.

        Args:
            market_data: Current market data
            strategy: Strategy name
            symbol: Asset symbol (for cross-asset features)

        Returns:
            Dictionary with predictions
        """
        try:
            if len(self.performance_history) < self.min_trades_for_learning:
                return self._get_default_prediction(strategy)

            # Extract current market state
            market_features = self.extract_market_features(market_data, symbol)
            regime_name, confidence, regime_params = self.regime_detector.detect_regime(market_data)

            # Create feature vector
            features = self._create_feature_vector(market_features, strategy, regime_name)

            # Get predictions from models
            predictions = {}
            for model_name, model in self.models.items():
                if model_name.startswith(strategy):
                    try:
                        pred = model.predict([features])[0]
                        predictions[model_name] = float(pred)
                    except Exception as e:
                        logger.warning(f"Model {model_name} prediction failed: {e}")

            # Aggregate predictions
            if predictions:
                avg_profit_potential = np.mean(list(predictions.values()))
                confidence_score = np.std(
                    list(predictions.values())
                )  # Lower std = higher confidence
            else:
                avg_profit_potential = 0.0
                confidence_score = 1.0

            return {
                "expected_profit_pct": float(avg_profit_potential),
                "confidence": float(1.0 - min(confidence_score, 1.0)),
                "model_predictions": predictions,
                "market_regime": regime_name,
                "regime_confidence": float(confidence),
            }

        except Exception as e:
            logger.error(f"Error predicting profit potential: {e}")
            return self._get_default_prediction(strategy)

    def _create_feature_vector(
        self, market_features: dict[str, float], strategy: str, regime: str
    ) -> list[float]:
        """Create feature vector for model prediction."""
        # Encode strategy and regime
        strategy_encoding = {
            "regime_aware_ensemble": [1, 0, 0, 0, 0],
            "momentum": [0, 1, 0, 0, 0],
            "mean_reversion": [0, 0, 1, 0, 0],
            "sma_crossover": [0, 0, 0, 1, 0],
            "ensemble_basic": [0, 0, 0, 0, 1],
        }

        regime_encoding = {
            "trend": [1, 0, 0],
            "chop": [0, 1, 0],
            "volatile": [0, 0, 1],
        }

        # Combine all features
        features = []
        features.extend(strategy_encoding.get(strategy, [0, 0, 0, 0, 0]))
        features.extend(regime_encoding.get(regime, [0, 0, 0]))
        features.extend(
            [
                market_features.get("volatility", 0.0),
                market_features.get("rsi", 50.0) / 100.0,  # Normalize to 0-1
                market_features.get("sma_ratio", 1.0),
                market_features.get("volume_ratio", 1.0),
                market_features.get("price_position", 0.5),
                market_features.get("momentum_5", 0.0),
                market_features.get("momentum_20", 0.0),
                market_features.get("z_score", 0.0),
                market_features.get("returns_1d", 0.0),
                market_features.get("returns_5d", 0.0),
            ]
        )

        return features

    def _update_models(self):
        """Update models with new trade data."""
        try:
            if len(self.performance_history) < self.min_trades_for_learning:
                return

            # Group trades by strategy
            strategy_trades = {}
            for trade in self.performance_history:
                if trade.strategy not in strategy_trades:
                    strategy_trades[trade.strategy] = []
                strategy_trades[trade.strategy].append(trade)

            # Train/update models for each strategy
            for strategy, trades in strategy_trades.items():
                if len(trades) < 10:  # Need minimum trades per strategy
                    continue

                self._train_strategy_model(strategy, trades)

            # Save updated models
            self._save_models()
            self.last_model_update = datetime.now()

            logger.info(f"Updated models with {len(self.performance_history)} trades")

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    def _train_strategy_model(self, strategy: str, trades: list[TradeOutcome]):
        """Train model for a specific strategy."""
        try:
            # Prepare training data
            X = []
            y = []

            for trade in trades:
                # Create feature vector
                features = self._create_feature_vector(
                    trade.market_features, trade.strategy, trade.regime
                )
                X.append(features)
                y.append(trade.profit_loss_pct)

            X = np.array(X)
            y = np.array(y)

            # Train simple linear model (can be enhanced with more sophisticated models)
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)

            # Store model and scaler
            model_name = f"{strategy}_profit_predictor"
            self.models[model_name] = model
            self.feature_scalers[model_name] = scaler

            # Calculate model performance
            predictions = model.predict(X_scaled)
            mse = np.mean((y - predictions) ** 2)
            r2 = model.score(X_scaled, y)

            logger.info(f"Trained {model_name}: MSE={mse:.4f}, R²={r2:.3f}")

        except Exception as e:
            logger.error(f"Error training model for {strategy}: {e}")

    def _get_default_prediction(self, strategy: str) -> dict[str, float]:
        """Get default prediction when insufficient data."""
        return {
            "expected_profit_pct": 0.0,
            "confidence": 0.1,
            "model_predictions": {},
            "market_regime": "unknown",
            "regime_confidence": 0.0,
        }

    def get_learning_summary(self) -> dict[str, Any]:
        """Get summary of learning progress."""
        return {
            "total_trades": self.trade_count,
            "models_trained": len(self.models),
            "last_update": self.last_model_update.isoformat(),
            "min_trades_for_learning": self.min_trades_for_learning,
            "performance_history_length": len(self.performance_history),
        }

    def get_strategy_performance(self, strategy: str) -> dict[str, float]:
        """Get performance summary for a specific strategy."""
        try:
            strategy_trades = [t for t in self.performance_history if t.strategy == strategy]

            if not strategy_trades:
                return {}

            profits = [t.profit_loss_pct for t in strategy_trades]

            return {
                "total_trades": len(strategy_trades),
                "avg_profit_pct": float(np.mean(profits)),
                "profit_std": float(np.std(profits)),
                "win_rate": float(np.mean([p > 0 for p in profits])),
                "best_trade": float(max(profits)),
                "worst_trade": float(min(profits)),
            }

        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}
