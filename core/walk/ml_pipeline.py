# core/walk/ml_pipeline.py
"""
ML Pipeline for Alpha v1 model integration with walkforward framework.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class MLPipeline:
    """ML Pipeline that uses Alpha v1 model for predictions."""

    def __init__(self, model_path: str = "artifacts/models/linear_v1.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_config = None
        self.feature_names = None
        self.mu = None
        self.sd = None
        self.current_regime = "ml_model"

        # Load model and config
        self._load_model()
        self._load_feature_config()

    def _load_model(self):
        """Load the trained Alpha v1 model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        logger.info(f"Loaded Alpha v1 model from {self.model_path}")

    def _load_feature_config(self):
        """Load feature configuration."""
        config_path = Path("config/features.yaml")
        if config_path.exists():
            self.feature_config = yaml.safe_load(config_path.read_text())
            self.feature_names = list(self.feature_config["features"].keys())
        else:
            # Fallback to default features
            self.feature_names = [
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "sma_20_minus_50",
                "vol_10d",
                "vol_20d",
                "rsi_14",
                "volu_z_20d",
            ]

        logger.info(f"Using features: {self.feature_names}")

    def fit_transforms(self, idx):
        """Fit feature normalization on training data."""
        # This is handled by the model's StandardScaler
        # We just store the indices for reference
        self.train_indices = idx
        logger.debug(f"Fitted transforms on {len(idx)} training samples")

    def transform(self, idx):
        """Transform features using the model's scaler."""
        # The model's StandardScaler handles normalization
        # We just need to ensure features are in the right order
        return self.X[idx]  # Features should already be normalized by the model

    def fit_model(self, Xtr, ytr, warm=None):
        """Fit the ML model (already trained, just store training data)."""
        self.X_train = Xtr
        self.y_train = ytr

        # Model is already trained, just store training info
        logger.info(f"Model already trained, stored {len(Xtr)} training samples")
        return {"model_type": "alpha_v1", "n_features": Xtr.shape[1]}

    def predict(self, Xte):
        """Generate predictions using Alpha v1 model."""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Ensure features are in the right order
            if hasattr(self.model, "named_steps"):
                # sklearn Pipeline
                predictions = self.model.predict(Xte)
            else:
                # Direct model
                predictions = self.model.predict(Xte)

            # Debug: Log prediction statistics
            logger.info(
                f"Raw predictions: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}, mean={np.mean(predictions):.6f}"
            )
            logger.info(f"Prediction std: {np.std(predictions):.6f}")

            # Convert predictions to signals (-1, 0, 1)
            signals = np.sign(predictions)

            # Apply confidence threshold (optional)
            threshold = 0.001  # Lower threshold to allow more trades
            signals = np.where(np.abs(predictions) < threshold, 0, signals)

            logger.info(f"Generated {len(signals)} predictions, {np.sum(signals != 0)} non-zero")
            logger.info(
                f"Signal distribution: {np.bincount(signals.astype(int) + 1)}"
            )  # +1 to handle -1,0,1
            return signals.astype(np.int8)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to zero signals
            return np.zeros(len(Xte), dtype=np.int8)


def create_ml_pipeline(model_path: str = "artifacts/models/linear_v1.pkl") -> MLPipeline:
    """Factory function to create ML pipeline."""
    return MLPipeline(model_path)
