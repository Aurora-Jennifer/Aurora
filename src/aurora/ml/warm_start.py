"""
Warm-Start Utilities for ML Models

This module provides warm-start capabilities for the ML trading system,
including feature priors, curriculum learning, and checkpoint management.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from experiments.persistence import FeaturePersistenceAnalyzer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WarmStartManager:
    """Manages warm-start capabilities for ML models."""

    def __init__(self, state_dir: str = "state/ml", runs_dir: str = "runs"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = Path(runs_dir)
        self.persistence_analyzer = FeaturePersistenceAnalyzer(runs_dir)

        # Warm-start configuration
        self.warm_start_config = {
            "use_feature_priors": True,
            "use_curriculum_learning": True,
            "use_checkpoint_warm_start": True,
            "ema_span": 3,
            "curriculum_weight_threshold": 0.1,
            "confidence_threshold": 0.3,
        }

    def get_warm_start_configuration(
        self, feature_names: list[str], run_id: str = None
    ) -> dict[str, Any]:
        """Get warm-start configuration for model training."""
        try:
            warm_start_data = self.persistence_analyzer.get_warm_start_data(feature_names, n_runs=5)

            if "error" in warm_start_data:
                logger.warning(f"No warm-start data available: {warm_start_data['error']}")
                return self._get_default_configuration(feature_names)

            # Feature priors
            feature_priors = warm_start_data.get("feature_priors", {})

            # Curriculum data
            curriculum_data = warm_start_data.get("curriculum_data", {})

            # Checkpoint data
            checkpoint_data = self._get_checkpoint_data(run_id) if run_id else None

            return {
                "feature_priors": feature_priors,
                "curriculum_data": curriculum_data,
                "checkpoint_data": checkpoint_data,
                "warm_start_config": self.warm_start_config,
                "recent_runs": warm_start_data.get("recent_runs", []),
                "total_features": warm_start_data.get("total_features", len(feature_names)),
            }

        except Exception as e:
            logger.error(f"Error getting warm-start configuration: {e}")
            return self._get_default_configuration(feature_names)

    def _get_default_configuration(self, feature_names: list[str]) -> dict[str, Any]:
        """Get default configuration when no warm-start data is available."""
        feature_priors = {
            name: {
                "ema_coefficient": 0.0,
                "avg_importance": 0.0,
                "alpha_potential": 0.0,
                "confidence": 0.0,
            }
            for name in feature_names
        }

        return {
            "feature_priors": feature_priors,
            "curriculum_data": {},
            "checkpoint_data": None,
            "warm_start_config": self.warm_start_config,
            "recent_runs": [],
            "total_features": len(feature_names),
        }

    def _get_checkpoint_data(self, run_id: str) -> dict[str, Any] | None:
        """Get checkpoint data for a specific run."""
        try:
            checkpoint_file = self.runs_dir / "checkpoints" / f"{run_id}_checkpoint.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, "rb") as f:
                    return pickle.load(f)  # nosec B301  # trusted local artifact; not user-supplied data
            return None
        except Exception as e:
            logger.error(f"Error loading checkpoint data: {e}")
            return None

    def apply_feature_priors(
        self, model: Ridge, feature_names: list[str], warm_start_config: dict[str, Any]
    ) -> Ridge:
        """Apply feature priors to warm-start the model."""
        try:
            if not warm_start_config.get("use_feature_priors", True):
                return model

            feature_priors = warm_start_config.get("feature_priors", {})
            if not feature_priors:
                return model

            # Get prior coefficients
            prior_coefficients = []
            for feature_name in feature_names:
                prior = feature_priors.get(feature_name, {})
                ema_coeff = prior.get("ema_coefficient", 0.0)
                confidence = prior.get("confidence", 0.0)

                # Apply confidence-weighted prior
                if confidence > self.warm_start_config["confidence_threshold"]:
                    prior_coefficients.append(ema_coeff * confidence)
                else:
                    prior_coefficients.append(0.0)

            # Apply priors to model coefficients
            if hasattr(model, "coef_") and model.coef_ is not None:
                # Blend current coefficients with priors
                alpha = 0.3  # Weight for priors
                model.coef_ = (1 - alpha) * model.coef_ + alpha * np.array(prior_coefficients)

                # Adjust intercept if needed
                if hasattr(model, "intercept_") and model.intercept_ is not None:
                    # Simple adjustment based on average prior
                    avg_prior = np.mean(prior_coefficients)
                    model.intercept_ += avg_prior * 0.1

            logger.info(f"Applied feature priors to model with {len(prior_coefficients)} features")
            return model

        except Exception as e:
            logger.error(f"Error applying feature priors: {e}")
            return model

    def apply_curriculum_learning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        warm_start_config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply curriculum learning based on regime performance."""
        try:
            if not warm_start_config.get("use_curriculum_learning", True):
                return X, y

            curriculum_data = warm_start_config.get("curriculum_data", {})
            if not curriculum_data:
                return X, y

            curriculum_weights = curriculum_data.get("curriculum_weights", {})
            if not curriculum_weights:
                return X, y

            # For now, we'll implement a simple weighting scheme
            # In a full implementation, you'd have regime labels for each sample
            # and would weight samples based on their regime performance

            # Simple approach: weight samples based on overall performance
            sample_weights = np.ones(len(y))

            # Apply weights based on performance quartiles
            performance_quartiles = np.percentile(y, [25, 50, 75])

            for i, performance in enumerate(y):
                if performance < performance_quartiles[0]:
                    # Bottom quartile - underperforming samples get higher weight
                    sample_weights[i] = 1.5
                elif performance > performance_quartiles[2]:
                    # Top quartile - overperforming samples get lower weight
                    sample_weights[i] = 0.7
                else:
                    # Middle quartiles - normal weight
                    sample_weights[i] = 1.0

            # Apply sample weights by duplicating samples
            weighted_indices = []
            for i, weight in enumerate(sample_weights):
                # Duplicate samples based on weight
                num_copies = max(1, int(weight))
                weighted_indices.extend([i] * num_copies)

            X_weighted = X[weighted_indices]
            y_weighted = y[weighted_indices]

            logger.info(f"Applied curriculum learning: {len(X)} -> {len(X_weighted)} samples")
            return X_weighted, y_weighted

        except Exception as e:
            logger.error(f"Error applying curriculum learning: {e}")
            return X, y

    def save_checkpoint(
        self,
        model: Ridge,
        scaler: StandardScaler,
        feature_names: list[str],
        run_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Save model checkpoint for warm-start."""
        try:
            checkpoint_dir = self.runs_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            checkpoint_file = checkpoint_dir / f"{run_id}_checkpoint.pkl"

            checkpoint_data = {
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)

            logger.info(f"Saved checkpoint for run {run_id}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Load model checkpoint for warm-start."""
        try:
            checkpoint_file = self.runs_dir / "checkpoints" / f"{run_id}_checkpoint.pkl"

            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_file}")
                return None

            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)  # nosec B301  # trusted local artifact; not user-supplied data

            logger.info(f"Loaded checkpoint for run {run_id}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def warm_start_model(
        self,
        model: Ridge,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        run_id: str = None,
    ) -> Ridge:
        """Apply warm-start to a model."""
        try:
            # Get warm-start configuration
            warm_start_config = self.get_warm_start_configuration(feature_names, run_id)

            # Apply curriculum learning to training data
            if warm_start_config.get("use_curriculum_learning", True):
                X, y = self.apply_curriculum_learning(X, y, feature_names, warm_start_config)

            # Apply feature priors to model
            if warm_start_config.get("use_feature_priors", True):
                model = self.apply_feature_priors(model, feature_names, warm_start_config)

            # Load checkpoint if available
            if warm_start_config.get("use_checkpoint_warm_start", True) and run_id:
                checkpoint_data = self.load_checkpoint(run_id)
                if checkpoint_data:
                    # Use checkpoint model as starting point
                    checkpoint_model = checkpoint_data["model"]
                    if hasattr(checkpoint_model, "coef_") and checkpoint_model.coef_ is not None:
                        # Blend checkpoint coefficients with current model
                        alpha = 0.5  # Weight for checkpoint
                        if hasattr(model, "coef_") and model.coef_ is not None:
                            model.coef_ = (1 - alpha) * model.coef_ + alpha * checkpoint_model.coef_
                        else:
                            model.coef_ = checkpoint_model.coef_

                        if (
                            hasattr(checkpoint_model, "intercept_")
                            and checkpoint_model.intercept_ is not None
                        ):
                            if hasattr(model, "intercept_") and model.intercept_ is not None:
                                model.intercept_ = (
                                    1 - alpha
                                ) * model.intercept_ + alpha * checkpoint_model.intercept_
                            else:
                                model.intercept_ = checkpoint_model.intercept_

            logger.info(f"Applied warm-start to model with {len(feature_names)} features")
            return model

        except Exception as e:
            logger.error(f"Error applying warm-start: {e}")
            return model

    def get_alpha_generation_features(
        self,
        feature_names: list[str],
        warm_start_config: dict[str, Any],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Get top alpha generation features based on warm-start data."""
        try:
            feature_priors = warm_start_config.get("feature_priors", {})

            # Calculate alpha scores
            alpha_scores = []
            for feature_name in feature_names:
                prior = feature_priors.get(feature_name, {})
                alpha_potential = prior.get("alpha_potential", 0.0)
                confidence = prior.get("confidence", 0.0)

                # Combined score
                alpha_score = alpha_potential * confidence
                alpha_scores.append((feature_name, alpha_score))

            # Sort by alpha score and return top N
            alpha_scores.sort(key=lambda x: x[1], reverse=True)
            return alpha_scores[:top_n]

        except Exception as e:
            logger.error(f"Error getting alpha generation features: {e}")
            return []

    def generate_alpha_features(
        self, feature_names: list[str], warm_start_config: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Generate advanced alpha features based on warm-start data."""
        try:
            feature_priors = warm_start_config.get("feature_priors", {})

            alpha_features = {}

            # Generate feature interaction scores
            interaction_scores = {}
            for i, feature1 in enumerate(feature_names):
                for j, feature2 in enumerate(feature_names):
                    if i < j:  # Avoid duplicates
                        prior1 = feature_priors.get(feature1, {})
                        prior2 = feature_priors.get(feature2, {})

                        # Interaction score based on individual alpha potentials
                        interaction_score = prior1.get("alpha_potential", 0.0) * prior2.get(
                            "alpha_potential", 0.0
                        )

                        if interaction_score > 0.001:  # Threshold for meaningful interactions
                            interaction_name = f"{feature1}_{feature2}_interaction"
                            interaction_scores[interaction_name] = interaction_score

            # Generate regime-specific features
            regime_features = {}
            curriculum_data = warm_start_config.get("curriculum_data", {})
            curriculum_weights = curriculum_data.get("curriculum_weights", {})

            for regime, weight in curriculum_weights.items():
                if weight > self.warm_start_config["curriculum_weight_threshold"]:
                    # Create regime-specific feature weights
                    for feature_name in feature_names:
                        prior = feature_priors.get(feature_name, {})
                        regime_feature_name = f"{feature_name}_{regime}_weighted"
                        regime_features[regime_feature_name] = (
                            prior.get("alpha_potential", 0.0) * weight
                        )

            alpha_features.update(interaction_scores)
            alpha_features.update(regime_features)

            logger.info(f"Generated {len(alpha_features)} alpha features")
            return alpha_features

        except Exception as e:
            logger.error(f"Error generating alpha features: {e}")
            return {}


# Global warm-start manager instance
warm_start_manager = WarmStartManager()
