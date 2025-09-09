"""
Ensemble System for Combining Global and Per-Asset Models

Implements a stacked ensemble that optimally blends predictions from:
1. Global cross-sectional ranker
2. Per-asset models (Ridge/XGB)
3. Deep learning models (optional)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class StackedEnsemble:
    """Stacked ensemble for combining multiple model predictions."""
    
    def __init__(self, 
                 meta_model_type: str = 'ridge',
                 meta_alpha: float = 1.0,
                 non_negative_weights: bool = True,
                 turnover_penalty: float = 0.001):
        """Initialize stacked ensemble."""
        self.meta_model_type = meta_model_type
        self.meta_alpha = meta_alpha
        self.non_negative_weights = non_negative_weights
        self.turnover_penalty = turnover_penalty
        
        self.meta_model = None
        self.model_weights = None
        self.feature_names = None
        
    def _create_meta_features(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Create meta-features from base model predictions."""
        # Stack all predictions horizontally
        meta_features = []
        feature_names = []
        
        for model_name, preds in predictions_dict.items():
            meta_features.append(preds)
            feature_names.append(f"{model_name}_pred")
        
        return np.column_stack(meta_features), feature_names
    
    def _calculate_turnover_penalty(self, weights: np.ndarray, 
                                  previous_weights: Optional[np.ndarray] = None) -> float:
        """Calculate turnover penalty for weight changes."""
        if previous_weights is None:
            return 0.0
        
        if len(weights) != len(previous_weights):
            return 0.0
        
        return np.sum(np.abs(weights - previous_weights))
    
    def fit(self, 
            predictions_dict: Dict[str, np.ndarray],
            targets: np.ndarray,
            dates: Optional[np.ndarray] = None,
            previous_weights: Optional[np.ndarray] = None) -> Dict:
        """Fit the meta-model to combine base predictions."""
        logger.info("Training stacked ensemble meta-model")
        
        # Create meta-features
        meta_features, self.feature_names = self._create_meta_features(predictions_dict)
        
        logger.info(f"Meta-features shape: {meta_features.shape}")
        logger.info(f"Feature names: {self.feature_names}")
        
        # Initialize meta-model
        if self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=self.meta_alpha, random_state=42)
        else:
            raise ValueError(f"Unknown meta model type: {self.meta_model_type}")
        
        # Time series cross-validation for robust fitting
        if dates is not None:
            # Use time-based splits
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(meta_features):
                X_train, X_val = meta_features[train_idx], meta_features[val_idx]
                y_train, y_val = targets[train_idx], targets[val_idx]
                
                # Fit on training fold
                self.meta_model.fit(X_train, y_train)
                
                # Evaluate on validation fold
                val_pred = self.meta_model.predict(X_val)
                val_score = np.corrcoef(val_pred, y_val)[0, 1]
                cv_scores.append(val_score)
            
            logger.info(f"CV scores: {cv_scores}")
            logger.info(f"Mean CV score: {np.mean(cv_scores):.4f}")
        
        # Final fit on all data
        self.meta_model.fit(meta_features, targets)
        
        # Get model weights
        self.model_weights = dict(zip(self.feature_names, self.meta_model.coef_))
        
        # Apply non-negative constraint if requested
        if self.non_negative_weights:
            negative_weights = {k: v for k, v in self.model_weights.items() if v < 0}
            if negative_weights:
                logger.warning(f"Negative weights found: {negative_weights}")
                # Set negative weights to zero
                for k in self.model_weights:
                    if self.model_weights[k] < 0:
                        self.model_weights[k] = 0.0
        
        # Normalize weights to sum to 1
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
        
        # Calculate turnover penalty
        if previous_weights is not None:
            turnover = self._calculate_turnover_penalty(
                np.array(list(self.model_weights.values())), 
                previous_weights
            )
            logger.info(f"Turnover penalty: {turnover:.4f}")
        
        logger.info(f"Final model weights: {self.model_weights}")
        
        return {
            'model_weights': self.model_weights,
            'cv_scores': cv_scores if dates is not None else None,
            'meta_model_score': self.meta_model.score(meta_features, targets)
        }
    
    def predict(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet")
        
        # Create meta-features
        meta_features, _ = self._create_meta_features(predictions_dict)
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            # Reorder features to match training order
            ordered_features = []
            for name in self.feature_names:
                model_name = name.replace('_pred', '')
                if model_name in predictions_dict:
                    ordered_features.append(predictions_dict[model_name])
                else:
                    logger.warning(f"Missing prediction for {model_name}")
                    # Fill with zeros
                    ordered_features.append(np.zeros(len(list(predictions_dict.values())[0])))
            
            meta_features = np.column_stack(ordered_features)
        
        # Make predictions
        ensemble_preds = self.meta_model.predict(meta_features)
        
        return ensemble_preds
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from meta-model."""
        if self.meta_model is None:
            return {}
        
        if hasattr(self.meta_model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.meta_model.coef_)))
        else:
            return {}
    
    def save(self, path: Path) -> None:
        """Save the ensemble model."""
        if self.meta_model is None:
            raise ValueError("No model to save")
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save meta-model
        joblib.dump(self.meta_model, path / "meta_model.pkl")
        
        # Save metadata
        metadata = {
            'model_weights': self.model_weights,
            'feature_names': self.feature_names,
            'meta_model_type': self.meta_model_type,
            'meta_alpha': self.meta_alpha,
            'non_negative_weights': self.non_negative_weights,
            'turnover_penalty': self.turnover_penalty
        }
        
        joblib.dump(metadata, path / "metadata.pkl")
        
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load the ensemble model."""
        # Load meta-model
        self.meta_model = joblib.load(path / "meta_model.pkl")
        
        # Load metadata
        metadata = joblib.load(path / "metadata.pkl")
        self.model_weights = metadata['model_weights']
        self.feature_names = metadata['feature_names']
        self.meta_model_type = metadata['meta_model_type']
        self.meta_alpha = metadata['meta_alpha']
        self.non_negative_weights = metadata['non_negative_weights']
        self.turnover_penalty = metadata['turnover_penalty']
        
        logger.info(f"Ensemble model loaded from {path}")


class EnsembleManager:
    """Manages multiple ensemble models and base predictions."""
    
    def __init__(self, models_dir: Path):
        """Initialize ensemble manager."""
        self.models_dir = models_dir
        self.global_models = {}
        self.ensemble_models = {}
        
    def load_global_model(self, horizon: int) -> Optional[object]:
        """Load global model for given horizon."""
        model_path = self.models_dir / f"horizon_{horizon}"
        
        if not model_path.exists():
            logger.warning(f"Global model not found for horizon {horizon}")
            return None
        
        try:
            from ml.global_ranker import GlobalRanker
            model = GlobalRanker()
            model.load(model_path)
            self.global_models[horizon] = model
            return model
        except Exception as e:
            logger.error(f"Error loading global model for horizon {horizon}: {e}")
            return None
    
    def load_ensemble_model(self, horizon: int) -> Optional[StackedEnsemble]:
        """Load ensemble model for given horizon."""
        ensemble_path = self.models_dir / f"ensemble_{horizon}"
        
        if not ensemble_path.exists():
            logger.warning(f"Ensemble model not found for horizon {horizon}")
            return None
        
        try:
            ensemble = StackedEnsemble()
            ensemble.load(ensemble_path)
            self.ensemble_models[horizon] = ensemble
            return ensemble
        except Exception as e:
            logger.error(f"Error loading ensemble model for horizon {horizon}: {e}")
            return None
    
    def predict_ensemble(self, 
                        per_asset_preds: Dict[str, np.ndarray],
                        global_preds: Dict[int, np.ndarray],
                        horizon: int) -> np.ndarray:
        """Make ensemble predictions combining per-asset and global models."""
        # Load ensemble model if not already loaded
        if horizon not in self.ensemble_models:
            self.load_ensemble_model(horizon)
        
        if horizon not in self.ensemble_models:
            logger.warning(f"No ensemble model for horizon {horizon}, using per-asset only")
            # Fallback to per-asset predictions only
            return per_asset_preds.get('ridge', np.zeros(len(list(per_asset_preds.values())[0])))
        
        # Combine predictions
        all_predictions = {}
        
        # Add per-asset predictions
        for model_name, preds in per_asset_preds.items():
            all_predictions[f"per_asset_{model_name}"] = preds
        
        # Add global predictions
        if horizon in global_preds:
            all_predictions["global_ranker"] = global_preds[horizon]
        
        # Make ensemble prediction
        ensemble_pred = self.ensemble_models[horizon].predict(all_predictions)
        
        return ensemble_pred
