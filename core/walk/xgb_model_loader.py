"""
XGBoost model loader for paper trading.
"""

import os
import json
import logging
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class XGBModelLoader:
    """Loads and manages XGBoost models for paper trading."""
    
    def __init__(self, model_path: str, features_path: str):
        """
        Initialize the XGBoost model loader.
        
        Args:
            model_path: Path to the XGBoost model JSON file
            features_path: Path to the features whitelist JSON file
        """
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.model: Optional[xgb.Booster] = None
        self.features_whitelist: List[str] = []
        self.features_config: Dict[str, Any] = {}
        
        self._load_model()
        self._load_features()
    
    def _load_model(self) -> None:
        """Load the XGBoost model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_path))
            logger.info(f"✅ XGBoost model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load XGBoost model: {e}")
            raise
    
    def _load_features(self) -> None:
        """Load the features whitelist configuration."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        try:
            with open(self.features_path, 'r') as f:
                self.features_config = json.load(f)
            
            self.features_whitelist = self.features_config['feature_cols']
            logger.info(f"✅ Loaded {len(self.features_whitelist)} features from whitelist")
        except Exception as e:
            logger.error(f"❌ Failed to load features whitelist: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the loaded model.
        
        Args:
            data: DataFrame with features matching the whitelist
            
        Returns:
            Series of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure we have the right features
        missing_features = set(self.features_whitelist) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the whitelisted features in the correct order
        feature_data = data[self.features_whitelist]
        
        # Convert to DMatrix
        dmatrix = xgb.DMatrix(feature_data)
        
        # Make predictions
        predictions = self.model.predict(dmatrix)
        
        # Return as Series with same index as input
        return pd.Series(predictions, index=data.index)
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names expected by the model."""
        return self.features_whitelist.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "features_path": str(self.features_path),
            "num_features": len(self.features_whitelist),
            "feature_names": self.features_whitelist,
            "model_type": "XGBoost",
            "total_panel_cols": self.features_config.get("total_panel_cols", "unknown"),
            "feature_count": self.features_config.get("feature_count", "unknown")
        }
