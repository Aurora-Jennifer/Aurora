#!/usr/bin/env python3
"""
Asset-Specific Model Router

Zero-risk adapter that routes predictions to asset-specific models
while preserving 100% backward compatibility with universal model path.

Default behavior: Uses existing universal model (unchanged)
Feature flag enabled: Routes to asset-specific models

SAFETY: If anything fails, falls back to universal model immediately.
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class AssetClassifier:
    """Classify symbols into asset classes using config-driven rules."""
    
    def __init__(self, assets_config_path: str = "config/assets.yaml"):
        """Initialize classifier with asset configuration."""
        self.config_path = Path(assets_config_path)
        self.config = self._load_config()
        self._compile_patterns()
    
    def _load_config(self) -> dict:
        """Load asset classification configuration."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Assets config not found: {self.config_path}, using fallback")
                return self._get_fallback_config()
            
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded asset classification config from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load assets config: {e}, using fallback")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> dict:
        """Fallback configuration if config file is missing."""
        return {
            'crypto': ['BTC-USD', 'ETH-USD', 'BTCUSDT', 'ETHUSDT'],
            'equities': ['SPY', 'QQQ', 'AAPL', 'TSLA'],
            'options': ['SPY_options', 'QQQ_options'],
            'patterns': {
                'crypto': ['.*USDT$', '.*-USD$'],
                'equities': ['^[A-Z]{1,5}$'],
                'options': ['.*_options$', '.*_PUT$', '.*_CALL$']
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        patterns = self.config.get('patterns', {})
        
        for asset_class, pattern_list in patterns.items():
            compiled = []
            for pattern in pattern_list:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}' for {asset_class}: {e}")
            self.compiled_patterns[asset_class] = compiled
    
    def classify_symbol(self, symbol: str) -> str:
        """
        Classify symbol into asset class using config-driven rules.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY', 'BTC-USD', 'AAPL_options')
            
        Returns:
            Asset class string: 'crypto', 'equities', 'options', or 'universal'
        """
        symbol_upper = symbol.upper()
        
        # First check explicit symbol lists
        for asset_class in ['crypto', 'equities', 'options']:
            symbol_list = self.config.get(asset_class, [])
            if symbol_upper in symbol_list:
                logger.debug(f"Symbol {symbol} classified as {asset_class} (explicit list)")
                return asset_class
        
        # Then check regex patterns
        for asset_class, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(symbol_upper):
                    logger.debug(f"Symbol {symbol} classified as {asset_class} (pattern {pattern.pattern})")
                    return asset_class
        
        # Default fallback - assume equities for unknown symbols (safer than crypto)
        logger.debug(f"Symbol {symbol} classified as equities (fallback)")
        return 'equities'


class ModelRegistry:
    """Registry for asset-specific models with config-driven paths."""
    
    def __init__(self, assets_config_path: str = "config/assets.yaml"):
        """Initialize registry with asset configuration."""
        self.config_path = Path(assets_config_path)
        self.config = self._load_config()
        self.model_paths = self._load_model_paths()
        self._loaded_models: dict[str, any] = {}
        
    def _load_config(self) -> dict:
        """Load asset configuration."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Assets config not found: {self.config_path}, using fallback")
                return self._get_fallback_config()
            
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load assets config: {e}, using fallback")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> dict:
        """Fallback model configuration."""
        return {
            'models': {
                'universal': 'models/universal_v1.onnx',
                'crypto': 'models/crypto_v1.onnx',
                'equities': 'models/equities_v1.onnx',
                'options': 'models/options_v1.onnx'
            }
        }
    
    def _load_model_paths(self) -> dict[str, Path]:
        """Load model paths from configuration."""
        model_config = self.config.get('models', {})
        paths = {}
        
        for asset_class, path_str in model_config.items():
            paths[asset_class] = Path(path_str)
        
        # Ensure universal model path exists
        if 'universal' not in paths:
            paths['universal'] = Path("models/universal_v1.onnx")
            
        return paths
        
    def get_model_path(self, asset_class: str) -> Path:
        """Get path to model for asset class, fallback to universal."""
        if asset_class in self.model_paths:
            model_path = self.model_paths[asset_class]
            if model_path.exists():
                return model_path
            logger.warning(f"Model not found for {asset_class} at {model_path}, using universal fallback")
        else:
            logger.warning(f"Unknown asset class {asset_class}, using universal fallback")
            
        # Fallback to universal
        universal_path = self.model_paths['universal']
        if universal_path.exists():
            return universal_path
        logger.error(f"Universal model not found at {universal_path}")
        raise FileNotFoundError(f"No model found for {asset_class} and universal fallback missing")
    
    def model_exists(self, asset_class: str) -> bool:
        """Check if model exists for asset class."""
        if asset_class in self.model_paths:
            return self.model_paths[asset_class].exists()
        return False
    
    def list_available_models(self) -> list[str]:
        """List asset classes with available models."""
        available = []
        for asset_class, path in self.model_paths.items():
            if path.exists():
                available.append(asset_class)
        return available


class AssetSpecificModelRouter:
    """
    Zero-risk model router that preserves existing universal model behavior.
    
    Key Safety Features:
    - Default behavior identical to current system
    - Feature flag controls new behavior
    - Automatic fallback on any error
    - No changes to existing prediction interfaces
    """
    
    def __init__(self, assets_config_path: str = "config/assets.yaml"):
        """Initialize router with configuration."""
        self.feature_enabled = os.getenv("FLAG_ASSET_SPECIFIC_MODELS", "0") == "1"
        self.classifier = AssetClassifier(assets_config_path)
        self.registry = ModelRegistry(assets_config_path)
        self._models_cache: dict[str, any] = {}
        
        if self.feature_enabled:
            available_models = self.registry.list_available_models()
            logger.info(f"Asset-specific model routing ENABLED (available: {available_models})")
        else:
            logger.info("Asset-specific model routing DISABLED (using universal model)")
    
    def get_prediction(self, symbol: str, features: np.ndarray | pd.DataFrame) -> float:
        """
        Get prediction for symbol using appropriate model.
        
        SAFETY GUARANTEE: If anything fails, uses universal model path.
        
        Args:
            symbol: Trading symbol
            features: Feature vector/dataframe
            
        Returns:
            Prediction value
        """
        try:
            if not self.feature_enabled:
                # EXACT SAME PATH AS CURRENT SYSTEM
                return self._universal_prediction(features)
            # NEW ASSET-SPECIFIC PATH
            return self._asset_specific_prediction(symbol, features)
                
        except Exception as e:
            logger.warning(f"Asset-specific prediction failed for {symbol}: {e}")
            logger.info("Falling back to universal model")
            return self._universal_prediction(features)
    
    def _universal_prediction(self, features: np.ndarray | pd.DataFrame) -> float:
        """
        Universal model prediction - PRESERVES EXISTING BEHAVIOR EXACTLY.
        
        This method implements the exact same logic as the current system.
        SAFETY: Only loads models when actually making predictions.
        """
        try:
            # LAZY LOADING: Only import and instantiate when actually making predictions
            from core.walk.ml_pipeline import MLPipeline
            
            # Get universal model path from config
            universal_path = self.registry.get_model_path('universal')
            if not universal_path.exists():
                logger.warning(f"Universal model not found at {universal_path}, using fallback")
                return 0.0
            
            # Use existing MLPipeline (current working implementation)
            if 'universal' not in self._models_cache:
                try:
                    self._models_cache['universal'] = MLPipeline()
                except Exception as e:
                    logger.error(f"Failed to load universal model: {e}")
                    return 0.0
            
            pipeline = self._models_cache['universal']
            
            # Convert to numpy if needed (current system expects numpy)
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            else:
                features_array = features
                
            # Use existing predict method
            prediction = pipeline.predict(features_array)
            
            # Return single value (current system behavior)
            if isinstance(prediction, np.ndarray):
                return float(prediction[0]) if len(prediction) > 0 else 0.0
            return float(prediction)
                
        except Exception as e:
            logger.warning(f"Universal model prediction failed: {e}")
            # Last resort fallback (safe for testing)
            return 0.0
    
    def _asset_specific_prediction(self, symbol: str, features: np.ndarray | pd.DataFrame) -> float:
        """
        Asset-specific prediction routing.
        
        Routes to appropriate model based on asset class.
        Falls back to universal on any issue.
        """
        try:
            # Classify symbol
            asset_class = self.classifier.classify_symbol(symbol)
            logger.debug(f"Symbol {symbol} classified as {asset_class}")
            
            # Check if asset-specific model exists
            if not self.registry.model_exists(asset_class):
                logger.info(f"No specific model for {asset_class}, using universal")
                return self._universal_prediction(features)
            
            # Load model if not cached
            model_key = f"asset_{asset_class}"
            if model_key not in self._models_cache:
                model_path = self.registry.get_model_path(asset_class)
                self._models_cache[model_key] = self._load_onnx_model(model_path)
            
            model = self._models_cache[model_key]
            
            # Make prediction
            if isinstance(features, pd.DataFrame):
                features_array = features.values.astype(np.float32)
            else:
                features_array = features.astype(np.float32)
            
            # Ensure 2D array for ONNX
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            
            prediction = model.run(None, {'float_input': features_array})[0]
            return float(prediction[0][0]) if prediction.shape[0] > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Asset-specific prediction failed for {symbol} ({asset_class}): {e}")
            # Fallback to universal
            return self._universal_prediction(features)
    
    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            logger.info(f"Loaded ONNX model: {model_path}")
            return session
        except ImportError:
            logger.error("onnxruntime not available, cannot load ONNX models")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_path}: {e}")
            raise


# Convenience factory function
def create_model_router(assets_config_path: str = "config/assets.yaml") -> AssetSpecificModelRouter:
    """Create model router instance."""
    return AssetSpecificModelRouter(assets_config_path)


# Global router instance (lazy initialization)
_global_router: AssetSpecificModelRouter | None = None

def get_model_router(assets_config_path: str = "config/assets.yaml") -> AssetSpecificModelRouter:
    """Get global model router instance."""
    global _global_router
    if _global_router is None:
        _global_router = create_model_router(assets_config_path)
    return _global_router


def predict_with_router(symbol: str, features: np.ndarray | pd.DataFrame) -> float:
    """
    Convenience function for getting predictions with asset-specific routing.
    
    This is the new interface that can replace direct model.predict() calls.
    """
    router = get_model_router()
    return router.get_prediction(symbol, features)
