"""
Global Cross-Sectional Model for Multi-Asset Trading

Trains a single XGBRanker model across all symbols simultaneously,
optimizing for cross-sectional ranking within each date.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class GlobalRanker:
    """Global cross-sectional ranking model."""
    
    def __init__(self, 
                 horizon: int = 5,
                 n_estimators: int = 600,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        """Initialize global ranker."""
        self.horizon = horizon
        self.model = None
        self.scaler = None
        self.features = None
        self.symbol_embeddings = None
        
        # XGBRanker parameters
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'tree_method': 'hist',
            'random_state': random_state,
            'verbosity': 0
        }
    
    def _make_groups(self, df: pd.DataFrame) -> List[int]:
        """Create group sizes for XGBRanker (one group per date)."""
        return df.groupby('date').size().astype(int).tolist()
    
    def _create_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add symbol-specific features."""
        df = df.copy()
        
        # Symbol embeddings (one-hot for now, can be learned later)
        symbol_dummies = pd.get_dummies(df['symbol'], prefix='symbol')
        df = pd.concat([df, symbol_dummies], axis=1)
        
        # Sector features (simplified)
        sector_map = {
            'AAPL': 'tech', 'NVDA': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 
            'META': 'tech', 'TSLA': 'tech',
            'JPM': 'financial', 'BAC': 'financial', 'WFC': 'financial', 'GS': 'financial',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
            'AMZN': 'consumer', 'WMT': 'consumer', 'KO': 'consumer', 'PG': 'consumer',
            'XOM': 'energy', 'CVX': 'energy',
            'COIN': 'crypto'
        }
        
        df['sector'] = df['symbol'].map(sector_map).fillna('other')
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Prepare features for training."""
        # Add symbol features
        df = self._create_symbol_features(df)
        
        # Combine base features with symbol features
        all_features = feature_cols + [col for col in df.columns if col.startswith(('symbol_', 'sector_'))]
        
        # Ensure all features exist
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            all_features = [f for f in all_features if f in df.columns]
        
        # Keep all columns including target columns
        keep_cols = ['date', 'symbol'] + all_features
        # Add any target columns that exist
        target_cols = [col for col in df.columns if col.startswith(('ret_fwd_', 'excess_ret_fwd_'))]
        keep_cols.extend(target_cols)
        
        return df[keep_cols], all_features
    
    def fit(self, 
            train_df: pd.DataFrame, 
            valid_df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'ret_fwd_5') -> Dict:
        """Train the global ranker."""
        logger.info(f"Training global ranker for horizon {self.horizon}")
        
        # Prepare features
        train_processed, self.features = self._prepare_features(train_df, feature_cols)
        valid_processed, _ = self._prepare_features(valid_df, feature_cols)
        
        # Clean data (replace inf/nan values)
        train_clean = train_processed[self.features].replace([np.inf, -np.inf], np.nan)
        valid_clean = valid_processed[self.features].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        train_clean = train_clean.fillna(train_clean.median())
        valid_clean = valid_clean.fillna(train_clean.median())  # Use train median for consistency
        
        # Standardize features (train-only)
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_clean)
        X_valid = self.scaler.transform(valid_clean)
        
        # Targets
        y_train = train_processed[target_col].values
        y_valid = valid_processed[target_col].values
        
        # Create groups (one per date)
        train_groups = self._make_groups(train_processed)
        valid_groups = self._make_groups(valid_processed)
        
        logger.info(f"Training data: {len(X_train)} rows, {len(self.features)} features")
        logger.info(f"Validation data: {len(X_valid)} rows")
        logger.info(f"Train groups: {len(train_groups)}, Valid groups: {len(valid_groups)}")
        
        # Train XGBRegressor (we'll use ranking loss later)
        self.model = xgb.XGBRegressor(**self.params)
        
        start_time = time.perf_counter()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        fit_time = time.perf_counter() - start_time
        
        # Get feature importance
        feature_importance = dict(zip(self.features, self.model.feature_importances_))
        
        # Calculate validation metrics
        val_preds = self.model.predict(X_valid)
        val_ic = self._calculate_ic(valid_processed, val_preds)
        
        logger.info(f"Training completed in {fit_time:.2f}s")
        logger.info(f"Validation IC: {val_ic:.4f}")
        
        return {
            'fit_time_seconds': fit_time,
            'n_train_rows': len(X_train),
            'n_valid_rows': len(X_valid),
            'n_features': len(self.features),
            'validation_ic': val_ic,
            'feature_importance': feature_importance,
            'used_trees': getattr(self.model, 'best_ntree_limit', self.params['n_estimators'])
        }
    
    def _calculate_ic(self, df: pd.DataFrame, predictions: np.ndarray) -> float:
        """Calculate cross-sectional Information Coefficient."""
        df = df.copy()
        df['pred'] = predictions
        
        # Calculate IC per date
        ic_by_date = []
        for date, group in df.groupby('date'):
            if len(group) > 1:  # Need at least 2 symbols
                ic = group['pred'].corr(group['ret_fwd_5'])
                if not np.isnan(ic):
                    ic_by_date.append(ic)
        
        return np.mean(ic_by_date) if ic_by_date else 0.0
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        df_processed, _ = self._prepare_features(df, self.features)
        
        # Ensure all features exist and are in the correct order
        missing_features = [f for f in self.features if f not in df_processed.columns]
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            # Fill missing features with zeros
            for f in missing_features:
                df_processed[f] = 0.0
        
        # Reorder features to match training order
        X = df_processed[self.features].values
        
        # Standardize and predict
        X = self.scaler.transform(X)
        predictions = self.model.predict(X)
        
        return predictions
    
    def save(self, path: Path) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(path / "model.json")
        
        # Save scaler and metadata
        import joblib
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.features, path / "features.pkl")
        joblib.dump(self.params, path / "params.pkl")
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load a trained model."""
        import joblib
        
        # Load model (could be XGBRanker or XGBRegressor)
        try:
            self.model = xgb.XGBRanker()
            self.model.load_model(path / "model.json")
        except:
            self.model = xgb.XGBRegressor()
            self.model.load_model(path / "model.json")
        
        # Load scaler and metadata
        self.scaler = joblib.load(path / "scaler.pkl")
        self.features = joblib.load(path / "features.pkl")
        self.params = joblib.load(path / "params.pkl")
        
        logger.info(f"Model loaded from {path}")


def build_panel_dataset(universe_config_path: str, 
                       output_path: str = None,
                       start_date: str = '2020-01-01',
                       end_date: str = '2024-01-01',
                       horizons: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """Build panel dataset with real market features."""
    from ml.panel_builder import build_panel_from_universe
    
    logger.info("Building panel dataset with real market features")
    
    # Build panel from universe configuration
    panel_df = build_panel_from_universe(
        universe_config_path, 
        output_path or "data/panel_dataset.csv",
        start_date, 
        end_date
    )
    
    logger.info(f"Panel dataset created: {len(panel_df)} rows, {len(panel_df.columns)} columns")
    
    return panel_df
