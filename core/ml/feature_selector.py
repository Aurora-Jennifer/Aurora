"""
Feature Selection for Trading Models

Implements feature selection to avoid overfitting and improve model performance.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TradingFeatureSelector:
    """
    Selects the most predictive features for trading models
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.selected_features = []
        self.feature_scores = {}
        self.scaler = StandardScaler()
        
    def select_features(self, X: pd.DataFrame, y: np.ndarray, 
                       method: str = 'mutual_info', top_k: int = 25) -> list[str]:
        """
        Select top K most predictive features
        
        Args:
            X: Feature matrix
            y: Target variable (rewards)
            method: Selection method ('mutual_info', 'f_score', 'correlation')
            top_k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        
        # Remove features with low variance
        X_clean = self._remove_low_variance_features(X)
        
        # Handle missing values
        X_clean = X_clean.fillna(X_clean.median())
        
        # Select features based on method
        if method == 'mutual_info':
            selected = self._select_by_mutual_info(X_clean, y, top_k)
        elif method == 'f_score':
            selected = self._select_by_f_score(X_clean, y, top_k)
        elif method == 'correlation':
            selected = self._select_by_correlation(X_clean, y, top_k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Remove highly correlated features
        selected = self._remove_correlated_features(X_clean[selected], threshold=0.95)
        
        self.selected_features = selected
        return selected
    
    def _remove_low_variance_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with very low variance"""
        variances = X.var()
        high_var_features = variances[variances > threshold].index
        return X[high_var_features]
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: np.ndarray, top_k: int) -> list[str]:
        """Select features using mutual information"""
        try:
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Get top K features
            feature_scores = pd.Series(mi_scores, index=X.columns)
            top_features = feature_scores.nlargest(top_k).index.tolist()
            
            # Store scores
            self.feature_scores = dict(zip(top_features, feature_scores[top_features], strict=False))
            
            return top_features
            
        except Exception as e:
            print(f"Mutual info selection failed: {e}, using correlation instead")
            return self._select_by_correlation(X, y, top_k)
    
    def _select_by_f_score(self, X: pd.DataFrame, y: np.ndarray, top_k: int) -> list[str]:
        """Select features using F-score"""
        try:
            selector = SelectKBest(score_func=f_regression, k=top_k)
            selector.fit(X, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store scores
            self.feature_scores = dict(zip(selected_features, selector.scores_[selector.get_support()], strict=False))
            
            return selected_features
            
        except Exception as e:
            print(f"F-score selection failed: {e}, using correlation instead")
            return self._select_by_correlation(X, y, top_k)
    
    def _select_by_correlation(self, X: pd.DataFrame, y: np.ndarray, top_k: int) -> list[str]:
        """Select features using correlation with target"""
        try:
            # Calculate correlation with target
            correlations = X.corrwith(pd.Series(y, index=X.index)).abs()
            
            # Get top K features
            top_features = correlations.nlargest(top_k).index.tolist()
            
            # Store scores
            self.feature_scores = dict(zip(top_features, correlations[top_features], strict=False))
            
            return top_features
            
        except Exception as e:
            print(f"Correlation selection failed: {e}, using all features")
            return X.columns.tolist()[:top_k]
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
        """Remove highly correlated features"""
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Remove correlated features
            selected_features = [col for col in X.columns if col not in to_drop]
            
            return selected_features
            
        except Exception as e:
            print(f"Correlation removal failed: {e}, keeping all features")
            return X.columns.tolist()
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores"""
        return self.feature_scores
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using selected features"""
        if not self.selected_features:
            return X
        
        # Select only the chosen features
        X_selected = X[self.selected_features]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            index=X_selected.index,
            columns=X_selected.columns
        )
        
        return X_scaled


def select_trading_features(features: pd.DataFrame, rewards: np.ndarray, 
                          method: str = 'mutual_info', top_k: int = 25) -> tuple[pd.DataFrame, list[str]]:
    """
    Convenience function to select trading features
    
    Args:
        features: Feature matrix
        rewards: Reward values
        method: Selection method
        top_k: Number of features to select
        
    Returns:
        Tuple of (selected_features, feature_names)
    """
    
    selector = TradingFeatureSelector({})
    selected_names = selector.select_features(features, rewards, method, top_k)
    selected_features = selector.transform_features(features)
    
    return selected_features, selected_names
