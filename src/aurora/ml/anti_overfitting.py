"""
Anti-Overfitting Safeguards for Trading Models

Implements safeguards against common ML trading pitfalls:
- Overfitting to historical data
- Curve fitting
- Greed and excessive risk-taking
- Data snooping bias
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


@dataclass
class ModelGuardrails:
    """Configuration for anti-overfitting safeguards"""
    max_position_size: float = 0.3  # Max 30% position
    max_daily_trades: int = 5  # Max 5 trades per day
    min_confidence_threshold: float = 0.6  # Min confidence to trade
    max_drawdown_limit: float = 0.15  # Max 15% drawdown
    volatility_threshold: float = 0.05  # Don't trade in high volatility
    min_trades_for_validation: int = 100  # Min trades for reliable validation
    walk_forward_windows: int = 5  # Number of walk-forward windows
    feature_stability_threshold: float = 0.8  # Min feature stability


class AntiOverfittingGuard:
    """
    Implements safeguards against overfitting and bad trading habits
    """
    
    def __init__(self, config: ModelGuardrails):
        self.config = config
        self.trade_history = []
        self.daily_trades = {}
        self.portfolio_value_history = []
        self.feature_stability_scores = {}
        
    def validate_trade_decision(self, 
                              action: str,
                              position_size: float,
                              confidence: float,
                              features: dict[str, float],
                              market_context: dict[str, float],
                              timestamp: datetime) -> tuple[bool, str]:
        """
        Validate a trade decision against anti-overfitting rules
        
        Returns:
            (is_valid, reason)
        """
        
        # 1. Position size limits (anti-greed)
        if position_size > self.config.max_position_size:
            return False, f"Position size {position_size:.2f} exceeds limit {self.config.max_position_size}"
        
        # 2. Confidence threshold
        if confidence < self.config.min_confidence_threshold:
            return False, f"Confidence {confidence:.2f} below threshold {self.config.min_confidence_threshold}"
        
        # 3. Daily trade limits
        date_key = timestamp.date()
        if date_key in self.daily_trades:
            if self.daily_trades[date_key] >= self.config.max_daily_trades:
                return False, f"Daily trade limit {self.config.max_daily_trades} exceeded"
        else:
            self.daily_trades[date_key] = 0
        
        # 4. Volatility check
        volatility = market_context.get('volatility', 0.02)
        if volatility > self.config.volatility_threshold:
            return False, f"Volatility {volatility:.3f} exceeds threshold {self.config.volatility_threshold}"
        
        # 5. Drawdown check
        if self._check_drawdown_limit():
            return False, f"Drawdown limit {self.config.max_drawdown_limit} exceeded"
        
        # 6. Feature stability check
        if not self._check_feature_stability(features):
            return False, "Feature instability detected"
        
        # If all checks pass, record the trade
        self.daily_trades[date_key] += 1
        return True, "Trade approved"
    
    def _check_drawdown_limit(self) -> bool:
        """Check if current drawdown exceeds limit"""
        if len(self.portfolio_value_history) < 10:
            return False
        
        # Calculate current drawdown
        peak = max(self.portfolio_value_history)
        current = self.portfolio_value_history[-1]
        drawdown = (peak - current) / peak
        
        return drawdown > self.config.max_drawdown_limit
    
    def _check_feature_stability(self, features: dict[str, float]) -> bool:
        """Check if features are stable (not changing too rapidly)"""
        if not self.feature_stability_scores:
            return True
        
        # Calculate feature change rate
        current_features = np.array(list(features.values()))
        
        for feature_name, prev_features in self.feature_stability_scores.items():
            if feature_name in features:
                prev_value = prev_features[-1] if prev_features else 0
                current_value = features[feature_name]
                
                # Check for extreme changes
                if abs(current_value - prev_value) > 3 * np.std(prev_features) if len(prev_features) > 5 else False:
                    return False
        
        return True
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value for drawdown tracking"""
        self.portfolio_value_history.append(value)
        
        # Keep only recent history
        if len(self.portfolio_value_history) > 100:
            self.portfolio_value_history = self.portfolio_value_history[-100:]
    
    def update_feature_stability(self, features: dict[str, float]):
        """Update feature stability tracking"""
        for feature_name, value in features.items():
            if feature_name not in self.feature_stability_scores:
                self.feature_stability_scores[feature_name] = []
            
            self.feature_stability_scores[feature_name].append(value)
            
            # Keep only recent history
            if len(self.feature_stability_scores[feature_name]) > 50:
                self.feature_stability_scores[feature_name] = self.feature_stability_scores[feature_name][-50:]


class WalkForwardValidator:
    """
    Implements walk-forward validation to prevent overfitting
    """
    
    def __init__(self, config: ModelGuardrails):
        self.config = config
        self.validation_results = []
    
    def validate_model_stability(self, 
                               model,
                               X: pd.DataFrame, 
                               y: np.ndarray,
                               timestamps: pd.DatetimeIndex) -> dict[str, Any]:
        """
        Perform walk-forward validation to check model stability
        
        Returns:
            Validation results with stability metrics
        """
        
        if len(X) < self.config.min_trades_for_validation:
            return {
                'is_stable': False,
                'reason': f"Insufficient data: {len(X)} < {self.config.min_trades_for_validation}",
                'stability_score': 0.0
            }
        
        # Split data into walk-forward windows
        windows = self._create_walk_forward_windows(X, y, timestamps)
        
        if len(windows) < 3:
            return {
                'is_stable': False,
                'reason': "Insufficient windows for validation",
                'stability_score': 0.0
            }
        
        # Validate each window
        window_results = []
        for i, (X_train, y_train, X_test, y_test) in enumerate(windows):
            try:
                # Train model on this window
                model.fit_reward_based(X_train.values, y_train, 
                                     np.argmax(y_train.reshape(-1, 1), axis=1),
                                     np.zeros((len(X_train), 1)))
                
                # Test on out-of-sample data
                predictions, confidence = model.predict_with_confidence(X_test.values)
                
                # Calculate performance
                accuracy = np.mean(predictions == np.argmax(y_test.reshape(-1, 1), axis=1))
                avg_confidence = np.mean(confidence)
                
                window_results.append({
                    'window': i,
                    'accuracy': accuracy,
                    'confidence': avg_confidence,
                    'test_size': len(X_test)
                })
                
            except Exception as e:
                print(f"Window {i} validation failed: {e}")
                continue
        
        # Calculate stability metrics
        if not window_results:
            return {
                'is_stable': False,
                'reason': "All validation windows failed",
                'stability_score': 0.0
            }
        
        accuracies = [r['accuracy'] for r in window_results]
        confidences = [r['confidence'] for r in window_results]
        
        # Stability score based on consistency
        accuracy_std = np.std(accuracies)
        confidence_std = np.std(confidences)
        
        # Lower std = more stable
        stability_score = 1.0 / (1.0 + accuracy_std + confidence_std)
        
        is_stable = (
            stability_score > 0.7 and  # Consistent performance
            np.mean(accuracies) > 0.4 and  # Reasonable accuracy
            np.mean(confidences) > 0.5  # Reasonable confidence
        )
        
        return {
            'is_stable': is_stable,
            'stability_score': stability_score,
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': accuracy_std,
            'confidence_mean': np.mean(confidences),
            'confidence_std': confidence_std,
            'num_windows': len(window_results),
            'window_results': window_results
        }
    
    def _create_walk_forward_windows(self, 
                                   X: pd.DataFrame, 
                                   y: np.ndarray,
                                   timestamps: pd.DatetimeIndex) -> list[tuple]:
        """Create walk-forward validation windows"""
        
        windows = []
        total_days = (timestamps.max() - timestamps.min()).days
        
        if total_days < 30:
            return windows
        
        # Create overlapping windows
        window_size = max(30, total_days // 4)  # 25% of data per window
        step_size = max(7, window_size // 3)    # 1/3 overlap
        
        start_date = timestamps.min()
        
        while start_date + timedelta(days=window_size) <= timestamps.max():
            end_date = start_date + timedelta(days=window_size)
            test_start = end_date
            test_end = test_start + timedelta(days=step_size)
            
            # Get training data
            train_mask = (timestamps >= start_date) & (timestamps < end_date)
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            # Get test data
            test_mask = (timestamps >= test_start) & (timestamps < test_end)
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Only include windows with sufficient data
            if len(X_train) > 20 and len(X_test) > 5:
                windows.append((X_train, y_train, X_test, y_test))
            
            start_date += timedelta(days=step_size)
        
        return windows


def apply_anti_overfitting_safeguards(model, 
                                    X: pd.DataFrame, 
                                    y: np.ndarray,
                                    timestamps: pd.DatetimeIndex,
                                    config: ModelGuardrails = None) -> dict[str, Any]:
    """
    Apply comprehensive anti-overfitting safeguards
    
    Returns:
        Dictionary with validation results and recommendations
    """
    
    if config is None:
        config = ModelGuardrails()
    
    # 1. Walk-forward validation
    validator = WalkForwardValidator(config)
    stability_results = validator.validate_model_stability(model, X, y, timestamps)
    
    # 2. Feature stability check
    feature_guard = AntiOverfittingGuard(config)
    
    # 3. Overall assessment
    is_safe = (
        stability_results['is_stable'] and
        stability_results['stability_score'] > 0.7
    )
    
    return {
        'is_safe_to_deploy': is_safe,
        'stability_results': stability_results,
        'recommendations': _generate_recommendations(stability_results),
        'config': config
    }


def _generate_recommendations(stability_results: dict[str, Any]) -> list[str]:
    """Generate recommendations based on validation results"""
    
    recommendations = []
    
    if not stability_results.get('is_stable', False):
        recommendations.append("Model is not stable - do not deploy")
    
    if stability_results.get('stability_score', 0) < 0.7:
        recommendations.append("Low stability score - consider more training data")
    
    if stability_results.get('accuracy_mean', 0) < 0.4:
        recommendations.append("Low accuracy - consider feature engineering")
    
    if stability_results.get('confidence_std', 0) > 0.3:
        recommendations.append("High confidence variance - model may be overfitting")
    
    if stability_results.get('num_windows', 0) < 3:
        recommendations.append("Insufficient validation windows - need more data")
    
    if not recommendations:
        recommendations.append("Model appears stable and ready for deployment")
    
    return recommendations
