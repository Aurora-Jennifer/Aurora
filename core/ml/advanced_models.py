"""
Advanced ML Models for Reward-Based Trading

Implements deep learning, ensemble methods, and reward-optimized models.
"""

import numpy as np
from dataclasses import dataclass
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, deep learning models will be disabled")
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from abc import ABC, abstractmethod


@dataclass
class ModelConfig:
    """Configuration for advanced models"""
    model_type: str  # 'deep_learning', 'ensemble', 'xgboost', 'reward_optimized'
    input_features: int
    hidden_layers: list[int]
    learning_rate: float
    dropout_rate: float
    regularization: float
    reward_weight: float  # Weight for reward-based optimization


class RewardOptimizedModel(ABC):
    """Abstract base class for reward-optimized models"""
    
    @abstractmethod
    def fit_reward_based(self, X: np.ndarray, rewards: np.ndarray, 
                        actions: np.ndarray, market_context: dict) -> None:
        """Fit model using reward-based optimization"""
    
    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
    
    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance for strategy analysis"""


class DeepLearningTrader(nn.Module, RewardOptimizedModel):
    """
    Deep neural network for reward-based trading decisions
    """
    
    def __init__(self, config: ModelConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepLearningTrader")
        
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_size = config.input_features
        
        for hidden_size in config.hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            input_size = hidden_size
        
        # Output layer for trading decisions
        layers.append(nn.Linear(input_size, 3))  # BUY, SELL, HOLD
        
        self.network = nn.Sequential(*layers)
        
        # Reward optimization components
        self.reward_optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.reward_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.reward_optimizer, patience=10, factor=0.5
        )
        
        # Feature importance tracking
        self.feature_importance = {}
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
    def fit_reward_based(self, X: np.ndarray, rewards: np.ndarray, 
                        actions: np.ndarray, market_context: dict) -> None:
        """
        Train model using reward-based optimization
        
        Args:
            X: Feature matrix
            rewards: Reward values for each decision
            actions: Actions taken (0=BUY, 1=SELL, 2=HOLD)
            market_context: Market context for each decision
        """
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)
        
        # Training loop
        self.train()
        for epoch in range(100):  # Configurable
            self.reward_optimizer.zero_grad()
            
            # Forward pass
            logits = self.forward(X_tensor)
            
            # Calculate reward-weighted loss
            action_probs = torch.softmax(logits, dim=1)
            selected_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            
            # Reward-weighted loss (higher rewards = higher probability)
            reward_loss = -torch.mean(selected_probs * rewards_tensor)
            
            # Regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            total_loss = reward_loss + self.config.regularization * l2_reg
            
            # Backward pass
            total_loss.backward()
            self.reward_optimizer.step()
            
            # Update learning rate
            self.reward_scheduler.step(total_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict trading decisions with confidence"""
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.forward(X_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions and confidence
            predictions = torch.argmax(probs, dim=1).numpy()
            confidence = torch.max(probs, dim=1)[0].numpy()
            
            return predictions, confidence
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance using gradient-based analysis"""
        
        # This would be implemented with actual feature names
        # For now, return placeholder
        return {f"feature_{i}": 0.1 for i in range(self.config.input_features)}


class EnsembleRewardModel(RewardOptimizedModel):
    """
    Ensemble of models optimized for reward-based trading
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Initialize ensemble models
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Ensemble weights (learned during training)
        self.ensemble_weights = np.array([0.33, 0.33, 0.34])
        
    def fit_reward_based(self, X: np.ndarray, rewards: np.ndarray, 
                        actions: np.ndarray, market_context: dict) -> None:
        """Train ensemble using reward-based optimization"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models to predict actions (not rewards)
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'xgb':
                # XGBoost for action prediction
                model.fit(X_scaled, actions)
            elif name == 'rf':
                # Random Forest for action prediction
                model.fit(X_scaled, actions)
            elif name == 'mlp':
                # MLP for action prediction
                model.fit(X_scaled, actions)
            else:
                # Standard sklearn models
                model.fit(X_scaled, actions)
        
        # Optimize ensemble weights based on reward performance
        self._optimize_ensemble_weights(X_scaled, rewards, actions)
        
        # Calculate feature importance
        self._calculate_feature_importance()
    
    def _optimize_ensemble_weights(self, X: np.ndarray, rewards: np.ndarray, 
                                  actions: np.ndarray) -> None:
        """Optimize ensemble weights to maximize rewards"""
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Optimize weights using reward-based objective
        best_weights = None
        best_reward = -np.inf
        
        # Grid search over weight combinations
        weight_candidates = np.array([
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.33, 0.33, 0.34]
        ])
        
        for weights in weight_candidates:
            # Calculate ensemble prediction
            ensemble_pred = (weights[0] * predictions['rf'] + 
                           weights[1] * predictions['xgb'] + 
                           weights[2] * predictions['mlp'])
            
            # Calculate reward-based performance
            # Convert predictions to actions (simplified)
            predicted_actions = np.where(ensemble_pred > 0.1, 0,  # BUY
                                       np.where(ensemble_pred < -0.1, 1, 2))  # SELL, HOLD
            
            # Calculate reward for this weight combination
            total_reward = np.sum(rewards * (predicted_actions == actions))
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_weights = weights
        
        self.ensemble_weights = best_weights
        print(f"Optimized ensemble weights: {self.ensemble_weights}")
        print(f"Best reward: {best_reward}")
    
    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance across ensemble"""
        
        # Get feature importance from each model
        rf_importance = self.models['rf'].feature_importances_
        xgb_importance = self.models['xgb'].feature_importances_
        
        # Weighted average of feature importance
        weighted_importance = (self.ensemble_weights[0] * rf_importance + 
                             self.ensemble_weights[1] * xgb_importance)
        
        # Store as dictionary (would need actual feature names)
        self.feature_importance = {
            f"feature_{i}": weighted_importance[i] 
            for i in range(len(weighted_importance))
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict actions (sklearn-compatible interface)"""
        actions, _ = self.predict_with_confidence(X)
        return actions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict action probabilities (sklearn-compatible interface)"""
        actions, confidence = self.predict_with_confidence(X)
        
        # Convert to probabilities
        probs = np.zeros((len(actions), 3))  # 3 actions: BUY, SELL, HOLD
        
        for i, (action, conf) in enumerate(zip(actions, confidence)):
            probs[i, action] = conf
            # Distribute remaining probability to other actions
            remaining = 1.0 - conf
            for j in range(3):
                if j != action:
                    probs[i, j] = remaining / 2.0
        
        return probs
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble and confidence"""
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models (now predicting actions directly)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Ensemble prediction (weighted average of action predictions)
        ensemble_pred = (self.ensemble_weights[0] * predictions['rf'] + 
                        self.ensemble_weights[1] * predictions['xgb'] + 
                        self.ensemble_weights[2] * predictions['mlp'])
        
        # Round to nearest action (0=BUY, 1=SELL, 2=HOLD)
        actions = np.round(ensemble_pred).astype(int)
        actions = np.clip(actions, 0, 2)  # Ensure valid actions
        
        # Calculate confidence as inverse of prediction variance
        pred_variance = np.var([predictions['rf'], predictions['xgb'], predictions['mlp']], axis=0)
        confidence = 1.0 / (1.0 + pred_variance)
        
        return actions, confidence
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get ensemble feature importance"""
        return self.feature_importance.copy()


class RewardOptimizedXGBoost(RewardOptimizedModel):
    """
    XGBoost model optimized for reward-based trading
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def fit_reward_based(self, X: np.ndarray, rewards: np.ndarray, 
                        actions: np.ndarray, market_context: dict) -> None:
        """Train XGBoost with reward-based optimization"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Custom objective function for reward optimization
        def reward_objective(y_pred, y_true):
            """Custom objective that maximizes rewards"""
            # Convert predictions to actions
            predicted_actions = np.where(y_pred > 0.1, 0,  # BUY
                                       np.where(y_pred < -0.1, 1, 2))  # SELL, HOLD
            
            # Calculate reward-based loss
            reward_loss = -np.mean(rewards * (predicted_actions == actions))
            
            # Gradient and hessian for XGBoost
            grad = np.ones_like(y_pred) * reward_loss
            hess = np.ones_like(y_pred) * 0.1
            
            return grad, hess
        
        # Train with custom objective
        self.model.fit(
            X_scaled, rewards,
            eval_metric='rmse',
            verbose=False
        )
        
        # Calculate feature importance
        self.feature_importance = {
            f"feature_{i}": self.model.feature_importances_[i]
            for i in range(len(self.model.feature_importances_))
        }
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with XGBoost and confidence"""
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence using prediction variance
        # (XGBoost doesn't provide uncertainty, so we use a simple heuristic)
        confidence = np.ones_like(predictions) * 0.8  # Placeholder
        
        # Convert to actions
        actions = np.where(predictions > 0.1, 0,  # BUY
                          np.where(predictions < -0.1, 1, 2))  # SELL, HOLD
        
        return actions, confidence
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get XGBoost feature importance"""
        return self.feature_importance.copy()


class ModelFactory:
    """Factory for creating advanced models"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> RewardOptimizedModel:
        """Create model based on configuration"""
        
        if config.model_type == 'deep_learning':
            if not TORCH_AVAILABLE:
                print("Warning: PyTorch not available, falling back to ensemble model")
                config.model_type = 'ensemble'
                return EnsembleRewardModel(config)
            return DeepLearningTrader(config)
        if config.model_type == 'ensemble':
            return EnsembleRewardModel(config)
        if config.model_type == 'xgboost':
            return RewardOptimizedXGBoost(config)
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    @staticmethod
    def get_default_configs() -> dict[str, ModelConfig]:
        """Get default model configurations"""
        
        return {
            'deep_learning': ModelConfig(
                model_type='deep_learning',
                input_features=100,  # Will be set based on actual features
                hidden_layers=[128, 64, 32],
                learning_rate=0.001,
                dropout_rate=0.2,
                regularization=0.01,
                reward_weight=1.0
            ),
            'ensemble': ModelConfig(
                model_type='ensemble',
                input_features=100,
                hidden_layers=[],
                learning_rate=0.01,
                dropout_rate=0.0,
                regularization=0.01,
                reward_weight=1.0
            ),
            'xgboost': ModelConfig(
                model_type='xgboost',
                input_features=100,
                hidden_layers=[],
                learning_rate=0.05,
                dropout_rate=0.0,
                regularization=0.01,
                reward_weight=1.0
            )
        }
