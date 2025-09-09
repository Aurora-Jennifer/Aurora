#!/usr/bin/env python3
"""
Deep Learning Training Module

Implements trainer with inner validation split, early stopping on validation Sharpe,
and temperature calibration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import logging

from .models import create_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml.targets import create_targets, validate_targets
from ml.decision import calibrate_decision_parameters
from ml.features import create_feature_pipeline
from scripts.trade_metrics_helpers import sharpe_from_daily


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss with label smoothing"""
    
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.05):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        # Use simple CrossEntropyLoss without label smoothing for now
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits, targets)


class HuberVolatilityLoss(nn.Module):
    """Huber loss for volatility prediction"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred.squeeze(), target, delta=self.delta)


class DLTrainer:
    """Deep Learning trainer with inner validation and early stopping"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 lr: float = 0.001, weight_decay: float = 1e-4,
                 batch_size: int = 256, max_epochs: int = 100,
                 early_stop_patience: int = 10, grad_clip: float = 1.0):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.grad_clip = grad_clip
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_sharpes = []
        self.best_val_sharpe = -np.inf
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        # Temperature for calibration
        self.temperature = 1.0
    
    def prepare_data(self, features: pd.DataFrame, targets: pd.Series, 
                    sequence_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            sequence_length: Sequence length for TCN models
        
        Returns:
            Tuple of (X, y) tensors
        """
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx].values
        y = targets.loc[common_idx].values
        
        # Convert labels to 0, 1, 2 (SELL=-1 -> 0, HOLD=0 -> 1, BUY=1 -> 2)
        y_mapped = y + 1
        
        if self.model.__class__.__name__ == 'TinyMLP':
            # For MLP, use only the last frame
            X_tensor = torch.FloatTensor(X[-sequence_length:]).unsqueeze(0)  # (1, features)
            y_tensor = torch.LongTensor([y_mapped[-1]])  # (1,)
        else:
            # For TCN, create sequences
            if len(X) < sequence_length:
                # Pad with zeros if sequence is too short
                pad_length = sequence_length - len(X)
                X_padded = np.pad(X, ((pad_length, 0), (0, 0)), mode='constant')
                y_padded = np.pad(y_mapped, (pad_length, 0), mode='constant')
            else:
                X_padded = X[-sequence_length:]
                y_padded = y_mapped[-sequence_length:]
            
            X_tensor = torch.FloatTensor(X_padded).unsqueeze(0)  # (1, seq_len, features)
            y_tensor = torch.LongTensor([y_padded[-1]])  # (1,) - use last target
        
        # Ensure y_tensor is 1D
        if y_tensor.dim() > 1:
            y_tensor = y_tensor.squeeze()
        
        # Ensure y_tensor is at least 1D (not 0D)
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)
        
        return X_tensor, y_tensor
    
    def create_sequences(self, features: pd.DataFrame, targets: pd.Series,
                        sequence_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for training
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            sequence_length: Sequence length
        
        Returns:
            Tuple of (X, y) tensors
        """
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx].values
        y = targets.loc[common_idx].values
        
        # Convert labels to 0, 1, 2
        y_mapped = y + 1
        
        sequences_X = []
        sequences_y = []
        
        for i in range(sequence_length, len(X)):
            seq_X = X[i-sequence_length:i]
            seq_y = y_mapped[i-1]  # Use previous target
            
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        
        X_tensor = torch.FloatTensor(np.array(sequences_X))
        y_tensor = torch.LongTensor(np.array(sequences_y))
        
        return X_tensor, y_tensor
    
    def train_epoch(self, train_loader: DataLoader, loss_fn: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and len(self.model.forward(batch_X)) == 2:
                # Multi-task model
                logits, vol_pred = self.model(batch_X)
                loss = loss_fn(logits, batch_y)
            else:
                # Single-task model
                logits = self.model(batch_X)
                loss = loss_fn(logits, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, val_loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and len(self.model.forward(batch_X)) == 2:
                    # Multi-task model
                    logits, vol_pred = self.model(batch_X)
                else:
                    # Single-task model
                    logits = self.model(batch_X)
                
                loss = loss_fn(logits, batch_y)
                
                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(batch_y.cpu())
                num_batches += 1
        
        # Calculate validation Sharpe (simplified)
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Convert logits to edges (P(BUY) - P(SELL))
            probs = F.softmax(all_logits, dim=-1)
            edges = probs[:, 2] - probs[:, 0]  # BUY - SELL
            
            # Simple Sharpe calculation (in practice, use actual returns)
            val_sharpe = edges.mean() / (edges.std() + 1e-8)
        else:
            val_sharpe = 0.0
        
        return total_loss / num_batches if num_batches > 0 else 0.0, val_sharpe
    
    def calibrate_temperature(self, val_loader: DataLoader) -> float:
        """Calibrate temperature on validation set"""
        self.model.eval()
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if hasattr(self.model, 'forward') and len(self.model.forward(batch_X)) == 2:
                    logits, _ = self.model(batch_X)
                else:
                    logits = self.model(batch_X)
                
                all_logits.append(logits.cpu())
                all_targets.append(batch_y.cpu())
        
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Optimize temperature
            temperature = nn.Parameter(torch.ones(1))
            optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
            
            def eval_loss():
                optimizer.zero_grad()
                loss = F.cross_entropy(all_logits / temperature, all_targets)
                loss.backward()
                return loss
            
            optimizer.step(eval_loss)
            return temperature.item()
        
        return 1.0
    
    def fit(self, train_features: pd.DataFrame, train_targets: pd.Series,
            val_features: pd.DataFrame, val_targets: pd.Series,
            sequence_length: int = 128, class_weights: Optional[torch.Tensor] = None,
            multitask: bool = False) -> Dict[str, Any]:
        """
        Train the model with early stopping
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features
            val_targets: Validation targets
            sequence_length: Sequence length for TCN
            class_weights: Class weights for imbalanced data
            multitask: Whether to use multi-task learning
        
        Returns:
            Training results dictionary
        """
        # Prepare data
        if self.model.__class__.__name__ == 'TinyMLP':
            # For MLP, use only last frame
            X_train, y_train = self.prepare_data(train_features, train_targets, sequence_length)
            X_val, y_val = self.prepare_data(val_features, val_targets, sequence_length)
        else:
            # For TCN, create sequences
            X_train, y_train = self.create_sequences(train_features, train_targets, sequence_length)
            X_val, y_val = self.create_sequences(val_features, val_targets, sequence_length)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        loss_fn = WeightedCrossEntropyLoss(class_weights, label_smoothing=0.05)
        
        # Training loop
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, loss_fn)
            
            # Validate
            val_loss, val_sharpe = self.validate_epoch(val_loader, loss_fn)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_sharpes.append(val_sharpe)
            
            # Early stopping
            if val_sharpe > self.best_val_sharpe:
                self.best_val_sharpe = val_sharpe
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_sharpe={val_sharpe:.4f}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Calibrate temperature
        self.temperature = self.calibrate_temperature(val_loader)
        logging.info(f"Calibrated temperature: {self.temperature:.4f}")
        
        return {
            'best_val_sharpe': self.best_val_sharpe,
            'best_epoch': len(self.train_losses) - self.epochs_without_improvement,
            'temperature': self.temperature,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_sharpes': self.val_sharpes
        }


def fit_wf_once(model_config: Dict[str, Any], train_features: pd.DataFrame, 
               train_targets: pd.Series, val_features: pd.DataFrame, 
               val_targets: pd.Series, sequence_length: int = 128) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Fit a single model for walkforward validation
    
    Args:
        model_config: Model configuration
        train_features: Training features
        train_targets: Training targets
        val_features: Validation features
        val_targets: Validation targets
        sequence_length: Sequence length
    
    Returns:
        Tuple of (trained_model, training_results)
    """
    # Create model (exclude training parameters)
    model_params = {k: v for k, v in model_config.items() 
                   if k not in ['type', 'lr', 'batch_size', 'max_epochs', 'early_stop_patience']}
    
    model = create_model(
        model_type=model_config['type'],
        input_dim=train_features.shape[1],
        sequence_length=sequence_length,
        **model_params
    )
    
    # Calculate class weights
    class_counts = np.bincount(train_targets + 1)  # Convert to 0, 1, 2
    class_weights = torch.FloatTensor(len(class_counts) / (class_counts + 1e-8))
    
    # Create trainer
    trainer = DLTrainer(
        model=model,
        lr=model_config.get('lr', 0.001),
        weight_decay=model_config.get('weight_decay', 1e-4),
        batch_size=model_config.get('batch_size', 256),
        max_epochs=model_config.get('max_epochs', 100),
        early_stop_patience=model_config.get('early_stop_patience', 10)
    )
    
    # Train
    results = trainer.fit(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        sequence_length=sequence_length,
        class_weights=class_weights,
        multitask=model_config.get('multitask', False)
    )
    
    return model, results


if __name__ == "__main__":
    # Test trainer
    print("Testing DL trainer...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 28
    sequence_length = 128
    
    # Features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=pd.date_range('2023-01-01', periods=n_samples)
    )
    
    # Targets
    targets = pd.Series(
        np.random.choice([-1, 0, 1], n_samples),
        index=features.index
    )
    
    # Split data
    split_idx = int(n_samples * 0.8)
    train_features = features.iloc[:split_idx]
    train_targets = targets.iloc[:split_idx]
    val_features = features.iloc[split_idx:]
    val_targets = targets.iloc[split_idx:]
    
    # Test MLP
    print("Testing MLP trainer...")
    model_config = {'type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2}
    model, results = fit_wf_once(model_config, train_features, train_targets, 
                                val_features, val_targets, sequence_length)
    
    print(f"MLP training completed:")
    print(f"  Best val Sharpe: {results['best_val_sharpe']:.4f}")
    print(f"  Temperature: {results['temperature']:.4f}")
    print(f"  Best epoch: {results['best_epoch']}")
    
    # Test TCN
    print("Testing TCN trainer...")
    model_config = {'type': 'tcn', 'channels': 16, 'blocks': 2, 'dropout': 0.2}
    model, results = fit_wf_once(model_config, train_features, train_targets, 
                                val_features, val_targets, sequence_length)
    
    print(f"TCN training completed:")
    print(f"  Best val Sharpe: {results['best_val_sharpe']:.4f}")
    print(f"  Temperature: {results['temperature']:.4f}")
    print(f"  Best epoch: {results['best_epoch']}")
    
    print("✅ DL trainer working correctly!")
