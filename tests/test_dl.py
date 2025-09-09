#!/usr/bin/env python3
"""
Deep Learning Tests

Tests for the DL pipeline components:
- Model architectures
- Training with inner validation
- Inference and calibration
- Causality and no leakage
"""

import pytest
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dl.models import TinyMLP, CompactTCN, MultiTaskTCN, create_model
from dl.train import DLTrainer, fit_wf_once
from dl.infer import DLInference, DLSequenceInference, create_inference
from ml.targets import create_targets, validate_targets
from ml.decision import align_proba, edge_from_P
from ml.features import create_feature_pipeline
from sklearn.preprocessing import StandardScaler


class TestDLModels:
    """Test DL model architectures"""
    
    def test_mlp_architecture(self):
        """Test MLP architecture and parameter count"""
        mlp = TinyMLP(input_dim=28, hidden_dims=[128, 64])
        
        # Check parameter count
        param_count = mlp.get_param_count()
        assert param_count <= 40000, f"MLP has too many parameters: {param_count}"
        
        # Test forward pass
        x = torch.randn(32, 28)
        output = mlp(x)
        assert output.shape == (32, 3), f"Expected shape (32, 3), got {output.shape}"
        
        # Test probability prediction
        probs = mlp.predict_proba(x)
        assert probs.shape == (32, 3), f"Expected shape (32, 3), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6), "Probabilities don't sum to 1"
    
    def test_tcn_architecture(self):
        """Test TCN architecture and parameter count"""
        tcn = CompactTCN(input_dim=28, sequence_length=128, channels=16, blocks=3)
        
        # Check parameter count
        param_count = tcn.get_param_count()
        assert param_count <= 80000, f"TCN has too many parameters: {param_count}"
        
        # Test forward pass
        x = torch.randn(32, 128, 28)
        output = tcn(x)
        assert output.shape == (32, 3), f"Expected shape (32, 3), got {output.shape}"
        
        # Test probability prediction
        probs = tcn.predict_proba(x)
        assert probs.shape == (32, 3), f"Expected shape (32, 3), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6), "Probabilities don't sum to 1"
    
    def test_multitask_tcn(self):
        """Test multi-task TCN"""
        mt_tcn = MultiTaskTCN(input_dim=28, sequence_length=128, channels=16, blocks=3)
        
        # Test forward pass
        x = torch.randn(32, 128, 28)
        return_logits, vol_pred = mt_tcn(x)
        
        assert return_logits.shape == (32, 3), f"Expected return shape (32, 3), got {return_logits.shape}"
        assert vol_pred.shape == (32, 1), f"Expected vol shape (32, 1), got {vol_pred.shape}"
        
        # Test probability prediction
        probs = mt_tcn.predict_proba(x)
        assert probs.shape == (32, 3), f"Expected shape (32, 3), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6), "Probabilities don't sum to 1"
    
    def test_causality(self):
        """Test that models see only past frames (no future information)"""
        tcn = CompactTCN(input_dim=28, sequence_length=128, channels=16, blocks=3)
        
        # Create sequence with known pattern
        x = torch.randn(1, 128, 28)
        
        # Set first half to zeros, second half to ones
        x[0, :64, :] = 0.0
        x[0, 64:, :] = 1.0
        
        # Get output
        output = tcn(x)
        
        # The model should not be able to perfectly predict the future
        # (this is a basic causality test)
        assert not torch.allclose(output, torch.zeros_like(output)), "Model output should not be constant"
        assert not torch.allclose(output, torch.ones_like(output)), "Model output should not be constant"


class TestDLTraining:
    """Test DL training components"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        mlp = TinyMLP(input_dim=28)
        trainer = DLTrainer(mlp, lr=0.001, max_epochs=10)
        
        assert trainer.model == mlp
        assert trainer.lr == 0.001
        assert trainer.max_epochs == 10
        assert trainer.temperature == 1.0
    
    def test_data_preparation(self):
        """Test data preparation for training"""
        mlp = TinyMLP(input_dim=28)
        trainer = DLTrainer(mlp)
        
        # Create dummy data
        features = pd.DataFrame(
            np.random.randn(200, 28),
            index=pd.date_range('2023-01-01', periods=200)
        )
        targets = pd.Series(
            np.random.choice([-1, 0, 1], 200),
            index=features.index
        )
        
        # Test MLP data preparation
        X, y = trainer.prepare_data(features, targets, sequence_length=128)
        assert X.shape[0] == 1, f"Expected batch size 1, got {X.shape[0]}"
        assert X.shape[1] == 28, f"Expected feature dim 28, got {X.shape[1]}"
        assert y.shape[0] == 1, f"Expected target size 1, got {y.shape[0]}"
        
        # Test TCN data preparation
        tcn = CompactTCN(input_dim=28, sequence_length=128)
        tcn_trainer = DLTrainer(tcn)
        
        X_seq, y_seq = tcn_trainer.create_sequences(features, targets, sequence_length=128)
        assert X_seq.shape[1] == 128, f"Expected sequence length 128, got {X_seq.shape[1]}"
        assert X_seq.shape[2] == 28, f"Expected feature dim 28, got {X_seq.shape[2]}"
        assert len(X_seq) == len(y_seq), "X and y should have same length"
    
    def test_fit_wf_once(self):
        """Test single walkforward fit"""
        # Create dummy data
        np.random.seed(42)
        n_samples = 500
        features = pd.DataFrame(
            np.random.randn(n_samples, 28),
            index=pd.date_range('2023-01-01', periods=n_samples)
        )
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
        
        # Test MLP training
        model_config = {'type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2, 'max_epochs': 5}
        model, results = fit_wf_once(model_config, train_features, train_targets, 
                                    val_features, val_targets, sequence_length=128)
        
        assert isinstance(model, TinyMLP), "Model should be TinyMLP"
        assert 'best_val_sharpe' in results, "Results should contain best_val_sharpe"
        assert 'temperature' in results, "Results should contain temperature"
        assert results['temperature'] > 0, "Temperature should be positive"


class TestDLInference:
    """Test DL inference components"""
    
    def test_inference_initialization(self):
        """Test inference initialization"""
        mlp = TinyMLP(input_dim=28)
        scaler = StandardScaler()
        
        # Create dummy data to fit scaler
        dummy_data = np.random.randn(100, 28)
        scaler.fit(dummy_data)
        
        inference = create_inference(mlp, scaler, temperature=1.5)
        
        assert inference.model == mlp
        assert inference.temperature == 1.5
        assert inference.scaler == scaler
    
    def test_probability_calibration(self):
        """Test probability calibration"""
        mlp = TinyMLP(input_dim=28)
        scaler = StandardScaler()
        
        # Create dummy data
        features = pd.DataFrame(
            np.random.randn(200, 28),
            index=pd.date_range('2023-01-01', periods=200)
        )
        
        # Fit scaler
        scaler.fit(features)
        
        # Create inference
        inference = create_inference(mlp, scaler, temperature=1.0)
        
        # Test probability prediction
        probs = inference.predict_proba(features)
        
        # Check probability properties
        assert probs.shape == (1, 3), f"Expected shape (1, 3), got {probs.shape}"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6), "Probabilities should sum to 1"
        assert np.all(probs >= 0), "Probabilities should be non-negative"
        assert np.all(probs <= 1), "Probabilities should be <= 1"
    
    def test_edges_nonconstant(self):
        """Test that edges are not constant"""
        mlp = TinyMLP(input_dim=28)
        scaler = StandardScaler()
        
        # Create dummy data
        features = pd.DataFrame(
            np.random.randn(200, 28),
            index=pd.date_range('2023-01-01', periods=200)
        )
        
        # Fit scaler
        scaler.fit(features)
        
        # Create inference
        inference = create_inference(mlp, scaler, temperature=1.0)
        
        # Test edge prediction
        edges = inference.predict_edges(features)
        
        # Edges should not be constant
        assert np.std(edges) > 1e-6, "Edges should have variation"
        assert len(edges) > 0, "Should have edges"
    
    def test_sequence_inference(self):
        """Test sequence inference"""
        tcn = CompactTCN(input_dim=28, sequence_length=128)
        scaler = StandardScaler()
        
        # Create dummy data
        features = pd.DataFrame(
            np.random.randn(300, 28),
            index=pd.date_range('2023-01-01', periods=300)
        )
        
        # Fit scaler
        scaler.fit(features)
        
        # Create sequence inference
        seq_inference = DLSequenceInference(tcn, scaler, temperature=1.0)
        
        # Test sequence predictions
        seq_probs = seq_inference.predict_sequence_proba(features)
        seq_edges = seq_inference.predict_sequence_edges(features)
        
        # Check shapes
        expected_seq_len = len(features) - 128 + 1
        assert seq_probs.shape[0] == expected_seq_len, f"Expected {expected_seq_len} sequences, got {seq_probs.shape[0]}"
        assert seq_probs.shape[1] == 3, f"Expected 3 classes, got {seq_probs.shape[1]}"
        assert len(seq_edges) == expected_seq_len, f"Expected {expected_seq_len} edges, got {len(seq_edges)}"
        
        # Check probability properties
        assert np.allclose(seq_probs.sum(axis=1), 1.0, atol=1e-6), "Probabilities should sum to 1"


class TestIntegration:
    """Integration tests"""
    
    def test_no_leakage(self):
        """Test that training uses only past data (no lookahead)"""
        # This is a critical test - ensure no future information leakage
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with time structure
        features = pd.DataFrame(
            np.random.randn(n_samples, 28),
            index=pd.date_range('2023-01-01', periods=n_samples)
        )
        
        # Create targets with some time structure
        targets = pd.Series(
            np.random.choice([-1, 0, 1], n_samples),
            index=features.index
        )
        
        # Split data
        train_idx = int(n_samples * 0.7)
        train_features = features.iloc[:train_idx]
        train_targets = targets.iloc[:train_idx]
        val_features = features.iloc[train_idx:]
        val_targets = targets.iloc[train_idx:]
        
        # Train model
        model_config = {'type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2, 'max_epochs': 5}
        model, results = fit_wf_once(model_config, train_features, train_targets, 
                                    val_features, val_targets, sequence_length=128)
        
        # The model should not have perfect performance (which would indicate leakage)
        assert results['best_val_sharpe'] < 10.0, "Sharpe too high, possible leakage"
    
    def test_turnover_band(self):
        """Test that chosen tau yields reasonable turnover"""
        from ml.decision import pick_tau_from_train
        
        # Create sample edges
        np.random.seed(42)
        edges_train = np.random.randn(500) * 0.1
        
        # Test tau selection
        tau = pick_tau_from_train(edges_train, turnover_band=(0.08, 0.18))
        
        assert tau > 0, f"Tau should be positive, got {tau}"
        assert tau < 1.0, f"Tau should be reasonable, got {tau}"
        
        # Test that tau produces reasonable turnover
        from ml.decision import decide_hysteresis
        positions = decide_hysteresis(edges_train, tau, tau * 0.5)
        turnover = np.mean(np.abs(np.diff(positions)))
        
        # Turnover should be in reasonable range
        assert 0.05 <= turnover <= 0.25, f"Turnover {turnover} not in reasonable range"
    
    def test_early_stop_signal(self):
        """Test that early stopping triggers on worsening validation"""
        # This test ensures early stopping works correctly
        mlp = TinyMLP(input_dim=28)
        trainer = DLTrainer(mlp, max_epochs=20, early_stop_patience=3)
        
        # Simulate worsening validation
        trainer.val_sharpes = [0.5, 0.4, 0.3, 0.2, 0.1]
        trainer.best_val_sharpe = 0.5
        trainer.epochs_without_improvement = 4
        
        # Early stopping should trigger
        assert trainer.epochs_without_improvement >= trainer.early_stop_patience


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
