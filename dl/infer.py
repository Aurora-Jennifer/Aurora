#!/usr/bin/env python3
"""
Deep Learning Inference Module

Implements inference producing calibrated probabilities with proper alignment
to canonical order and hysteresis decision making.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .models import create_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml.decision import align_proba, decide_hysteresis, edge_from_P


class DLInference:
    """Deep Learning inference with calibration and decision making"""
    
    def __init__(self, model: torch.nn.Module, scaler: StandardScaler, 
                 temperature: float = 1.0, device: str = 'cpu'):
        self.model = model
        self.scaler = scaler
        self.temperature = temperature
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def prepare_sequence(self, features: pd.DataFrame, sequence_length: int = 128) -> torch.Tensor:
        """
        Prepare sequence for inference
        
        Args:
            features: Feature DataFrame
            sequence_length: Sequence length for TCN models
        
        Returns:
            Prepared tensor
        """
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        if self.model.__class__.__name__ == 'TinyMLP':
            # For MLP, use only the last frame
            if len(X_scaled) < sequence_length:
                # Pad with zeros if sequence is too short
                pad_length = sequence_length - len(X_scaled)
                X_padded = np.pad(X_scaled, ((pad_length, 0), (0, 0)), mode='constant')
            else:
                X_padded = X_scaled[-sequence_length:]
            
            # Use only the last frame for MLP
            X_tensor = torch.FloatTensor(X_padded[-1:]).to(self.device)  # (1, features)
        else:
            # For TCN, create sequence
            if len(X_scaled) < sequence_length:
                # Pad with zeros if sequence is too short
                pad_length = sequence_length - len(X_scaled)
                X_padded = np.pad(X_scaled, ((pad_length, 0), (0, 0)), mode='constant')
            else:
                X_padded = X_scaled[-sequence_length:]
            
            X_tensor = torch.FloatTensor(X_padded).unsqueeze(0).to(self.device)  # (1, seq_len, features)
        
        return X_tensor
    
    def predict_proba(self, features: pd.DataFrame, sequence_length: int = 128) -> np.ndarray:
        """
        Predict probabilities with temperature scaling
        
        Args:
            features: Feature DataFrame
            sequence_length: Sequence length
        
        Returns:
            Probabilities of shape (1, 3) for [SELL, HOLD, BUY]
        """
        with torch.no_grad():
            X_tensor = self.prepare_sequence(features, sequence_length)
            
            # Forward pass
            if hasattr(self.model, 'forward') and len(self.model.forward(X_tensor)) == 2:
                # Multi-task model
                logits, _ = self.model(X_tensor)
            else:
                # Single-task model
                logits = self.model(X_tensor)
            
            # Apply temperature scaling
            probs = F.softmax(logits / self.temperature, dim=-1)
            
            return probs.cpu().numpy()
    
    def predict_edges(self, features: pd.DataFrame, sequence_length: int = 128) -> np.ndarray:
        """
        Predict edges (P(BUY) - P(SELL))
        
        Args:
            features: Feature DataFrame
            sequence_length: Sequence length
        
        Returns:
            Edge values
        """
        probs = self.predict_proba(features, sequence_length)
        
        # Align to canonical order [SELL, HOLD, BUY]
        probs_aligned = align_proba(self.model, probs)
        
        # Calculate edges
        edges = edge_from_P(probs_aligned)
        
        return edges
    
    def make_decisions(self, features: pd.DataFrame, tau_in: float, tau_out: float,
                      sequence_length: int = 128) -> np.ndarray:
        """
        Make trading decisions with hysteresis
        
        Args:
            features: Feature DataFrame
            tau_in: Entry threshold
            tau_out: Exit threshold
            sequence_length: Sequence length
        
        Returns:
            Position array: -1 (SELL), 0 (HOLD), +1 (BUY)
        """
        edges = self.predict_edges(features, sequence_length)
        positions = decide_hysteresis(edges, tau_in, tau_out)
        
        return positions


class DLSequenceInference:
    """Inference for sequences of data (for walkforward validation)"""
    
    def __init__(self, model: torch.nn.Module, scaler: StandardScaler,
                 temperature: float = 1.0, device: str = 'cpu'):
        self.model = model
        self.scaler = scaler
        self.temperature = temperature
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    def predict_sequence_proba(self, features: pd.DataFrame, sequence_length: int = 128) -> np.ndarray:
        """
        Predict probabilities for a sequence of data
        
        Args:
            features: Feature DataFrame
            sequence_length: Sequence length
        
        Returns:
            Probabilities of shape (n_samples, 3) for [SELL, HOLD, BUY]
        """
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        if self.model.__class__.__name__ == 'TinyMLP':
            # For MLP, use only the last frame of each sequence
            probs = []
            for i in range(sequence_length, len(X_scaled) + 1):
                seq = X_scaled[i-sequence_length:i]
                X_tensor = torch.FloatTensor(seq[-1:]).to(self.device)  # (1, features)
                
                with torch.no_grad():
                    logits = self.model(X_tensor)
                    prob = F.softmax(logits / self.temperature, dim=-1)
                    probs.append(prob.cpu().numpy())
            
            return np.vstack(probs)
        else:
            # For TCN, create sequences
            sequences = []
            for i in range(sequence_length, len(X_scaled) + 1):
                seq = X_scaled[i-sequence_length:i]
                sequences.append(seq)
            
            if sequences:
                X_tensor = torch.FloatTensor(np.array(sequences)).to(self.device)  # (n_seq, seq_len, features)
                
                with torch.no_grad():
                    if hasattr(self.model, 'forward') and len(self.model.forward(X_tensor)) == 2:
                        # Multi-task model
                        logits, _ = self.model(X_tensor)
                    else:
                        # Single-task model
                        logits = self.model(X_tensor)
                    
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    return probs.cpu().numpy()
            else:
                return np.array([]).reshape(0, 3)
    
    def predict_sequence_edges(self, features: pd.DataFrame, sequence_length: int = 128) -> np.ndarray:
        """
        Predict edges for a sequence of data
        
        Args:
            features: Feature DataFrame
            sequence_length: Sequence length
        
        Returns:
            Edge values
        """
        probs = self.predict_sequence_proba(features, sequence_length)
        
        if len(probs) == 0:
            return np.array([])
        
        # Align to canonical order [SELL, HOLD, BUY]
        probs_aligned = align_proba(self.model, probs)
        
        # Calculate edges
        edges = edge_from_P(probs_aligned)
        
        return edges
    
    def make_sequence_decisions(self, features: pd.DataFrame, tau_in: float, tau_out: float,
                               sequence_length: int = 128) -> np.ndarray:
        """
        Make trading decisions for a sequence of data
        
        Args:
            features: Feature DataFrame
            tau_in: Entry threshold
            tau_out: Exit threshold
            sequence_length: Sequence length
        
        Returns:
            Position array: -1 (SELL), 0 (HOLD), +1 (BUY)
        """
        edges = self.predict_sequence_edges(features, sequence_length)
        positions = decide_hysteresis(edges, tau_in, tau_out)
        
        return positions


def create_inference(model: torch.nn.Module, scaler: StandardScaler,
                    temperature: float = 1.0, device: str = 'cpu') -> DLInference:
    """
    Create inference object
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        temperature: Temperature for calibration
        device: Device to run on
    
    Returns:
        DLInference object
    """
    return DLInference(model, scaler, temperature, device)


def create_sequence_inference(model: torch.nn.Module, scaler: StandardScaler,
                             temperature: float = 1.0, device: str = 'cpu') -> DLSequenceInference:
    """
    Create sequence inference object
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        temperature: Temperature for calibration
        device: Device to run on
    
    Returns:
        DLSequenceInference object
    """
    return DLSequenceInference(model, scaler, temperature, device)


if __name__ == "__main__":
    # Test inference
    print("Testing DL inference...")
    
    # Create dummy model and data
    from .models import TinyMLP, CompactTCN
    from sklearn.preprocessing import StandardScaler
    
    # Test MLP inference
    print("Testing MLP inference...")
    mlp = TinyMLP(input_dim=28)
    scaler = StandardScaler()
    
    # Create dummy features
    features = pd.DataFrame(
        np.random.randn(200, 28),
        index=pd.date_range('2023-01-01', periods=200)
    )
    
    # Fit scaler
    scaler.fit(features)
    
    # Create inference
    inference = create_inference(mlp, scaler, temperature=1.0)
    
    # Test predictions
    probs = inference.predict_proba(features)
    edges = inference.predict_edges(features)
    positions = inference.make_decisions(features, tau_in=0.1, tau_out=0.05)
    
    print(f"MLP probs shape: {probs.shape}")
    print(f"MLP edges shape: {edges.shape}")
    print(f"MLP positions: {positions}")
    
    # Test TCN inference
    print("Testing TCN inference...")
    tcn = CompactTCN(input_dim=28, sequence_length=128)
    
    # Create sequence inference
    seq_inference = create_sequence_inference(tcn, scaler, temperature=1.0)
    
    # Test sequence predictions
    seq_probs = seq_inference.predict_sequence_proba(features)
    seq_edges = seq_inference.predict_sequence_edges(features)
    seq_positions = seq_inference.make_sequence_decisions(features, tau_in=0.1, tau_out=0.05)
    
    print(f"TCN seq probs shape: {seq_probs.shape}")
    print(f"TCN seq edges shape: {seq_edges.shape}")
    print(f"TCN seq positions: {seq_positions}")
    
    print("✅ DL inference working correctly!")
