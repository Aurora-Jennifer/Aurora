#!/usr/bin/env python3
"""
Deep Learning Models Module

Implements compact DL models that beat clean baselines:
- Tiny MLP on last frame (≤40k params)
- Compact TCN with causal 1D CNN over time (≤80k params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class TinyMLP(nn.Module):
    """
    Tiny MLP on last frame - fast baseline
    
    Input: last x_{t} ∈ ℝ^F
    Architecture: [F -> 128 -> 64 -> 3], GELU, dropout=0.2
    Params: ≤ ~40k
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], 
                 dropout: float = 0.2, weight_decay: float = 1e-4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (3 classes: SELL, HOLD, BUY)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, 3)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Predict probabilities with temperature scaling
        
        Args:
            x: Input tensor
            temperature: Temperature for scaling
        
        Returns:
            Probabilities of shape (batch_size, 3)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits / temperature, dim=-1)
    
    def get_param_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution (no future information)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             dilation=dilation, padding=self.padding)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
        
        Returns:
            Output tensor with causal padding removed
        """
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Remove future padding
        return self.dropout(x)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
        
        Returns:
            Output tensor
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = F.gelu(out)
        
        # Second convolution
        out = self.conv2(out)
        out = F.gelu(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        out = out + residual
        
        # Layer normalization (transpose for LayerNorm)
        out = out.transpose(1, 2)  # (batch, seq, channels)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # (batch, channels, seq)
        
        return self.dropout(out)


class CompactTCN(nn.Module):
    """
    Compact Temporal Convolutional Network
    
    Input: sequence X[t-L+1:t] ∈ ℝ^{L×F}
    Architecture: 3-4 causal blocks, channels=16, kernel_size=3, dilations=[1,2,4,8]
    Global average pooling over time → 3-class head
    Params: ≤ ~80k
    """
    
    def __init__(self, input_dim: int, sequence_length: int, channels: int = 16,
                 blocks: int = 3, kernel_size: int = 3, dilations: list = [1, 2, 4, 8],
                 dropout: float = 0.2, weight_decay: float = 1e-4):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, channels, 1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        in_channels = channels
        
        for i in range(blocks):
            dilation = dilations[i % len(dilations)]
            self.tcn_blocks.append(
                TCNBlock(in_channels, channels, kernel_size, dilation, dropout)
            )
            in_channels = channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Linear(channels, 3)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Logits of shape (batch_size, 3)
        """
        # Transpose to (batch_size, input_dim, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Predict probabilities with temperature scaling
        
        Args:
            x: Input tensor
            temperature: Temperature for scaling
        
        Returns:
            Probabilities of shape (batch_size, 3)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits / temperature, dim=-1)
    
    def get_param_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


class MultiTaskTCN(CompactTCN):
    """
    TCN with multi-task learning (returns + volatility)
    """
    
    def __init__(self, input_dim: int, sequence_length: int, channels: int = 16,
                 blocks: int = 3, kernel_size: int = 3, dilations: list = [1, 2, 4, 8],
                 dropout: float = 0.2, weight_decay: float = 1e-4):
        super().__init__(input_dim, sequence_length, channels, blocks, 
                        kernel_size, dilations, dropout, weight_decay)
        
        # Remove the original classifier
        del self.classifier
        
        # Add multi-task heads
        self.return_head = nn.Linear(channels, 3)  # SELL, HOLD, BUY
        self.vol_head = nn.Linear(channels, 1)     # Volatility prediction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-task outputs
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Tuple of (return_logits, vol_prediction)
        """
        # Transpose to (batch_size, input_dim, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Multi-task heads
        return_logits = self.return_head(x)
        vol_pred = self.vol_head(x)
        
        return return_logits, vol_pred
    
    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Predict probabilities with temperature scaling (returns only)
        
        Args:
            x: Input tensor
            temperature: Temperature for scaling
        
        Returns:
            Probabilities of shape (batch_size, 3)
        """
        with torch.no_grad():
            return_logits, _ = self.forward(x)
            return F.softmax(return_logits / temperature, dim=-1)


def create_model(model_type: str, input_dim: int, sequence_length: int = 128, **kwargs) -> nn.Module:
    """
    Create a model based on type
    
    Args:
        model_type: 'mlp' or 'tcn'
        input_dim: Input feature dimension
        sequence_length: Sequence length for TCN
        **kwargs: Additional model parameters
    
    Returns:
        PyTorch model
    """
    if model_type == 'mlp':
        return TinyMLP(input_dim, **kwargs)
    elif model_type == 'tcn':
        return CompactTCN(input_dim, sequence_length, **kwargs)
    elif model_type == 'multitask_tcn':
        return MultiTaskTCN(input_dim, sequence_length, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing DL models...")
    
    # Test MLP
    mlp = TinyMLP(input_dim=28)
    print(f"MLP params: {mlp.get_param_count():,}")
    
    x_mlp = torch.randn(32, 28)
    out_mlp = mlp(x_mlp)
    prob_mlp = mlp.predict_proba(x_mlp)
    print(f"MLP output shape: {out_mlp.shape}, prob shape: {prob_mlp.shape}")
    
    # Test TCN
    tcn = CompactTCN(input_dim=28, sequence_length=128)
    print(f"TCN params: {tcn.get_param_count():,}")
    
    x_tcn = torch.randn(32, 128, 28)
    out_tcn = tcn(x_tcn)
    prob_tcn = tcn.predict_proba(x_tcn)
    print(f"TCN output shape: {out_tcn.shape}, prob shape: {prob_tcn.shape}")
    
    # Test Multi-task TCN
    mt_tcn = MultiTaskTCN(input_dim=28, sequence_length=128)
    print(f"Multi-task TCN params: {mt_tcn.get_param_count():,}")
    
    out_mt, vol_mt = mt_tcn(x_tcn)
    prob_mt = mt_tcn.predict_proba(x_tcn)
    print(f"Multi-task TCN output shapes: {out_mt.shape}, {vol_mt.shape}")
    print(f"Multi-task TCN prob shape: {prob_mt.shape}")
    
    print("✅ All models working correctly!")
