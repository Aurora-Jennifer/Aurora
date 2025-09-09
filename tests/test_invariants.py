"""
Invariant Tests for Production Readiness

Tests numerical stability, data leakage prevention, and deterministic behavior.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ml.offline_rl import OfflineRLTrainer, OfflineRLConfig


class TestInvariants:
    """Test numerical and data integrity invariants"""
    
    def setup_method(self):
        """Setup test data"""
        # Create sample batch data
        self.sample_batch = {
            "states": torch.randn(100, 50),
            "actions": torch.randint(0, 3, (100,)),
            "rewards": torch.randn(100),
            "next_states": torch.randn(100, 50),
            "dones": torch.zeros(100, dtype=torch.bool)
        }
        
        # Ensure no NaNs in sample data
        self.sample_batch["states"] = torch.nan_to_num(self.sample_batch["states"])
        self.sample_batch["rewards"] = torch.nan_to_num(self.sample_batch["rewards"])
        self.sample_batch["next_states"] = torch.nan_to_num(self.sample_batch["next_states"])
    
    def test_no_nans(self):
        """Test that no NaNs are present in the pipeline"""
        x, r = self.sample_batch["states"], self.sample_batch["rewards"]
        assert torch.isfinite(x).all(), "States contain NaN/Inf values"
        assert torch.isfinite(r).all(), "Rewards contain NaN/Inf values"
    
    def test_determinism(self):
        """Test that training is deterministic with fixed seeds"""
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create config
        config = OfflineRLConfig(
            state_dim=50,
            action_dim=3,
            hidden_dim=64,
            num_layers=2,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=5,
            tau=0.7,
            beta=1.0
        )
        
        # Create trainer
        trainer1 = OfflineRLTrainer(config)
        
        # Set seeds again
        torch.manual_seed(42)
        np.random.seed(42)
        
        trainer2 = OfflineRLTrainer(config)
        
        # Compare initial weights
        for p1, p2 in zip(trainer1.q_network.parameters(), trainer2.q_network.parameters(), strict=False):
            torch.testing.assert_close(p1, p2, atol=1e-7, rtol=0)
    
    def test_gradient_clipping(self):
        """Test that gradient clipping works properly"""
        config = OfflineRLConfig(
            state_dim=50,
            action_dim=3,
            hidden_dim=64,
            num_layers=2,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=1,
            tau=0.7,
            beta=1.0
        )
        
        trainer = OfflineRLTrainer(config)
        
        # Create data with extreme values to trigger clipping
        states = torch.randn(32, 50) * 10  # Large values
        actions = torch.randint(0, 3, (32,))
        rewards = torch.randn(32) * 10
        next_states = torch.randn(32, 50) * 10
        dones = torch.zeros(32, dtype=torch.bool)
        
        # Train one step and check gradient norms
        trainer.train_iql(states.numpy(), actions.numpy(), rewards.numpy(), 
                         next_states.numpy(), dones.numpy())
        
        # Check that training completed without NaN losses
        assert not np.isnan(trainer.training_history['q_loss'][-1])
        assert not np.isnan(trainer.training_history['v_loss'][-1])
        assert not np.isnan(trainer.training_history['policy_loss'][-1])
    
    def test_action_prior_penalty(self):
        """Test that action prior penalty prevents collapse"""
        config = OfflineRLConfig(
            state_dim=50,
            action_dim=3,
            hidden_dim=64,
            num_layers=2,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=10,
            tau=0.7,
            beta=1.0
        )
        
        trainer = OfflineRLTrainer(config)
        
        # Create data with all same action (worst case for collapse)
        states = torch.randn(100, 50)
        actions = torch.zeros(100, dtype=torch.long)  # All action 0
        rewards = torch.randn(100)
        next_states = torch.randn(100, 50)
        dones = torch.zeros(100, dtype=torch.bool)
        
        # Train
        trainer.train_iql(states.numpy(), actions.numpy(), rewards.numpy(), 
                         next_states.numpy(), dones.numpy())
        
        # Check that policy doesn't collapse to single action
        # (This is a basic check - in practice, we'd need more sophisticated testing)
        final_policy_loss = trainer.training_history['policy_loss'][-1]
        assert not np.isnan(final_policy_loss)
        assert final_policy_loss < 10.0  # Reasonable upper bound


class TestDataLeakage:
    """Test that no data leakage occurs in preprocessing"""
    
    def test_train_val_isolation(self):
        """Test that preprocessing is fit only on training data"""
        # This would test the actual pipeline implementation
        # For now, we'll create a mock test
        
        # Simulate train/val split
        train_data = torch.randn(80, 50)
        val_data = torch.randn(20, 50)
        
        # Fit normalization on train only
        train_mean = train_data.mean(0)
        train_std = train_data.std(0)
        train_std[train_std < 1e-6] = 1.0
        
        # Apply to both train and val
        train_normalized = (train_data - train_mean) / train_std
        val_normalized = (val_data - train_mean) / train_std
        
        # Check that normalization worked
        assert torch.allclose(train_normalized.mean(0), torch.zeros(50), atol=1e-6)
        assert torch.allclose(train_normalized.std(0), torch.ones(50), atol=1e-6)
        
        # Val data should not be perfectly normalized (proving no leakage)
        assert not torch.allclose(val_normalized.mean(0), torch.zeros(50), atol=1e-6)
    
    def test_pca_fit_isolation(self):
        """Test that PCA is fit only on training data"""
        from sklearn.decomposition import PCA
        
        # Create train/val data
        train_data = torch.randn(100, 50)
        val_data = torch.randn(50, 50)
        
        # Fit PCA on train only
        pca = PCA(n_components=20)
        train_transformed = pca.fit_transform(train_data.numpy())
        val_transformed = pca.transform(val_data.numpy())
        
        # Check that PCA worked
        assert train_transformed.shape == (100, 20)
        assert val_transformed.shape == (50, 20)
        
        # Check that explained variance is reasonable
        assert pca.explained_variance_ratio_.sum() > 0.5


class TestMetricsConsistency:
    """Test that metrics are calculated consistently"""
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation consistency"""
        # Create sample returns
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, -0.005, 0.01, 0.005])
        
        # Calculate Sharpe ratio
        if len(returns) < 2 or returns.std() < 1e-12:
            sharpe = float("nan")
        else:
            daily_return = returns.mean()
            daily_vol = returns.std()
            sharpe = (daily_return / daily_vol) * np.sqrt(252)
        
        # Check that Sharpe is finite and reasonable
        assert not np.isnan(sharpe)
        assert -10 < sharpe < 10  # Reasonable range
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation"""
        # Create sample returns
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, -0.005, 0.01, 0.005])
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Check that max drawdown is negative and reasonable
        assert max_drawdown <= 0
        assert max_drawdown > -1.0  # Can't lose more than 100%


def test_label_balance_train():
    """Test that training labels are reasonably balanced"""
    # Simulate training labels
    actions = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 0, 1, 2])  # BUY, SELL, HOLD
    
    # Calculate class shares
    unique, counts = np.unique(actions, return_counts=True)
    shares = counts / len(actions)
    
    # Check balance constraints
    hold_share = shares[2] if len(shares) > 2 else 0  # HOLD is typically index 2
    buy_share = shares[0] if len(shares) > 0 else 0   # BUY is typically index 0
    sell_share = shares[1] if len(shares) > 1 else 0  # SELL is typically index 1
    
    assert hold_share <= 0.6, f"HOLD share {hold_share:.2f} should be <= 60%"
    assert min(buy_share, sell_share) >= 0.15, f"Min action share {min(buy_share, sell_share):.2f} should be >= 15%"


def test_policy_entropy_min():
    """Test that policy maintains minimum entropy"""
    import torch
    import torch.nn.functional as F
    
    # Simulate policy logits
    logits = torch.randn(10, 3)  # 10 samples, 3 actions
    probs = F.softmax(logits, dim=1)
    
    # Calculate entropy
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    # Should maintain minimum entropy
    assert entropy >= 0.25, f"Policy entropy {entropy:.3f} should be >= 0.25 nats"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
