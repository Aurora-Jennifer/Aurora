"""
Offline Reinforcement Learning for Trading

Implements IQL (Implicit Q-Learning) and CQL (Conservative Q-Learning)
for stable offline RL training on historical trading data.

Based on the framework for offline-heavy data regime with risk constraints.
"""

import numpy as np
import pandas as pd
from typing import Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
from .balanced_sampler import create_balanced_dataloader, analyze_action_distribution
from .diagnostics import diagnose_pytorch_step, get_system_info

def assert_ok(name, x):
    """Assert that tensor/array contains only finite values"""
    if isinstance(x, (list, tuple)):
        for i, t in enumerate(x):
            if torch.is_tensor(t):
                if not torch.isfinite(t).all():
                    raise ValueError(f"{name}[{i}] has non-finite values: "
                                     f"nan={torch.isnan(t).any().item()}, "
                                     f"inf={(~torch.isfinite(t)).any().item()}")
    elif torch.is_tensor(x):
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} invalid: nan/inf present")
    elif hasattr(x, 'dtype') and 'float' in str(x.dtype):  # numpy arrays
        if not np.isfinite(x).all():
            raise ValueError(f"{name} invalid: nan/inf present")
    return x

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Offline RL features disabled.")


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training"""
    # Model architecture (right-sized for small datasets)
    state_dim: int = 64  # Reduced from 114
    action_dim: int = 3  # BUY, SELL, HOLD
    hidden_dim: int = 32  # Further reduced for small datasets (398-638 samples)
    num_layers: int = 2  # Reduced from 3
    
    # Training parameters (optimized for production)
    learning_rate: float = 3e-4  # Higher for faster convergence
    warm_start_lr: float = 3e-4  # Learning rate for warm-start phase
    batch_size: int = 1024  # Increased for better GPU utilization
    num_epochs: int = 20  # Reduced from 100 for faster training
    tau: float = 0.7  # For IQL
    beta: float = 1.0  # For IQL
    alpha: float = 1.0  # For CQL
    
    # Risk constraints
    max_position_size: float = 0.3
    max_daily_trades: int = 5
    min_confidence_threshold: float = 0.6
    
    # Offline RL specific
    use_iql: bool = True
    use_cql: bool = False
    conservative_weight: float = 1.0  # For CQL
    
    # Action diversity parameters
    use_class_weights: bool = True
    max_class_weight: float = 2.0
    entropy_start_beta: float = 0.05
    entropy_end_beta: float = 0.01
    logit_l2_reg: float = 1e-4
    dead_zone_threshold: float = 0.25
    min_entropy_threshold: float = 0.25


class QNetwork(nn.Module):
    """Q-network for offline RL with proper initialization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)


class VNetwork(nn.Module):
    """Value network for IQL with proper initialization"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Multi-head policy network for per-symbol action prediction"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 3, num_symbols: int = 3):
        super().__init__()
        
        # Shared feature encoder
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_encoder = nn.Sequential(*layers)
        
        # Separate heads for each symbol (3 actions each: BUY, SELL, HOLD)
        self.symbol_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_symbols)  # Remove Softmax, add learnable scale
        ])
        
        # Learnable logit scale (temperature^-1) to boost logits
        self.logit_scale = nn.Parameter(torch.tensor(1.5))  # >1 boosts logits
        
        self.num_symbols = num_symbols
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        # Encode shared features
        features = self.feature_encoder(state)
        
        # Get action logits for each symbol and apply learnable scale
        symbol_logits = []
        for head in self.symbol_heads:
            logits = head(features) * self.logit_scale.clamp(0.5, 10.0)
            symbol_logits.append(logits)
        
        # Concatenate all symbol logits
        return torch.cat(symbol_logits, dim=1)
    
    def get_symbol_actions(self, state):
        """Get action probabilities for each symbol separately"""
        features = self.feature_encoder(state)
        symbol_probs = []
        for head in self.symbol_heads:
            logits = head(features) * self.logit_scale.clamp(0.5, 10.0)
            probs = torch.softmax(logits, dim=1)
            symbol_probs.append(probs)
        return symbol_probs
    
    def get_symbol_logits(self, state):
        """Get raw logits for each symbol separately (before softmax)"""
        features = self.feature_encoder(state)
        symbol_logits = []
        for head in self.symbol_heads:
            logits = head(features) * self.logit_scale.clamp(0.5, 10.0)
            symbol_logits.append(logits)
        return symbol_logits


class OfflineRLTrainer:
    """
    Offline RL trainer implementing IQL and CQL
    
    IQL (Implicit Q-Learning): Stable offline RL that avoids extrapolation errors
    CQL (Conservative Q-Learning): Adds pessimism against out-of-distribution actions
    """
    
    def __init__(self, config: OfflineRLConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for offline RL training")
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.q_network = QNetwork(config.state_dim, config.action_dim, 
                                 config.hidden_dim, config.num_layers).to(self.device)
        self.v_network = VNetwork(config.state_dim, config.hidden_dim, 
                                 config.num_layers).to(self.device)
        self.policy_network = PolicyNetwork(config.state_dim, config.action_dim, 
                                           config.hidden_dim, config.num_layers).to(self.device)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        # Training history
        self.training_history = {
            'q_loss': [],
            'v_loss': [],
            'policy_loss': [],
            'rewards': []
        }
    
    def train_iql(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                  next_states: np.ndarray, dones: np.ndarray) -> dict[str, Any]:
        """
        Train using Implicit Q-Learning (IQL) with numerical safety
        
        IQL is stable for offline RL because it:
        1. Learns Q-values implicitly through value function
        2. Avoids extrapolation errors from out-of-distribution actions
        3. Uses expectile regression for robust value estimation
        """
        
        print("Training with Implicit Q-Learning (IQL)...")
        
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Numerical safety checks on inputs
        states = assert_ok("states", states)
        actions = assert_ok("actions", actions)
        rewards = assert_ok("rewards", rewards)
        next_states = assert_ok("next_states", next_states)
        
        # Proper train/val split to prevent data leakage
        print("Creating proper train/val split...")
        train_size = int(0.8 * len(states))
        
        # Split data
        states_train = states[:train_size]
        states_val = states[train_size:]
        next_states_train = next_states[:train_size]
        next_states_val = next_states[train_size:]
        actions_train = actions[:train_size]
        actions_val = actions[train_size:]
        rewards_train = rewards[:train_size]
        rewards_val = rewards[train_size:]
        dones_train = dones[:train_size]
        dones_val = dones[train_size:]
        
        print(f"Train: {len(states_train)} samples, Val: {len(states_val)} samples")
        
        # Feature normalization with winsorization - FIT ONLY ON TRAIN
        print("Applying feature normalization with winsorization (train-only fit)...")
        
        # Winsorize features at 1st/99th percentiles - FIT ON TRAIN ONLY
        states_train_winsorized = np.copy(states_train)
        next_states_train_winsorized = np.copy(next_states_train)
        
        # Fit winsorization bounds on train data
        winsorize_bounds = []
        for i in range(states_train.shape[1]):
            q01, q99 = np.percentile(states_train[:, i], [1, 99])
            winsorize_bounds.append((q01, q99))
        
        # Apply winsorization to train data
        for i in range(states_train.shape[1]):
            q01, q99 = winsorize_bounds[i]
            states_train_winsorized[:, i] = np.clip(states_train[:, i], q01, q99)
            next_states_train_winsorized[:, i] = np.clip(next_states_train[:, i], q01, q99)
        
        # Apply same bounds to val data (no fitting on val)
        states_val_winsorized = np.copy(states_val)
        next_states_val_winsorized = np.copy(next_states_val)
        for i in range(states_val.shape[1]):
            q01, q99 = winsorize_bounds[i]
            states_val_winsorized[:, i] = np.clip(states_val[:, i], q01, q99)
            next_states_val_winsorized[:, i] = np.clip(next_states_val[:, i], q01, q99)
        
        # Standardize features - FIT ON TRAIN ONLY
        mu = states_train_winsorized.mean(0)
        sd = states_train_winsorized.std(0)
        sd[sd < 1e-6] = 1.0  # Prevent division by zero
        
        # Apply to train data
        states_train_normalized = (states_train_winsorized - mu) / sd
        next_states_train_normalized = (next_states_train_winsorized - mu) / sd
        
        # Apply same normalization to val data (no fitting on val)
        states_val_normalized = (states_val_winsorized - mu) / sd
        next_states_val_normalized = (next_states_val_winsorized - mu) / sd
        
        # Drop near-constant features (variance < 1e-8) - FIT ON TRAIN ONLY
        feature_vars = np.var(states_train_normalized, axis=0)
        keep_features = feature_vars > 1e-8
        
        if np.sum(keep_features) < states_train.shape[1]:
            print(f"Dropping {states_train.shape[1] - np.sum(keep_features)} near-constant features")
            states_train_normalized = states_train_normalized[:, keep_features]
            next_states_train_normalized = next_states_train_normalized[:, keep_features]
            states_val_normalized = states_val_normalized[:, keep_features]
            next_states_val_normalized = next_states_val_normalized[:, keep_features]
        
        # PCA dimensionality reduction - FIT ON TRAIN ONLY (adaptive components)
        if states_train_normalized.shape[1] > 64:  # Reduced threshold for production
            from sklearn.decomposition import PCA
            n_samples, n_features = states_train_normalized.shape
            max_components = min(n_samples - 1, n_features)
            n_components = min(64, max_components)  # Cap at 64 for production
            pca = PCA(n_components=n_components)
            states_train_normalized = pca.fit_transform(states_train_normalized)
            next_states_train_normalized = pca.transform(next_states_train_normalized)
            states_val_normalized = pca.transform(states_val_normalized)
            next_states_val_normalized = pca.transform(next_states_val_normalized)
            print(f"Applied PCA: {states_train.shape[1]} -> {states_train_normalized.shape[1]} features")
        
        # Combine train and val data
        states_normalized = np.vstack([states_train_normalized, states_val_normalized])
        next_states_normalized = np.vstack([next_states_train_normalized, next_states_val_normalized])
        actions = np.concatenate([actions_train, actions_val])
        rewards = np.concatenate([rewards_train, rewards_val])
        dones = np.concatenate([dones_train, dones_val])
        
        # Reward normalization and clipping
        print("Normalizing and clipping rewards...")
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std < 1e-6:
            reward_std = 1.0
        
        rewards_normalized = (rewards - reward_mean) / (reward_std + 1e-6)
        rewards_clipped = np.clip(rewards_normalized, -5, 5)  # Clip at 5 sigma
        
        # Convert to tensors
        states = torch.FloatTensor(states_normalized).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards_clipped).to(self.device)
        next_states = torch.FloatTensor(next_states_normalized).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update config with new state dimension
        self.config.state_dim = states.shape[1]
        print(f"Final state dimension: {self.config.state_dim}")
        
        # Reinitialize networks with correct dimensions
        self.q_network = QNetwork(self.config.state_dim, self.config.action_dim, 
                                 self.config.hidden_dim, self.config.num_layers).to(self.device)
        self.v_network = VNetwork(self.config.state_dim, self.config.hidden_dim, 
                                 self.config.num_layers).to(self.device)
        self.policy_network = PolicyNetwork(self.config.state_dim, self.config.action_dim, 
                                           self.config.hidden_dim, self.config.num_layers, 
                                           num_symbols=3).to(self.device)
        
        # Reinitialize optimizers
        # Use warm-start learning rate for policy, lower LR for Q/V networks
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate * 0.3, weight_decay=1e-4)
        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=self.config.learning_rate * 0.3, weight_decay=1e-4)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.warm_start_lr, weight_decay=1e-4)
        
        # Verify optimizer parameter counts
        num_pol = sum(p.numel() for n, p in self.policy_network.named_parameters() if p.requires_grad)
        num_q = sum(p.numel() for n, p in self.q_network.named_parameters() if p.requires_grad)
        num_v = sum(p.numel() for n, p in self.v_network.named_parameters() if p.requires_grad)
        
        print(f"Parameter counts: policy={num_pol}, q_network={num_q}, v_network={num_v}")
        
        # Verify policy params are in optimizer
        policy_param_ids = {id(p) for p in self.policy_network.parameters()}
        optimizer_param_ids = {id(p) for group in self.policy_optimizer.param_groups for p in group['params']}
        
        if policy_param_ids != optimizer_param_ids:
            print("WARNING: Policy parameters not properly included in optimizer!")
        else:
            print("✅ Policy parameters properly included in optimizer")
        
        # Analyze action distribution
        print("Analyzing action distribution...")
        action_dist = analyze_action_distribution(actions)
        
        print("Action distribution:")
        for action_id, stats in action_dist.items():
            print(f"  Action {action_id}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        # Create dataset with balanced sampling (ensure tensors are on CPU for DataLoader)
        dataset = TensorDataset(
            states.cpu(), 
            actions.cpu(), 
            rewards.cpu(), 
            next_states.cpu(), 
            dones.cpu()
        )
        dataloader = create_balanced_dataloader(
            dataset, 
            actions, 
            self.config.batch_size,
            target_ratios=[0.33, 0.33, 0.34]  # Slightly favor HOLD
        )
        
        # Warm-start training with weighted CE
        print("Starting warm-start training...")
        warm_start_epochs = 25
        
        # Ensure training mode during warm-start
        self.policy_network.train()
        
        # Enable mixed precision for RTX 3080
        scaler = torch.cuda.amp.GradScaler()
        
        # Calculate inverse frequency weights
        # Convert actions to numpy if it's a torch tensor
        if hasattr(actions, 'cpu'):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
            
        action_counts = np.bincount(actions_np, minlength=3)
        inv_freq_weights = len(actions_np) / (3 * action_counts + 1e-8)
        inv_freq_weights = inv_freq_weights / inv_freq_weights.mean()  # Normalize to mean 1.0
        class_weights = torch.FloatTensor(inv_freq_weights).to(self.device)
        
        print(f"Class weights: {class_weights}")
        
        for warm_epoch in range(warm_start_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in dataloader:
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
                states_batch = states_batch.to(self.device, non_blocking=True)
                actions_batch = actions_batch.to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    # Forward pass - get logits for training
                    symbol_logits = self.policy_network.get_symbol_logits(states_batch)
                    symbol_probs = self.policy_network.get_symbol_actions(states_batch)
                    
                    # Use primary symbol for warm-start
                    primary_logits = symbol_logits[0]
                    primary_probs = symbol_probs[0]
                    log_probs = torch.log_softmax(primary_logits, dim=1)
                    
                    # Weighted cross-entropy loss (use logits directly)
                    ce_loss = F.cross_entropy(primary_logits, actions_batch, weight=class_weights)
                    
                    # Add stronger entropy bonus to prevent action collapse
                    entropy = -(primary_probs * log_probs).sum(-1).mean()
                    total_loss = ce_loss - 0.05 * entropy  # Increased from 0.01 to 0.05 for stronger exploration
                
                # Backward pass with mixed precision
                self.policy_optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                
                # Simple progress logging every 10 batches
                if batch_count % 10 == 0:
                    print(f"  Warm-start epoch {warm_epoch+1}/{warm_start_epochs}, batch {batch_count}: loss={total_loss.item():.4f}")
                
                scaler.unscale_(self.policy_optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
                scaler.step(self.policy_optimizer)
                scaler.update()
                
                
                epoch_loss += total_loss.item()
                batch_count += 1
            
            if warm_epoch % 5 == 0:
                print(f"Warm-start epoch {warm_epoch}: avg_loss={epoch_loss/len(dataloader):.4f}")
        
        print("Warm-start completed, starting IQL training...")
        
        # Enable fast kernels for RTX 3080
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Run GPU diagnostics
        print("\n=== GPU DIAGNOSTICS ===")
        try:
            system_info = get_system_info()
            print("System info:", system_info)
            
            # Create a simple model for diagnostics
            class SimpleModel(nn.Module):
                def __init__(self, input_dim, hidden_dim=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 3)  # 3 actions
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            # Use a subset of data for diagnostics
            diag_states = states[:1000].to(self.device)
            diag_actions = actions[:1000].to(self.device)
            diag_dataset = TensorDataset(diag_states, diag_actions)
            diag_loader = DataLoader(diag_dataset, batch_size=64, shuffle=True, num_workers=0)
            
            diag_model = SimpleModel(states.shape[1]).to(self.device)
            diag_result = diagnose_pytorch_step(diag_model, diag_loader, device=self.device, steps=20)
            print("GPU diagnostics completed")
        except Exception as e:
            print(f"GPU diagnostics failed: {e}")
        
        # Training monitoring variables
        total_steps = 0
        clip_hits = 0
        grad_l2_sum = 0.0
        
        # Enable mixed precision for IQL training too
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.config.num_epochs):
            epoch_q_loss = 0
            epoch_v_loss = 0
            epoch_policy_loss = 0
            epoch_grad_norms = []
            epoch_clip_hits = []
            
            for batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones in dataloader:
                # Move batch to device
                batch_states = batch_states.to(self.device, non_blocking=True)
                batch_actions = batch_actions.to(self.device, non_blocking=True)
                batch_rewards = batch_rewards.to(self.device, non_blocking=True)
                batch_next_states = batch_next_states.to(self.device, non_blocking=True)
                batch_dones = batch_dones.to(self.device, non_blocking=True)
                
                # 1. Train V-network (expectile regression)
                v_loss, v_grad_norm, v_clip_hit = self._train_v_network(batch_states, batch_actions, batch_rewards, 
                                                                       batch_next_states, batch_dones)
                epoch_v_loss += v_loss
                epoch_grad_norms.append(v_grad_norm)
                epoch_clip_hits.append(v_clip_hit)
                
                # 2. Train Q-network (implicit Q-learning)
                q_loss, q_grad_norm, q_clip_hit = self._train_q_network(batch_states, batch_actions, batch_rewards, 
                                                                       batch_next_states, batch_dones)
                epoch_q_loss += q_loss
                epoch_grad_norms.append(q_grad_norm)
                epoch_clip_hits.append(q_clip_hit)
                
                # 3. Train policy network (advantage-weighted regression)
                policy_loss, policy_grad_norm, policy_clip_hit = self._train_policy_network(batch_states, batch_actions)
                epoch_policy_loss += policy_loss
                epoch_grad_norms.append(policy_grad_norm)
                epoch_clip_hits.append(policy_clip_hit)
                
                # DEBUG: Log reward signal for Symbol 0 (first 5 batches only)
                if total_steps < 5:
                    symbol_0_rewards = batch_rewards[batch_actions == 0]  # BUY actions
                    symbol_0_states = batch_states[batch_actions == 0]
                    if len(symbol_0_rewards) > 0:
                        print(f"DEBUG Symbol 0 (BUY): rewards={symbol_0_rewards[:3].cpu().numpy()}, "
                              f"state_sample={symbol_0_states[0, :5].cpu().numpy()}")
                
                total_steps += 1
                clip_hits += sum(epoch_clip_hits[-3:])  # Last 3 networks
                grad_l2_sum += sum(epoch_grad_norms[-3:])
            
            # Record training history
            self.training_history['q_loss'].append(epoch_q_loss / len(dataloader))
            self.training_history['v_loss'].append(epoch_v_loss / len(dataloader))
            self.training_history['policy_loss'].append(epoch_policy_loss / len(dataloader))
            
            if epoch % 10 == 0:
                avg_grad_norm = np.mean([float(x) for x in epoch_grad_norms]) if epoch_grad_norms else 0.0
                clip_hit_rate = np.mean([float(x) for x in epoch_clip_hits]) if epoch_clip_hits else 0.0
                
                # Calculate entropy and logit norm for monitoring
                with torch.no_grad():
                    sample_states = states[:min(100, len(states))]
                    if isinstance(sample_states, torch.Tensor):
                        sample_states_tensor = sample_states.to(self.device)
                    else:
                        sample_states_tensor = torch.FloatTensor(sample_states).to(self.device)
                    
                    # Get per-symbol actions for multi-head policy
                    symbol_probs = self.policy_network.get_symbol_actions(sample_states_tensor)
                    
                    # Calculate entropy per symbol and average (from actual policy probs)
                    total_entropy = 0.0
                    for symbol_prob in symbol_probs:
                        # Use actual policy probabilities, not prior
                        probs = symbol_prob.clamp_min(1e-12)
                        entropy_per_symbol = -(probs * probs.log()).sum(-1).mean()
                        total_entropy += entropy_per_symbol.item()
                    
                    entropy = total_entropy / len(symbol_probs)
                    entropy = min(entropy, np.log(3))  # Clamp to max possible
                    
                    # Calculate logit norm
                    all_logits = torch.cat(symbol_probs, dim=1)
                    logit_norm = torch.norm(all_logits, p=2, dim=1).mean().item()
                    
                    # DEBUG: Detailed policy analysis for action collapse
                    if epoch % 5 == 0:  # Every 5 epochs
                        print(f"DEBUG Policy Analysis Epoch {epoch}:")
                        for i, probs in enumerate(symbol_probs):
                            # Calculate per-symbol entropy
                            probs_clamped = probs.clamp_min(1e-12)
                            symbol_entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1).mean()
                            
                            # Get action probability distribution
                            action_probs = probs.mean(dim=0)  # Average across batch
                            
                            # Get raw logits for this symbol (before softmax)
                            feature_output = self.policy_network.feature_encoder(sample_states_tensor)
                            # Get the linear layer (now directly accessible)
                            linear_layer = self.policy_network.symbol_heads[i]  # Direct Linear layer
                            logits = linear_layer(feature_output) * self.policy_network.logit_scale.clamp(0.5, 10.0)
                            logit_norm_per_symbol = torch.norm(logits, dim=-1).mean()
                            
                            print(f"  Symbol {i}: entropy={symbol_entropy:.4f}, logit_norm={logit_norm_per_symbol:.4f}, "
                                  f"action_probs=[BUY={action_probs[0]:.3f}, SELL={action_probs[1]:.3f}, HOLD={action_probs[2]:.3f}]")
                
                print(f"Epoch {epoch}: Q-loss={epoch_q_loss/len(dataloader):.4f}, "
                      f"V-loss={epoch_v_loss/len(dataloader):.4f}, "
                      f"Policy-loss={epoch_policy_loss/len(dataloader):.4f}, "
                      f"Grad-norm={avg_grad_norm:.4f}, "
                      f"Clip-rate={clip_hit_rate:.2%}, "
                      f"Entropy={entropy:.3f}, "
                      f"Logit-norm={logit_norm:.3f}")
        
        # Final gradient statistics
        if total_steps > 0:
            avg_grad_l2 = grad_l2_sum / total_steps
            overall_clip_rate = clip_hits / total_steps
            print(f"Training completed: avg_grad_l2={avg_grad_l2:.4f}, overall_clip_rate={overall_clip_rate:.2%}")
            
            # Check gradient clipping rate
            if overall_clip_rate > 0.20:
                print(f"⚠️  Warning: High gradient clipping rate ({overall_clip_rate:.2%} > 20%)")
        
        # Multi-head action collapse check on validation data
        if len(states) > 0:
            self.policy_network.eval()
            with torch.no_grad():
                # Get predictions on a sample of states
                sample_size = min(100, len(states))
                sample_indices = np.random.choice(len(states), sample_size, replace=False)
                sample_states = states[sample_indices]
                
                # Get per-symbol action predictions
                symbol_actions = [[] for _ in range(self.policy_network.num_symbols)]
                
                for state in sample_states:
                    if isinstance(state, torch.Tensor):
                        state_tensor = state.unsqueeze(0).to(self.device)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # Get per-symbol action probabilities
                    symbol_probs = self.policy_network.get_symbol_actions(state_tensor)
                    
                    for i, symbol_prob in enumerate(symbol_probs):
                        # Use stochastic sampling for validation (same as test phase)
                        action_dist = torch.distributions.Categorical(probs=symbol_prob)
                        action = action_dist.sample().item()
                        symbol_actions[i].append(action)
                
                # Check action distribution per symbol
                print("Per-symbol validation action distribution:")
                max_share_overall = 0.0
                
                for i, actions in enumerate(symbol_actions):
                    action_counts = np.bincount(actions, minlength=3)
                    max_share = action_counts.max() / action_counts.sum()
                    max_share_overall = max(max_share_overall, max_share)
                    
                    print(f"  Symbol {i}: {action_counts.tolist()}, max_share={max_share:.2f}")
                
                # Action collapse guard
                if max_share_overall > 0.80:
                    print(f"⚠️  Warning: Action collapse detected (max_share={max_share_overall:.2f} > 0.80)")
                else:
                    print(f"✅ Action diversity OK (max_share={max_share_overall:.2f} <= 0.80)")
            
            self.policy_network.train()
        
        return {
            'training_completed': True,
            'final_q_loss': self.training_history['q_loss'][-1],
            'final_v_loss': self.training_history['v_loss'][-1],
            'final_policy_loss': self.training_history['policy_loss'][-1]
        }
    
    def _train_v_network(self, states, actions, rewards, next_states, dones):
        """Train V-network using expectile regression with numerical safety"""
        
        with torch.no_grad():
            # Get Q-values for next states
            next_q_values = self.q_network(next_states)
            next_v_values = torch.max(next_q_values, dim=1)[0]
            
            # Compute target values with clipping
            target_values = rewards + (1 - dones.float()) * 0.99 * next_v_values
            target_values = torch.clamp(target_values, -10, 10)  # Clip targets
        
        # Get current V-values
        v_values = self.v_network(states).squeeze()
        
        # Expectile regression loss with Huber loss for stability
        tau = self.config.tau
        errors = target_values - v_values
        
        # Use Huber loss instead of squared error for robustness
        huber_loss = torch.where(
            torch.abs(errors) < 1.0,
            0.5 * errors ** 2,
            torch.abs(errors) - 0.5
        )
        
        expectile_loss = torch.where(
            errors > 0,
            tau * huber_loss,
            (1 - tau) * huber_loss
        ).mean()
        
        # Update V-network with gradient clipping
        self.v_optimizer.zero_grad()
        expectile_loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), 1.0)
        clip_hit = grad_norm > 1.0
        
        self.v_optimizer.step()
        
        return expectile_loss.item(), grad_norm.item(), clip_hit
    
    def _train_q_network(self, states, actions, rewards, next_states, dones):
        """Train Q-network using implicit Q-learning with numerical safety"""
        
        with torch.no_grad():
            # Get V-values for next states
            next_v_values = self.v_network(next_states).squeeze()
            
            # Compute target Q-values with clipping
            target_q_values = rewards + (1 - dones.float()) * 0.99 * next_v_values
            target_q_values = torch.clamp(target_q_values, -10, 10)  # Clip targets
        
        # Get current Q-values
        q_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Q-learning loss with Huber loss for stability
        q_loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Update Q-network with gradient clipping
        self.q_optimizer.zero_grad()
        q_loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        clip_hit = grad_norm > 1.0
        
        self.q_optimizer.step()
        
        return q_loss.item(), grad_norm.item(), clip_hit
    
    def _train_policy_network(self, states, actions):
        """Train multi-head policy network using per-symbol advantage-weighted regression"""
        
        with torch.no_grad():
            # Get Q-values and V-values
            q_values = self.q_network(states)
            v_values = self.v_network(states).squeeze()
            
            # Compute advantages with clipping
            advantages = q_values.gather(1, actions.unsqueeze(1)).squeeze() - v_values
            advantages = torch.clamp(advantages, -5.0, 5.0)  # Clip advantages
            
            # Compute weights (exponential of advantages) with safety
            weights = torch.exp(advantages / self.config.beta)
            weights = torch.clamp(weights, max=50.0)  # Cap weights to prevent explosion
        
        # Get per-symbol action probabilities
        symbol_probs = self.policy_network.get_symbol_actions(states)
        
        # Convert actions to per-symbol format (assuming actions are encoded as symbol_idx * 3 + action_idx)
        batch_size = actions.shape[0]
        symbol_actions = []
        for i in range(self.policy_network.num_symbols):
            symbol_action = (actions // 3) == i  # Which samples have actions for this symbol
            symbol_action_idx = actions % 3  # The actual action (0, 1, 2)
            symbol_actions.append(symbol_action_idx)
        
        # Compute per-symbol losses
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        
        for i, (symbol_prob, symbol_action) in enumerate(zip(symbol_probs, symbol_actions, strict=False)):
            # Get action probabilities for this symbol
            action_probs = symbol_prob.gather(1, symbol_action.unsqueeze(1)).squeeze()
            action_probs = torch.clamp(action_probs, 1e-8, 1.0)
            
            # Per-symbol advantage-weighted regression loss
            symbol_loss = -(weights * torch.log(action_probs)).mean()
            total_policy_loss += symbol_loss
            
            # Per-symbol entropy
            symbol_dist = torch.distributions.Categorical(probs=symbol_prob)
            total_entropy += symbol_dist.entropy().mean()
            
            # Per-symbol KL divergence to uniform prior
            uniform_prior = torch.ones(3, device=symbol_prob.device) / 3
            prior_dist = torch.distributions.Categorical(probs=uniform_prior.unsqueeze(0).expand(symbol_prob.shape[0], -1))
            kl_div = torch.distributions.kl.kl_divergence(symbol_dist, prior_dist).mean()
            total_kl_div += kl_div
        
        # Average across symbols
        total_policy_loss = total_policy_loss / self.policy_network.num_symbols
        total_entropy = total_entropy / self.policy_network.num_symbols
        total_kl_div = total_kl_div / self.policy_network.num_symbols
        
        # Ensure entropy is within valid range for 3 actions
        max_entropy = np.log(3)  # ~1.099
        total_entropy = torch.clamp(total_entropy, 0, max_entropy)
        
        # Add entropy bonus and prior penalty (increased for better exploration)
        entropy_bonus = 0.3  # Increased from 0.1 to encourage more exploration
        lambda_prior = 0.5   # Increased from 0.2 to penalize deviation from uniform
        
        # Cost-aware margin loss: penalize weak-edge trades
        cost_bps = 4.0  # 4 basis points cost
        edge_threshold = 0.001  # 10bp threshold
        margin_loss = 0.0
        
        for i, symbol_prob in enumerate(symbol_probs):
            # Calculate expected edge for each action
            buy_edge = symbol_prob[:, 1] - 0.5 - cost_bps * 1e-4
            sell_edge = 0.5 - symbol_prob[:, 2] - cost_bps * 1e-4
            
            # Penalize trades with edge < threshold
            buy_mask = (symbol_actions[i] == 1) & (buy_edge < edge_threshold)
            if buy_mask.any():
                margin_loss += F.mse_loss(buy_edge[buy_mask], 
                                        torch.full_like(buy_edge[buy_mask], edge_threshold))
            
            sell_mask = (symbol_actions[i] == 2) & (sell_edge < edge_threshold)
            if sell_mask.any():
                margin_loss += F.mse_loss(sell_edge[sell_mask], 
                                        torch.full_like(sell_edge[sell_mask], edge_threshold))
        
        total_policy_loss = total_policy_loss + lambda_prior * total_kl_div - entropy_bonus * total_entropy + 0.1 * margin_loss
        
        # Update policy network with gradient clipping
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        clip_hit = grad_norm > 1.0
        
        self.policy_optimizer.step()
        
        return total_policy_loss.item(), grad_norm.item(), clip_hit
    
    def predict_action(self, state: np.ndarray, stochastic: bool = True, 
                      temperature: float = 1.0, cost_bps: float = 4.0, 
                      edge_threshold: float = 0.001) -> tuple[int, float]:
        """Predict action with temperature scaling and EV thresholding"""
        
        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            symbol_logits = self.policy_network.get_symbol_logits(state_tensor)
            
            # Use primary symbol (symbol 0) for prediction
            primary_logits = symbol_logits[0]
            
            # Temperature scaling for calibrated probabilities
            calibrated_probs = torch.softmax(primary_logits / temperature, dim=-1)
            
            # Calculate expected edge for each action
            # BUY: positive edge if prob > 0.5, SELL: negative edge if prob < 0.5
            buy_edge = calibrated_probs[0, 1] - 0.5 - cost_bps * 1e-4  # BUY action
            sell_edge = 0.5 - calibrated_probs[0, 2] - cost_bps * 1e-4  # SELL action
            
            # Apply EV thresholding with no-trade band
            if buy_edge > edge_threshold:
                action = 1  # BUY
                confidence = calibrated_probs[0, 1].item()
            elif sell_edge > edge_threshold:
                action = 2  # SELL
                confidence = calibrated_probs[0, 2].item()
            else:
                action = 0  # HOLD (no-trade band)
                confidence = calibrated_probs[0, 0].item()
            
            # Assert action is in valid range
            assert 0 <= action <= 2, f"Invalid action {action}, expected 0-2"
        
        return action, confidence
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def save_model(self, path: str):
        """Save trained model"""
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'v_network_state_dict': self.v_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.v_network.load_state_dict(checkpoint['v_network_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.training_history = checkpoint['training_history']


class OfflineRLPipeline:
    """
    Complete offline RL pipeline for trading
    
    Implements the full workflow:
    1. Data preparation and validation
    2. Offline RL training (IQL/CQL)
    3. Policy evaluation and deployment
    """
    
    def __init__(self, config: OfflineRLConfig):
        self.config = config
        self.trainer = OfflineRLTrainer(config) if TORCH_AVAILABLE else None
    
    def prepare_offline_data(self, trading_data: pd.DataFrame) -> tuple[np.ndarray, ...]:
        """
        Prepare data for offline RL training
        
        Args:
            trading_data: DataFrame with columns ['action', 'reward', 'features', 'market_context'] 
                         or multi-symbol format with ['symbols', 'decisions', 'rewards', 'features']
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        
        print("Preparing offline RL data...")
        
        # Extract features and actions
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Check if this is multi-symbol data
        is_multi_symbol = 'symbols' in trading_data.columns
        
        if is_multi_symbol:
            # Multi-symbol data format
            for i in range(len(trading_data) - 1):
                # Current state
                current_features = trading_data.iloc[i]['features']
                if isinstance(current_features, dict):
                    state = np.array(list(current_features.values()))
                else:
                    state = current_features
                
                # For multi-symbol, we'll use a simplified action (average across symbols)
                decisions = trading_data.iloc[i]['decisions']
                symbol_rewards = trading_data.iloc[i]['rewards']
                
                # Calculate average action and reward
                action_values = []
                reward_values = []
                
                for symbol, decision in decisions.items():
                    action_map = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
                    action_values.append(action_map.get(decision['action'], 2))
                    reward_values.append(symbol_rewards[symbol])
                
                # Use most common action and average reward
                action = max(set(action_values), key=action_values.count) if action_values else 2
                reward = np.mean(reward_values) if reward_values else 0.0
                
                # Next state
                next_features = trading_data.iloc[i + 1]['features']
                if isinstance(next_features, dict):
                    next_state = np.array(list(next_features.values()))
                else:
                    next_state = next_features
                
                # Done (end of episode)
                done = i == len(trading_data) - 2
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
        else:
            # Single-symbol data format
            for i in range(len(trading_data) - 1):
                # Current state
                current_features = trading_data.iloc[i]['features']
                if isinstance(current_features, dict):
                    state = np.array(list(current_features.values()))
                else:
                    state = current_features
                
                # Action (convert to integer)
                action_str = trading_data.iloc[i]['action']
                action_map = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
                action = action_map.get(action_str, 2)
                
                # Reward
                reward = trading_data.iloc[i]['reward']
                
                # Next state
                next_features = trading_data.iloc[i + 1]['features']
                if isinstance(next_features, dict):
                    next_state = np.array(list(next_features.values()))
                else:
                    next_state = next_features
                
                # Done (end of episode)
                done = i == len(trading_data) - 2
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def train_offline_rl(self, trading_data: pd.DataFrame) -> dict[str, Any]:
        """Train offline RL model"""
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        # Prepare data
        states, actions, rewards, next_states, dones = self.prepare_offline_data(trading_data)
        
        print(f"Training data: {len(states)} samples")
        print(f"State dimension: {states.shape[1]}")
        
        # Fix action distribution logging
        action_counts = np.bincount(actions, minlength=3)
        action_names = ['BUY', 'SELL', 'HOLD']
        action_dist = {name: count for name, count in zip(action_names, action_counts, strict=False)}
        print(f"Action distribution: {action_dist}")
        print(f"Reward stats: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
        
        # Train model
        if self.config.use_iql:
            results = self.trainer.train_iql(states, actions, rewards, next_states, dones)
        else:
            results = {'error': 'Only IQL implemented currently'}
        
        return results
    
    def evaluate_policy(self, test_data: pd.DataFrame) -> dict[str, Any]:
        """Evaluate trained policy on test data"""
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        print("Evaluating offline RL policy...")
        
        # Prepare test data
        states, actions, rewards, next_states, dones = self.prepare_offline_data(test_data)
        
        # Get predictions
        predictions = []
        confidences = []
        q_values_list = []
        
        for state in states:
            action, confidence = self.trainer.predict_action(state)
            q_values = self.trainer.get_q_values(state)
            
            predictions.append(action)
            confidences.append(confidence)
            q_values_list.append(q_values)
        
        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == actions)
        avg_confidence = np.mean(confidences)
        avg_reward = np.mean(rewards)
        
        # Action distribution
        action_counts = np.bincount(predictions)
        action_names = ['BUY', 'SELL', 'HOLD']
        action_distribution = {action_names[i]: count for i, count in enumerate(action_counts)}
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_reward': avg_reward,
            'action_distribution': action_distribution,
            'predictions': predictions,
            'confidences': confidences,
            'q_values': q_values_list
        }
