"""
Balanced action sampling for offline RL training.
Addresses the HOLD-heavy bias in the dataset.
"""

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedActionSampler(Sampler):
    """Sampler that balances actions to avoid HOLD-heavy bias"""
    
    def __init__(self, actions, batch_size: int, target_ratios: list[float] = None):
        """
        Args:
            actions: Array of action indices (numpy or torch tensor)
            batch_size: Size of each batch (used for balancing, not for sampling)
            target_ratios: Target ratios for each action [BUY, SELL, HOLD]
        """
        # Convert to numpy if needed
        if hasattr(actions, 'cpu'):
            self.actions = actions.cpu().numpy()
        elif hasattr(actions, 'numpy'):
            self.actions = actions.numpy()
        else:
            self.actions = np.array(actions)
        
        self.batch_size = batch_size
        
        if target_ratios is None:
            target_ratios = [0.33, 0.33, 0.34]  # Slightly favor HOLD
        
        self.target_ratios = target_ratios
        self.num_actions = len(target_ratios)
        
        # Group indices by action
        self.action_indices = {}
        for action_id in range(self.num_actions):
            self.action_indices[action_id] = np.where(self.actions == action_id)[0].tolist()
        
        # Create balanced indices for one epoch
        self.balanced_indices = []
        total_samples = len(self.actions)
        
        # Calculate how many samples we need from each action
        samples_per_action = [int(total_samples * ratio) for ratio in target_ratios]
        
        # Sample from each action group
        for action_id, n_samples in enumerate(samples_per_action):
            if n_samples > 0 and len(self.action_indices[action_id]) > 0:
                # Sample with replacement if needed
                if len(self.action_indices[action_id]) >= n_samples:
                    self.balanced_indices.extend(random.sample(self.action_indices[action_id], n_samples))
                else:
                    self.balanced_indices.extend(random.choices(self.action_indices[action_id], k=n_samples))
        
        # Fill remaining slots if needed
        if len(self.balanced_indices) < total_samples:
            remaining = total_samples - len(self.balanced_indices)
            all_indices = sum(self.action_indices.values(), [])
            self.balanced_indices.extend(random.choices(all_indices, k=remaining))
        
        # Shuffle the balanced indices
        random.shuffle(self.balanced_indices)
        
        print(f"Balanced sampler: {samples_per_action} samples per action, total: {len(self.balanced_indices)}")
    
    def __iter__(self):
        """Generate balanced indices for one epoch"""
        # Yield the pre-computed balanced indices
        for idx in self.balanced_indices:
            yield idx
    
    def __len__(self):
        return len(self.actions)


def create_balanced_dataloader(dataset, actions: np.ndarray, batch_size: int, 
                              target_ratios: list[float] = None) -> torch.utils.data.DataLoader:
    """Create a DataLoader with balanced action sampling"""
    sampler = BalancedActionSampler(actions, batch_size, target_ratios)
    
    # Use single-threaded DataLoader to avoid CUDA worker issues
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=0,  # Single-threaded to avoid CUDA issues
        pin_memory=True
    )


def analyze_action_distribution(actions) -> dict[str, Any]:
    """Analyze the action distribution in the dataset"""
    # Handle both numpy arrays and torch tensors
    if hasattr(actions, 'cpu'):
        actions = actions.cpu().numpy()
    elif hasattr(actions, 'numpy'):
        actions = actions.numpy()
    else:
        actions = np.array(actions)
    
    unique, counts = np.unique(actions, return_counts=True)
    total = len(actions)
    
    distribution = {}
    for action_id, count in zip(unique, counts, strict=False):
        distribution[action_id] = {
            'count': count,
            'percentage': count / total * 100
        }
    
    return distribution


def rebalance_dataset(actions: np.ndarray, target_ratios: list[float] = None) -> np.ndarray:
    """Rebalance dataset by oversampling minority actions"""
    if target_ratios is None:
        target_ratios = [0.33, 0.33, 0.34]
    
    # Calculate current distribution
    unique, counts = np.unique(actions, return_counts=True)
    total = len(actions)
    
    # Calculate target counts
    target_counts = [int(total * ratio) for ratio in target_ratios]
    
    # Create rebalanced indices
    rebalanced_indices = []
    
    for action_id, target_count in enumerate(target_counts):
        if action_id in unique:
            current_count = counts[unique == action_id][0]
            action_indices = np.where(actions == action_id)[0]
            
            if current_count < target_count:
                # Oversample
                oversample_count = target_count - current_count
                oversample_indices = np.random.choice(action_indices, oversample_count, replace=True)
                rebalanced_indices.extend(action_indices.tolist())
                rebalanced_indices.extend(oversample_indices.tolist())
            else:
                # Use all samples
                rebalanced_indices.extend(action_indices.tolist())
    
    return np.array(rebalanced_indices)
