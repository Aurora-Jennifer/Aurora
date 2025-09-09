"""
Q-Learning for Trading

Implements Q-learning algorithm for trading decisions with exploration
and exploitation strategies.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
import pickle


@dataclass
class QLearningConfig:
    """Configuration for Q-learning algorithm"""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1  # exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    state_size: int = 20
    action_size: int = 3  # BUY, SELL, HOLD


class QLearningTrader:
    """
    Q-learning implementation for trading decisions
    """
    
    def __init__(self, config: QLearningConfig):
        self.config = config
        self.q_table = np.zeros((config.state_size, config.action_size))
        self.epsilon = config.epsilon
        self.training_history = []
        
        # Action mapping
        self.actions = ['BUY', 'SELL', 'HOLD']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        
    def choose_action(self, state: np.ndarray) -> str:
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state: State vector
            
        Returns:
            Action string ('BUY', 'SELL', 'HOLD')
        """
        # Discretize state for Q-table lookup
        state_idx = self._discretize_state(state)
        
        # Epsilon-greedy: explore or exploit
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.config.action_size)
        else:
            # Exploit: best action
            action_idx = np.argmax(self.q_table[state_idx])
            
        return self.idx_to_action[action_idx]
    
    def update(self, state: np.ndarray, action: str, reward: float, 
              next_state: np.ndarray, done: bool = False):
        """
        Update Q-table using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Discretize states
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)
        action_idx = self.action_to_idx[action]
        
        # Q-learning update rule
        current_q = self.q_table[state_idx, action_idx]
        
        if done:
            # Terminal state: no future rewards
            max_next_q = 0
        else:
            # Non-terminal state: max Q-value of next state
            max_next_q = np.max(self.q_table[next_state_idx])
            
        # Q-learning formula
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_idx, action_idx] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Record training step
        self.training_history.append({
            'state_idx': state_idx,
            'action': action,
            'reward': reward,
            'next_state_idx': next_state_idx,
            'epsilon': self.epsilon,
            'q_value': new_q
        })
    
    def _discretize_state(self, state: np.ndarray) -> int:
        """
        Convert continuous state vector to discrete state index
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discrete state index for Q-table lookup
        """
        # Simple discretization: hash the state vector
        # In practice, you might want more sophisticated discretization
        state_hash = hash(tuple(np.round(state, 2)))
        return abs(state_hash) % self.config.state_size
    
    def get_q_values(self, state: np.ndarray) -> dict[str, float]:
        """
        Get Q-values for all actions in a given state
        
        Args:
            state: State vector
            
        Returns:
            Dictionary mapping actions to Q-values
        """
        state_idx = self._discretize_state(state)
        q_values = self.q_table[state_idx]
        
        return {
            action: q_values[idx] 
            for action, idx in self.action_to_idx.items()
        }
    
    def get_best_action(self, state: np.ndarray) -> str:
        """
        Get the best action for a given state (no exploration)
        
        Args:
            state: State vector
            
        Returns:
            Best action string
        """
        q_values = self.get_q_values(state)
        return max(q_values, key=q_values.get)
    
    def get_action_confidence(self, state: np.ndarray, action: str) -> float:
        """
        Get confidence in an action (Q-value normalized)
        
        Args:
            state: State vector
            action: Action string
            
        Returns:
            Confidence score (0-1)
        """
        q_values = self.get_q_values(state)
        action_q = q_values[action]
        max_q = max(q_values.values())
        min_q = min(q_values.values())
        
        if max_q == min_q:
            return 0.5  # Equal confidence if all Q-values are the same
            
        # Normalize to 0-1
        confidence = (action_q - min_q) / (max_q - min_q)
        return confidence
    
    def get_training_stats(self) -> dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary with training statistics
        """
        if not self.training_history:
            return {"error": "No training history"}
            
        recent_history = self.training_history[-100:]  # Last 100 steps
        
        return {
            "total_steps": len(self.training_history),
            "current_epsilon": self.epsilon,
            "avg_reward": np.mean([h['reward'] for h in recent_history]),
            "avg_q_value": np.mean([h['q_value'] for h in recent_history]),
            "exploration_rate": np.mean([h['epsilon'] for h in recent_history]),
            "q_table_stats": {
                "mean": np.mean(self.q_table),
                "std": np.std(self.q_table),
                "min": np.min(self.q_table),
                "max": np.max(self.q_table),
                "sparsity": np.mean(self.q_table == 0)
            }
        }
    
    def save_model(self, filepath: str):
        """Save Q-learning model to file"""
        model_data = {
            'q_table': self.q_table,
            'config': self.config.__dict__,
            'epsilon': self.epsilon,
            'action_mapping': {
                'actions': self.actions,
                'action_to_idx': self.action_to_idx,
                'idx_to_action': self.idx_to_action
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load Q-learning model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.q_table = model_data['q_table']
        self.config = QLearningConfig(**model_data['config'])
        self.epsilon = model_data['epsilon']
        
        # Restore action mappings
        self.actions = model_data['action_mapping']['actions']
        self.action_to_idx = model_data['action_mapping']['action_to_idx']
        self.idx_to_action = model_data['action_mapping']['idx_to_action']
    
    def reset_training(self):
        """Reset training state (keep Q-table)"""
        self.epsilon = self.config.epsilon
        self.training_history = []
    
    def reset_model(self):
        """Reset entire model (clear Q-table)"""
        self.q_table = np.zeros((self.config.state_size, self.config.action_size))
        self.epsilon = self.config.epsilon
        self.training_history = []

