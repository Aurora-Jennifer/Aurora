#!/usr/bin/env python3
"""
Reinforcement Learning Training Script

Trains RL models using reward-based learning with strategy analysis
to identify profitable trading patterns.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from core.rl import (
    QLearningConfig,
    QLearningTrader,
    StrategyAnalyzer,
    TradingRewardCalculator,
    TradingStateManager,
)

# Core imports
from core.utils import setup_logging

from ml.runtime import build_features


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_training_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical data for training"""
    logging.info(f"Downloading {period} of data for {symbol}")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")
        
    logging.info(f"Downloaded {len(data)} bars for {symbol}")
    return data


def create_feature_builder():
    """Create a feature builder compatible with state manager"""
    class FeatureBuilder:
        def build_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
            return build_features(price_data)
    
    return FeatureBuilder()


def train_rl_model(config: dict[str, Any], symbol: str, output_dir: str):
    """
    Train RL model with reward-based learning and strategy analysis
    
    Args:
        config: Configuration dictionary
        symbol: Trading symbol
        output_dir: Output directory for artifacts
    """
    rl_config = config.get('rl', {})
    
    if not rl_config.get('enabled', False):
        logging.warning("RL training disabled in config. Set rl.enabled: true to enable.")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download training data
    data = download_training_data(symbol, period="2y")
    
    # Initialize RL components
    reward_calc = TradingRewardCalculator(rl_config)
    feature_builder = create_feature_builder()
    state_manager = TradingStateManager(feature_builder)
    strategy_analyzer = StrategyAnalyzer(rl_config)
    
    # Initialize Q-learning agent
    q_config = QLearningConfig(
        learning_rate=rl_config.get('learning_rate', 0.1),
        discount_factor=rl_config.get('discount_factor', 0.95),
        epsilon=rl_config.get('exploration_rate', 0.1),
        state_size=rl_config.get('state_size', 100),
        action_size=3
    )
    agent = QLearningTrader(q_config)
    
    # Training parameters
    episodes = rl_config.get('episodes', 1000)
    max_steps = rl_config.get('max_steps_per_episode', 252)
    
    logging.info(f"Starting RL training for {symbol}")
    logging.info(f"Episodes: {episodes}, Max steps per episode: {max_steps}")
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        episode_reward = 0
        episode_length = 0
        
        # Reset for new episode
        reward_calc.reset_analysis()
        state_manager.reset_history()
        
        # Start with random position in data
        start_idx = np.random.randint(100, len(data) - max_steps)
        current_data = data.iloc[start_idx:start_idx + max_steps].copy()
        
        # Initial state
        current_state_dict = state_manager.get_state(
            symbol, current_data.index[0], current_data.iloc[:100]
        )
        current_state = state_manager.get_state_vector(current_state_dict)
        
        for step in range(max_steps):
            if step + 100 >= len(current_data):
                break
                
            # Choose action
            action = agent.choose_action(current_state)
            
            # Execute action and get reward
            # For now, use simple position sizing
            position_size = 0.1 if action == 'BUY' else (-0.1 if action == 'SELL' else 0.0)
            
            # Calculate price change (next bar)
            if step + 1 < len(current_data):
                price_change = (current_data.iloc[step + 1]['Close'] - 
                              current_data.iloc[step]['Close']) / current_data.iloc[step]['Close']
            else:
                price_change = 0.0
            
            # Calculate reward
            reward = reward_calc.calculate_reward(
                action=action,
                price_change=price_change,
                position_size=position_size,
                portfolio_value=10000.0,  # Mock portfolio value
                features=current_state_dict['features'],
                market_context=current_state_dict['market_context'],
                timestamp=current_data.index[step]
            )
            
            episode_reward += reward
            episode_length += 1
            
            # Get next state
            if step + 1 < len(current_data):
                next_data_window = current_data.iloc[step + 1:step + 101]
                next_state_dict = state_manager.get_state(
                    symbol, current_data.index[step + 1], next_data_window
                )
                next_state = state_manager.get_state_vector(next_state_dict)
            else:
                next_state = current_state
                done = True
            
            # Update agent
            done = (step + 1 >= max_steps)
            agent.update(current_state, action, reward, next_state, done)
            
            # Move to next state
            current_state = next_state
            current_state_dict = next_state_dict
            
            # Early stopping if reward is very negative
            if episode_reward < -100:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            logging.info(f"Episode {episode + 1}/{episodes}: "
                        f"Avg reward: {avg_reward:.4f}, "
                        f"Avg length: {avg_length:.1f}, "
                        f"Epsilon: {agent.epsilon:.3f}")
    
    # Analyze strategies
    logging.info("Analyzing trading strategies...")
    all_trades = reward_calc.successful_trades + reward_calc.failed_trades
    strategy_analysis = strategy_analyzer.analyze_trades(all_trades)
    
    # Get top strategies
    top_strategies = reward_calc.get_top_strategies(top_n=10)
    
    # Save results
    results = {
        "training_config": rl_config,
        "symbol": symbol,
        "episodes": episodes,
        "final_stats": agent.get_training_stats(),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "strategy_analysis": strategy_analysis,
        "top_strategies": top_strategies,
        "training_timestamp": datetime.now().isoformat()
    }
    
    # Save model
    model_path = output_path / f"rl_model_{symbol}.pkl"
    agent.save_model(str(model_path))
    
    # Save results
    results_path = output_path / f"rl_training_results_{symbol}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save strategy analysis
    strategy_path = output_path / f"strategy_analysis_{symbol}.json"
    strategy_analyzer.save_analysis(strategy_analysis, str(strategy_path))
    
    # Print summary
    print("\n" + "="*60)
    print("RL TRAINING COMPLETE")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Episodes: {episodes}")
    print(f"Final avg reward: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Success rate: {strategy_analysis.get('statistics', {}).get('success_rate', 0):.2%}")
    print(f"Total trades: {strategy_analysis.get('statistics', {}).get('total_trades', 0)}")
    
    if top_strategies and 'top_features' in top_strategies:
        print("\nTOP STRATEGIES:")
        for i, strategy in enumerate(top_strategies['top_features'][:5]):
            print(f"{i+1}. {strategy['feature']}: "
                  f"Success rate: {strategy['contribution_score']:.3f}")
    
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print(f"Strategy analysis saved to: {strategy_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RL trading model")
    parser.add_argument("--config", default="config/base.yaml", 
                       help="Configuration file path")
    parser.add_argument("--symbol", default="SPY", 
                       help="Trading symbol")
    parser.add_argument("--output-dir", default="models/rl", 
                       help="Output directory for artifacts")
    parser.add_argument("--log-level", default="INFO", 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    train_rl_model(config, args.symbol, args.output_dir)


if __name__ == "__main__":
    main()
