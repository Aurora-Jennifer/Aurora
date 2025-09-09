#!/usr/bin/env python3
"""
Train a neural network model for paper trading

This script creates a trained model that can be used for paper trading.
It uses the existing reward-based training pipeline with a deep learning model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
from core.ml.advanced_models import ModelFactory, ModelConfig
from core.utils import setup_logging


def download_training_data(symbol: str, lookback_days: int = 500) -> pd.DataFrame:
    """Download historical data for training"""
    logging.info(f"Downloading {lookback_days} days of data for {symbol}")
    
    # Calculate start date with extra buffer for feature calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    # Clean data
    data = data.dropna()
    
    # Ensure we have enough data
    if len(data) < lookback_days:
        raise ValueError(f"Not enough data: {len(data)} days, need at least {lookback_days}")
    
    logging.info(f"Downloaded {len(data)} days of data for {symbol}")
    return data


def create_training_config(model_type: str = "deep_learning", input_features: int = 100, 
                          lookback_days: int = None) -> TrainingConfig:
    """Create training configuration"""
    # Use CLI value if provided, otherwise default
    actual_lookback_days = lookback_days if lookback_days is not None else 50
    
    return TrainingConfig(
        model_type=model_type,
        lookback_days=actual_lookback_days,  # Use CLI value or default
        min_trades_for_training=50,  # Minimum trades needed
        reward_threshold=0.001,  # Minimum reward threshold
        risk_free_rate=0.02,  # 2% risk-free rate
        transaction_cost_bps=10,  # 10 bps transaction cost
        validation_split=0.2,  # 20% validation split
        early_stopping_patience=10,  # Early stopping patience
        max_epochs=100  # Maximum training epochs
    )


def create_model_config(model_type: str = "deep_learning", input_features: int = 100) -> ModelConfig:
    """Create model configuration"""
    if model_type == "deep_learning":
        return ModelConfig(
            model_type="deep_learning",
            input_features=input_features,
            hidden_layers=[128, 64, 32],  # Deep network
            learning_rate=0.001,
            dropout_rate=0.2,
            regularization=0.01,
            reward_weight=1.0
        )
    elif model_type == "ensemble":
        return ModelConfig(
            model_type="ensemble",
            input_features=input_features,
            hidden_layers=[],
            learning_rate=0.01,
            dropout_rate=0.0,
            regularization=0.01,
            reward_weight=1.0
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_paper_trading_model(symbol: str, model_type: str = "deep_learning", 
                            lookback_days: int = 500, output_dir: str = "models/paper_trading"):
    """Train a model for paper trading"""
    
    print("=" * 60)
    print("PAPER TRADING MODEL TRAINING")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Model Type: {model_type}")
    print(f"Lookback Days: {lookback_days}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    try:
        # 1. Download market data
        print(f"\n1. Downloading market data for {symbol}...")
        data = download_training_data(symbol, lookback_days)
        print(f"   Downloaded {len(data)} days of data")
        
        # 2. Create training configuration
        print("\n2. Creating training configuration...")
        training_config = create_training_config(model_type, lookback_days=lookback_days)
        print(f"   Model type: {training_config.model_type}")
        print(f"   Lookback days: {training_config.lookback_days} (CLI: {lookback_days})")
        print(f"   Min trades: {training_config.min_trades_for_training}")
        
        # 3. Initialize training pipeline
        print("\n3. Initializing training pipeline...")
        pipeline = RewardBasedTrainingPipeline(training_config)
        
        # 4. Train model
        print("\n4. Training paper trading model...")
        result = pipeline.train_reward_based_model(data, symbol)
        
        # 5. Display results
        print("\n5. Training Results:")
        print(f"   Training Time: {result.training_time:.2f} seconds")
        print(f"   Training Accuracy: {result.training_metrics.get('training_accuracy', 0):.3f}")
        print(f"   Validation Reward: {result.validation_metrics.get('total_reward', 0):.4f}")
        print(f"   Validation Accuracy: {result.validation_metrics.get('accuracy', 0):.3f}")
        print(f"   Sharpe Ratio: {result.validation_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Success Rate: {result.strategy_analysis.get('overall_success_rate', 0):.3f}")
        
        # 6. Save model
        print("\n6. Saving model...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{symbol}_{model_type}_paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = output_path / f"{model_filename}.pkl"
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'model_type': model_type,
            'training_config': training_config.__dict__,
            'training_metrics': result.training_metrics,
            'validation_metrics': result.validation_metrics,
            'strategy_analysis': result.strategy_analysis,
            'feature_importance': result.feature_importance,
            'training_time': result.training_time,
            'created_at': datetime.now().isoformat(),
            'data_period': f"{data.index[0].date()} to {data.index[-1].date()}",
            'data_points': len(data),
            'paper_trading_ready': True
        }
        
        pipeline.save_model(result.model, str(model_path), metadata)
        
        # 7. Display feature importance
        print("\n7. Top Feature Importance:")
        if result.feature_importance:
            sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"   {feature}: {importance:.4f}")
        
        # 8. Display strategy analysis
        print("\n8. Strategy Analysis:")
        print(f"   Overall Success Rate: {result.strategy_analysis.get('overall_success_rate', 0):.3f}")
        print(f"   Average Reward: {result.strategy_analysis.get('avg_reward', 0):.4f}")
        print(f"   Reward Std: {result.strategy_analysis.get('reward_std', 0):.4f}")
        
        # 9. Save training summary
        summary = pipeline.get_training_summary()
        summary_path = output_path / f"{model_filename}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n9. Training summary saved to {summary_path}")
        
        # 10. Create paper trading config
        paper_config = {
            'model_path': str(model_path),
            'symbol': symbol,
            'model_type': model_type,
            'trained_at': datetime.now().isoformat(),
            'validation_metrics': result.validation_metrics,
            'strategy_analysis': result.strategy_analysis,
            'paper_trading': {
                'enabled': True,
                'initial_capital': 10000,
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.10
            }
        }
        
        config_path = output_path / f"{model_filename}_paper_config.json"
        with open(config_path, 'w') as f:
            json.dump(paper_config, f, indent=2, default=str)
        
        print(f"10. Paper trading config saved to {config_path}")
        
        print("\n" + "=" * 60)
        print("PAPER TRADING MODEL TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
        print(f"Summary saved to: {summary_path}")
        print("\nThe model is ready for paper trading!")
        
        return {
            'model_path': str(model_path),
            'config_path': str(config_path),
            'summary_path': str(summary_path),
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train a neural network model for paper trading")
    parser.add_argument("--symbol", default="SPY", help="Trading symbol to train on")
    parser.add_argument("--model-type", default="deep_learning", 
                       choices=["deep_learning", "ensemble"], 
                       help="Type of model to train")
    parser.add_argument("--lookback-days", type=int, default=500, 
                       help="Days of historical data to use")
    parser.add_argument("--output-dir", default="models/paper_trading", 
                       help="Output directory for models")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Train model
    result = train_paper_trading_model(
        symbol=args.symbol,
        model_type=args.model_type,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Model ready for paper trading: {result['model_path']}")


if __name__ == "__main__":
    main()
