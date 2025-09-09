#!/usr/bin/env python3
"""
Train Reward-Based Trading Models

Trains models to optimize for actual trading profits using comprehensive market analysis.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
from core.utils import setup_logging


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train reward-based trading models')
    parser.add_argument('--symbol', default='SPY', help='Trading symbol to train on')
    parser.add_argument('--model-type', default='ensemble', choices=['deep_learning', 'ensemble', 'xgboost'], 
                       help='Type of model to train')
    parser.add_argument('--lookback-days', type=int, default=252, help='Days of historical data to use')
    parser.add_argument('--min-trades', type=int, default=100, help='Minimum trades required for training')
    parser.add_argument('--reward-threshold', type=float, default=0.001, help='Minimum reward threshold')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--output-dir', default='models/reward_based', help='Output directory for models')
    parser.add_argument('--config-file', help='Path to training configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    print("=" * 60)
    print("REWARD-BASED TRADING MODEL TRAINING")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Model Type: {args.model_type}")
    print(f"Lookback Days: {args.lookback_days}")
    print(f"Min Trades: {args.min_trades}")
    print(f"Reward Threshold: {args.reward_threshold}")
    print(f"Validation Split: {args.validation_split}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    try:
        # 1. Download market data
        print(f"\n1. Downloading market data for {args.symbol}...")
        data = download_market_data(args.symbol, args.lookback_days)
        print(f"   Downloaded {len(data)} days of data")
        
        # 2. Create training configuration
        print("\n2. Creating training configuration...")
        config = create_training_config(args)
        print(f"   Model type: {config.model_type}")
        print(f"   Lookback days: {config.lookback_days}")
        print(f"   Min trades: {config.min_trades_for_training}")
        
        # 3. Initialize training pipeline
        print("\n3. Initializing training pipeline...")
        pipeline = RewardBasedTrainingPipeline(config)
        
        # 4. Train model
        print("\n4. Training reward-based model...")
        result = pipeline.train_reward_based_model(data, args.symbol)
        
        # 5. Display results
        print("\n5. Training Results:")
        print(f"   Training Time: {result.training_time:.2f} seconds")
        print(f"   Training Accuracy: {result.training_metrics['training_accuracy']:.3f}")
        print(f"   Validation Reward: {result.validation_metrics['total_reward']:.4f}")
        print(f"   Validation Accuracy: {result.validation_metrics['accuracy']:.3f}")
        print(f"   Sharpe Ratio: {result.validation_metrics['sharpe_ratio']:.3f}")
        print(f"   Success Rate: {result.strategy_analysis['overall_success_rate']:.3f}")
        
        # 6. Save model
        print("\n6. Saving model...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{args.symbol}_{args.model_type}_reward_based_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = output_dir / f"{model_filename}.pkl"
        
        # Create metadata
        metadata = {
            'symbol': args.symbol,
            'model_type': args.model_type,
            'training_config': config.__dict__,
            'training_metrics': result.training_metrics,
            'validation_metrics': result.validation_metrics,
            'strategy_analysis': result.strategy_analysis,
            'feature_importance': result.feature_importance,
            'training_time': result.training_time,
            'created_at': datetime.now().isoformat(),
            'data_period': f"{data.index[0].date()} to {data.index[-1].date()}",
            'data_points': len(data)
        }
        
        pipeline.save_model(result.model, str(model_path), metadata)
        
        # 7. Display feature importance
        print("\n7. Top Feature Importance:")
        sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            print(f"   {feature}: {importance:.4f}")
        
        # 8. Display strategy analysis
        print("\n8. Strategy Analysis:")
        print(f"   Overall Success Rate: {result.strategy_analysis['overall_success_rate']:.3f}")
        print(f"   Average Reward: {result.strategy_analysis['avg_reward']:.4f}")
        print(f"   Reward Std: {result.strategy_analysis['reward_std']:.4f}")
        
        if 'action_analysis' in result.strategy_analysis:
            print("   Action Analysis:")
            for action, analysis in result.strategy_analysis['action_analysis'].items():
                print(f"     {action}: {analysis['count']} trades, {analysis['avg_reward']:.4f} avg reward, {analysis['success_rate']:.3f} success rate")
        
        # 9. Save training summary
        summary = pipeline.get_training_summary()
        summary_path = output_dir / f"{model_filename}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n9. Training summary saved to {summary_path}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def download_market_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Download market data for training"""
    
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer for feature calculation
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    # Ensure we have required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean data
    data = data.dropna()
    
    # Ensure we have enough data
    if len(data) < lookback_days:
        raise ValueError(f"Not enough data: {len(data)} days, need at least {lookback_days}")
    
    return data


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments"""
    
    # Load config file if provided
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file) as f:
            config_data = json.load(f)
    else:
        config_data = {}
    
    # Create config with defaults and overrides
    config = TrainingConfig(
        model_type=config_data.get('model_type', args.model_type),
        lookback_days=config_data.get('lookback_days', args.lookback_days),
        min_trades_for_training=config_data.get('min_trades_for_training', args.min_trades),
        reward_threshold=config_data.get('reward_threshold', args.reward_threshold),
        risk_free_rate=config_data.get('risk_free_rate', 0.02),
        transaction_cost_bps=config_data.get('transaction_cost_bps', 10),
        validation_split=config_data.get('validation_split', args.validation_split),
        early_stopping_patience=config_data.get('early_stopping_patience', 10),
        max_epochs=config_data.get('max_epochs', 100)
    )
    
    return config


if __name__ == "__main__":
    main()
