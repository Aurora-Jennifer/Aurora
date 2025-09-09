#!/usr/bin/env python3
"""
Train a multi-asset neural network model for paper trading

This script trains on multiple assets to get better signal diversity
and more robust trading decisions.
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
from typing import List, Dict

from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
from core.ml.advanced_models import ModelFactory, ModelConfig
from core.utils import setup_logging


def download_multi_asset_data(symbols: List[str], lookback_days: int = 500) -> Dict[str, pd.DataFrame]:
    """Download historical data for multiple assets"""
    logging.info(f"Downloading {lookback_days} days of data for {len(symbols)} assets")
    
    # Calculate start date with extra buffer
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    
    data_dict = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logging.warning(f"No data for {symbol}, skipping")
                continue
                
            # Clean data
            data = data.dropna()
            
            if len(data) < lookback_days:
                logging.warning(f"Insufficient data for {symbol}: {len(data)} days")
                continue
                
            data_dict[symbol] = data
            logging.info(f"Downloaded {len(data)} days for {symbol}")
            
        except Exception as e:
            logging.error(f"Failed to download {symbol}: {e}")
            continue
    
    return data_dict


def create_multi_asset_training_config(model_type: str = "deep_learning", 
                                     lookback_days: int = None) -> TrainingConfig:
    """Create training configuration for multi-asset model"""
    actual_lookback_days = lookback_days if lookback_days is not None else 100
    
    return TrainingConfig(
        model_type=model_type,
        lookback_days=actual_lookback_days,
        min_trades_for_training=100,  # Higher threshold for multi-asset
        reward_threshold=0.0005,  # Lower threshold for more trades
        risk_free_rate=0.02,
        transaction_cost_bps=5,  # Lower costs for more trading
        validation_split=0.2,
        early_stopping_patience=15,
        max_epochs=150  # More epochs for complex multi-asset model
    )


def train_multi_asset_model(symbols: List[str], model_type: str = "deep_learning", 
                          lookback_days: int = 500, output_dir: str = "models/multi_asset"):
    """Train a model on multiple assets"""
    
    print("=" * 60)
    print("MULTI-ASSET PAPER TRADING MODEL TRAINING")
    print("=" * 60)
    print(f"Assets: {', '.join(symbols)}")
    print(f"Model Type: {model_type}")
    print(f"Lookback Days: {lookback_days}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    try:
        # 1. Download data for all assets
        print(f"\n1. Downloading data for {len(symbols)} assets...")
        data_dict = download_multi_asset_data(symbols, lookback_days)
        
        if len(data_dict) < 2:
            raise ValueError(f"Need at least 2 assets, got {len(data_dict)}")
        
        print(f"   Successfully downloaded data for {len(data_dict)} assets")
        
        # 2. Combine data (simple approach - use SPY as primary, others as features)
        print("\n2. Combining multi-asset data...")
        primary_symbol = "SPY" if "SPY" in data_dict else list(data_dict.keys())[0]
        primary_data = data_dict[primary_symbol].copy()
        
        # Add other assets as additional features
        for symbol, data in data_dict.items():
            if symbol != primary_symbol:
                # Add returns and volatility as features
                returns = data['Close'].pct_change()
                volatility = returns.rolling(20).std()
                
                primary_data[f'{symbol}_return'] = returns
                primary_data[f'{symbol}_vol'] = volatility
        
        # Clean combined data
        primary_data = primary_data.dropna()
        print(f"   Combined dataset: {len(primary_data)} days with {len(primary_data.columns)} features")
        
        # 3. Create training configuration
        print("\n3. Creating multi-asset training configuration...")
        training_config = create_multi_asset_training_config(model_type, lookback_days)
        print(f"   Model type: {training_config.model_type}")
        print(f"   Lookback days: {training_config.lookback_days} (CLI: {lookback_days})")
        print(f"   Min trades: {training_config.min_trades_for_training}")
        print(f"   Reward threshold: {training_config.reward_threshold}")
        
        # 4. Initialize training pipeline
        print("\n4. Initializing multi-asset training pipeline...")
        pipeline = RewardBasedTrainingPipeline(training_config)
        
        # 5. Train model
        print("\n5. Training multi-asset model...")
        result = pipeline.train_reward_based_model(primary_data, primary_symbol)
        
        # 6. Display results
        print("\n6. Training Results:")
        print(f"   Training Time: {result.training_time:.2f} seconds")
        print(f"   Training Accuracy: {result.training_metrics.get('training_accuracy', 0):.3f}")
        print(f"   Validation Reward: {result.validation_metrics.get('total_reward', 0):.4f}")
        print(f"   Validation Accuracy: {result.validation_metrics.get('accuracy', 0):.3f}")
        print(f"   Sharpe Ratio: {result.validation_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Success Rate: {result.strategy_analysis.get('overall_success_rate', 0):.3f}")
        
        # 7. Save model
        print("\n7. Saving multi-asset model...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"multi_asset_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = output_path / f"{model_filename}.pkl"
        
        # Create metadata
        metadata = {
            'symbols': list(data_dict.keys()),
            'primary_symbol': primary_symbol,
            'model_type': model_type,
            'training_config': training_config.__dict__,
            'training_metrics': result.training_metrics,
            'validation_metrics': result.validation_metrics,
            'strategy_analysis': result.strategy_analysis,
            'feature_importance': result.feature_importance,
            'training_time': result.training_time,
            'created_at': datetime.now().isoformat(),
            'data_period': f"{primary_data.index[0].date()} to {primary_data.index[-1].date()}",
            'data_points': len(primary_data),
            'paper_trading_ready': True,
            'multi_asset': True
        }
        
        pipeline.save_model(result.model, str(model_path), metadata)
        
        # 8. Display feature importance
        print("\n8. Top Feature Importance:")
        if result.feature_importance:
            sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:15]:  # Show more features
                print(f"   {feature}: {importance:.4f}")
        
        # 9. Save training summary
        summary = pipeline.get_training_summary()
        summary_path = output_path / f"{model_filename}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n9. Training summary saved to {summary_path}")
        
        # 10. Create paper trading config
        paper_config = {
            'model_path': str(model_path),
            'symbols': list(data_dict.keys()),
            'primary_symbol': primary_symbol,
            'model_type': model_type,
            'trained_at': datetime.now().isoformat(),
            'validation_metrics': result.validation_metrics,
            'strategy_analysis': result.strategy_analysis,
            'paper_trading': {
                'enabled': True,
                'initial_capital': 10000,
                'max_position_size': 0.15,  # Higher for multi-asset
                'stop_loss': 0.03,  # Tighter stops
                'take_profit': 0.06,  # Tighter targets
                'rebalance_frequency': 'daily'
            }
        }
        
        config_path = output_path / f"{model_filename}_paper_config.json"
        with open(config_path, 'w') as f:
            json.dump(paper_config, f, indent=2, default=str)
        
        print(f"10. Paper trading config saved to {config_path}")
        
        print("\n" + "=" * 60)
        print("MULTI-ASSET MODEL TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"\nTrained on {len(data_dict)} assets: {', '.join(data_dict.keys())}")
        print("The model is ready for multi-asset paper trading!")
        
        return {
            'model_path': str(model_path),
            'config_path': str(config_path),
            'summary_path': str(summary_path),
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"\nERROR: Multi-asset training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train a multi-asset neural network model for paper trading")
    parser.add_argument("--symbols", nargs="+", 
                       default=["SPY", "QQQ", "IWM", "GLD", "TLT", "VTI", "EFA", "EEM"],
                       help="List of trading symbols to train on")
    parser.add_argument("--model-type", default="deep_learning", 
                       choices=["deep_learning", "ensemble"], 
                       help="Type of model to train")
    parser.add_argument("--lookback-days", type=int, default=500, 
                       help="Days of historical data to use")
    parser.add_argument("--output-dir", default="models/multi_asset", 
                       help="Output directory for models")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Train model
    result = train_multi_asset_model(
        symbols=args.symbols,
        model_type=args.model_type,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir
    )
    
    print(f"\nMulti-asset training completed successfully!")
    print(f"Model ready for paper trading: {result['model_path']}")


if __name__ == "__main__":
    main()
