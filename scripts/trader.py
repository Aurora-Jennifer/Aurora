#!/usr/bin/env python3
"""
Simplified Trader - One script to rule them all
Train reward-based models and run paper trading
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline
from core.ml.market_analyzer import ComprehensiveMarketAnalyzer
import yfinance as yf
import json
from datetime import datetime


class SimplifiedTrader:
    """One-stop shop for training and trading"""
    
    def __init__(self, config_path: str = "config/reward_training.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.models_dir = Path("models/reward_based")
        self.models_dir.mkdir(exist_ok=True)
        
    def _load_config(self):
        """Load configuration"""
        import yaml
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def train(self, symbol: str, model_type: str = "profit_optimized", lookback_days: int = 200, min_trades: int = 50, symbols: list[str] | None = None):
        """Train a profit-optimized model"""
        print(f"🚀 Training {model_type} model for {symbol}")
        print(f"   Lookback: {lookback_days} days, Min trades: {min_trades}")
        
        # Download data
        print("📊 Downloading market data...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{lookback_days}d")
        print(f"   Downloaded {len(data)} days of data")
        
        # Create training config
        from core.ml.reward_training_pipeline import TrainingConfig
        
        training_config = TrainingConfig(
            model_type=model_type,
            lookback_days=min(lookback_days // 2, 50),  # Use half of lookback days, max 50
            min_trades_for_training=min_trades,
            reward_threshold=self.config.get('reward_threshold', 0.001),
            risk_free_rate=self.config.get('risk_free_rate', 0.02),
            transaction_cost_bps=self.config.get('transaction_cost_bps', 10),
            validation_split=self.config.get('validation_split', 0.2),
            early_stopping_patience=self.config.get('early_stopping_patience', 10),
            max_epochs=self.config.get('max_epochs', 100)
        )
        
        # Initialize pipeline
        pipeline = RewardBasedTrainingPipeline(training_config)
        
        # Train model
        print("🧠 Training model...")
        if model_type == "multi_symbol":
            # Multi-symbol training
            if not symbols:
                raise ValueError("Multi-symbol training requires symbols argument")
            result = pipeline.train_multi_symbol_system(symbols, lookback_days)
        elif model_type == "enhanced_offline_rl":
            result = pipeline.train_enhanced_offline_rl(data, symbol)
        elif model_type == "profit_optimized":
            result = pipeline.train_profit_optimized_model(data, symbol)
        else:
            result = pipeline.train_reward_based_model(data, symbol)
        
        # Get features count from the result
        features_count = result.training_metrics.get('features_count', 114)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{symbol}_{model_type}_reward_based_{timestamp}"
        
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}.json"
        summary_path = self.models_dir / f"{model_name}_summary.json"
        
        # Save model and metadata
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(result.model, f)
        
        metadata = {
            'symbol': symbol,
            'model_type': model_type,
            'training_date': timestamp,
            'lookback_days': lookback_days,
            'min_trades': min_trades,
            'features_count': features_count,
            'training_time': result.training_time,
            'validation_reward': result.validation_metrics.get('reward', 0),
            'validation_accuracy': result.validation_metrics.get('accuracy', 0),
            'sharpe_ratio': result.validation_metrics.get('sharpe_ratio', 0),
            'success_rate': result.validation_metrics.get('success_rate', 0)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Convert strategy analysis to JSON-serializable format
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        strategy_analysis = convert_to_json_serializable(result.strategy_analysis)
        
        # Convert to JSON-serializable format
        strategy_analysis_serializable = self._convert_to_json_serializable(strategy_analysis)
        with open(summary_path, 'w') as f:
            json.dump(strategy_analysis_serializable, f, indent=2)
        
        print(f"✅ Model saved: {model_path}")
        print("📊 Results:")
        print(f"   Training time: {result.training_time:.2f}s")
        print(f"   Validation reward: {result.validation_metrics.get('total_reward', 0):.4f}")
        print(f"   Success rate: {result.validation_metrics.get('success_rate', 0):.2%}")
        print(f"   Sharpe ratio: {result.validation_metrics.get('sharpe_ratio', 0):.3f}")
        
        return model_path, metadata
    
    def _convert_to_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        if hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            return {key: self._convert_to_json_serializable(value) for key, value in obj.__dict__.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def list_models(self):
        """List available trained models"""
        print("📋 Available Models:")
        if not self.models_dir.exists():
            print("   No models found")
            return []
        
        models = []
        for file in self.models_dir.glob("*.json"):
            if not file.name.endswith("_summary.json"):
                try:
                    with open(file) as f:
                        metadata = json.load(f)
                    models.append((file, metadata))
                except:
                    continue
        
        if not models:
            print("   No models found")
            return []
        
        # Sort by training date
        models.sort(key=lambda x: x[1].get('training_date', ''), reverse=True)
        
        for i, (file, metadata) in enumerate(models):
            print(f"   {i+1}. {metadata['symbol']} ({metadata['model_type']})")
            print(f"      Date: {metadata.get('training_date', 'Unknown')}")
            print(f"      Reward: {metadata.get('validation_reward', 0):.4f}")
            print(f"      Success: {metadata.get('success_rate', 0):.2%}")
            print()
        
        return models
    
    def paper_trade(self, model_path: str, symbol: str, days: int = 30):
        """Run paper trading with a trained model"""
        print(f"📈 Paper Trading {symbol} with {model_path}")
        
        # Load model
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get recent data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        
        # Build features
        analyzer = ComprehensiveMarketAnalyzer({})
        features = analyzer.build_comprehensive_features(data)
        
        if len(features) == 0:
            print("❌ No features generated")
            return
        
        # Use all features (feature selection disabled for now)
        X = features.values
        print(f"✅ Using all {features.shape[1]} features for trading")
        
        # Get predictions (handle different model types)
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'predict_action'):
            # OfflineRLPipeline
            predictions = []
            confidence = []
            for i in range(len(X)):
                action, conf = model.trainer.predict_action(X[i])
                predictions.append(action)
                confidence.append(conf)
            predictions = np.array(predictions)
            confidence = np.array(confidence)
        elif hasattr(model, 'predict_with_exploration'):
            # ExplorationModel
            predictions, confidence = model.predict_with_exploration(X)
        elif hasattr(model, 'predict_with_confidence'):
            # Standard model
            predictions, confidence = model.predict_with_confidence(X)
        else:
            raise ValueError("Model does not have a prediction method")
        
        # Simulate trading
        print("🎯 Trading Simulation:")
        print("   Date       | Action | Confidence | Price")
        print("   " + "-" * 50)
        
        portfolio_value = 10000  # Starting with $10k
        position = 0  # 0 = no position, 1 = long, -1 = short
        
        for i, (date, pred, conf, price) in enumerate(zip(features.index, predictions, confidence, data['Close'], strict=False)):
            action = ['SELL', 'HOLD', 'BUY'][pred]
            
            if action == 'BUY' and position <= 0 and conf > 0.05:  # Lower confidence threshold
                position = 1
                portfolio_value = portfolio_value * (price / data['Close'].iloc[i-1] if i > 0 else 1)
                print(f"   {date.strftime('%Y-%m-%d')} | {action:4} | {conf:.3f}     | ${price:.2f}")
            elif action == 'SELL' and position >= 0 and conf > 0.05:  # Lower confidence threshold
                position = -1
                portfolio_value = portfolio_value * (data['Close'].iloc[i-1] / price if i > 0 else 1)
                print(f"   {date.strftime('%Y-%m-%d')} | {action:4} | {conf:.3f}     | ${price:.2f}")
        
        # Final portfolio value
        final_price = data['Close'].iloc[-1]
        if position == 1:
            portfolio_value = portfolio_value * (final_price / data['Close'].iloc[-2])
        elif position == -1:
            portfolio_value = portfolio_value * (data['Close'].iloc[-2] / final_price)
        
        print("\\n💰 Final Results:")
        print("   Starting value: $10,000")
        print(f"   Final value: ${portfolio_value:.2f}")
        print(f"   Return: {((portfolio_value / 10000) - 1) * 100:.2f}%")
        print(f"   Buy & Hold: {((final_price / data['Close'].iloc[0]) - 1) * 100:.2f}%")
    
    def quick_train(self, symbol: str = "SPY"):
        """Quick training with default settings"""
        return self.train(symbol, "ensemble", 200, 50)
    
    def quick_trade(self, symbol: str = "SPY"):
        """Quick paper trading with latest model"""
        models = self.list_models()
        if not models:
            print("❌ No models found. Run training first.")
            return
        
        # Use latest model
        latest_model = models[0][0]
        self.paper_trade(str(latest_model).replace('.json', '.pkl'), symbol)


def main():
    parser = argparse.ArgumentParser(description="Simplified Trader - Train and Trade")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--symbol', default='SPY', help='Symbol to train on (for single-symbol models)')
    train_parser.add_argument('--symbols', nargs='+', help='Multiple symbols for multi-symbol training (e.g., --symbols SPY QQQ IWM)')
    train_parser.add_argument('--model-type', default='enhanced_offline_rl', choices=['enhanced_offline_rl', 'multi_symbol', 'profit_optimized', 'ensemble', 'deep_learning', 'xgboost'], help='Model type')
    train_parser.add_argument('--lookback-days', type=int, default=200, help='Days of historical data')
    train_parser.add_argument('--min-trades', type=int, default=20, help='Minimum trades for training')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Paper trade with a model')
    trade_parser.add_argument('--model', help='Model file path')
    trade_parser.add_argument('--symbol', default='SPY', help='Symbol to trade')
    trade_parser.add_argument('--days', type=int, default=30, help='Days to simulate')
    
    # List command
    subparsers.add_parser('list', help='List available models')
    
    # Quick commands
    subparsers.add_parser('quick-train', help='Quick training with defaults')
    subparsers.add_parser('quick-trade', help='Quick paper trading with latest model')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    trader = SimplifiedTrader()
    
    if args.command == 'train':
        trader.train(args.symbol, args.model_type, args.lookback_days, args.min_trades, getattr(args, 'symbols', None))
    elif args.command == 'trade':
        if not args.model:
            print("❌ Model path required for trading")
            return
        trader.paper_trade(args.model, args.symbol, args.days)
    elif args.command == 'list':
        trader.list_models()
    elif args.command == 'quick-train':
        trader.quick_train()
    elif args.command == 'quick-trade':
        trader.quick_trade()


if __name__ == "__main__":
    main()
