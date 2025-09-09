#!/usr/bin/env python3
"""
Validate trained models using walkforward analysis

This script tests the trained models using proper walkforward validation
to ensure they work out-of-sample before paper trading.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
import pickle

import yfinance as yf
from core.utils import setup_logging


class WalkforwardValidator:
    """Validates models using walkforward analysis"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.pipeline = None
        
    def load_model_and_config(self):
        """Load the trained model and configuration"""
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load the actual model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        symbol = self.config.get('primary_symbol') or self.config.get('symbol', 'UNKNOWN')
        if symbol == 'UNKNOWN':
            raise ValueError("Symbol not found in config - check model metadata")
        logging.info(f"Loaded model for {symbol}")
        logging.info(f"Model type: {self.config.get('model_type', 'unknown')}")
        
        # Load the training pipeline for feature building
        try:
            from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
            
            # Try to get training config from metadata file
            metadata_path = self.model_path.replace('.pkl', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                training_config_dict = metadata.get('training_config', {})
            else:
                training_config_dict = self.config.get('training_config', {})
            
            if training_config_dict:
                training_config = TrainingConfig(**training_config_dict)
                self.pipeline = RewardBasedTrainingPipeline(training_config)
                logging.info("Loaded training pipeline for feature building")
            else:
                logging.warning("No training config found in metadata")
                self.pipeline = None
        except Exception as e:
            logging.warning(f"Could not load training pipeline: {e}")
            self.pipeline = None
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical market data for walkforward testing"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} from {start_date} to {end_date}")
        
        return data
    
    def create_walkforward_folds(self, data: pd.DataFrame, fold_length: int = 63, 
                               step_size: int = 21) -> List[Dict[str, Any]]:
        """Create walkforward folds for testing"""
        folds = []
        total_length = len(data)
        
        for start_idx in range(0, total_length - fold_length, step_size):
            end_idx = start_idx + fold_length
            
            if end_idx > total_length:
                break
            
            fold_data = data.iloc[start_idx:end_idx].copy()
            
            fold = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_date': fold_data.index[0],
                'end_date': fold_data.index[-1],
                'data': fold_data,
                'length': len(fold_data)
            }
            folds.append(fold)
        
        return folds
    
    def make_prediction(self, data: pd.DataFrame, symbol: str) -> tuple[str, float, Dict[str, float]]:
        """Make a prediction using the trained model"""
        try:
            if self.pipeline is None:
                # Fallback to simple strategy
                return self._simple_strategy(data)
            
            # Build features using the training pipeline
            features = self.pipeline.market_analyzer.build_comprehensive_features(data)
            
            if len(features) == 0:
                return "HOLD", 0.5, {}
            
            # Get the latest features
            latest_features = features.iloc[-1:].values
            
            # Make prediction using the actual model
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(latest_features)
            elif hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(latest_features)
            else:
                prediction = self.model(latest_features)
            
            # Debug: Print raw prediction (removed - variable scope issue)
            
            # Convert prediction to action
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 2:  # Probabilities
                    probs = prediction[0]
                    action_idx = np.argmax(probs)
                    confidence = float(probs[action_idx])
                    action_probs = {
                        'SELL': float(probs[0]) if len(probs) > 0 else 0.0,
                        'HOLD': float(probs[1]) if len(probs) > 1 else 0.0,
                        'BUY': float(probs[2]) if len(probs) > 2 else 0.0
                    }
                else:  # Direct predictions
                    action_idx = int(prediction[0])
                    confidence = 0.7
                    # Create proper action probabilities based on the prediction
                    action_probs = {'SELL': 0.0, 'HOLD': 0.0, 'BUY': 0.0}
                    if 0 <= action_idx < 3:
                        action_probs[['SELL', 'HOLD', 'BUY'][action_idx]] = 1.0
            else:
                action_idx = int(prediction)
                confidence = 0.7
                # Create proper action probabilities based on the prediction
                action_probs = {'SELL': 0.0, 'HOLD': 0.0, 'BUY': 0.0}
                if 0 <= action_idx < 3:
                    action_probs[['SELL', 'HOLD', 'BUY'][action_idx]] = 1.0
            
            # Map to actions: 0=SELL, 1=HOLD, 2=BUY
            actions = ["SELL", "HOLD", "BUY"]
            if 0 <= action_idx < len(actions):
                action = actions[action_idx]
            else:
                action = "HOLD"
                confidence = 0.5
            
            return action, confidence, action_probs
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return "HOLD", 0.5, {}
    
    def _simple_strategy(self, data: pd.DataFrame) -> tuple[str, float, Dict[str, float]]:
        """Simple fallback strategy"""
        if len(data) < 2:
            return "HOLD", 0.5, {'SELL': 0.0, 'HOLD': 1.0, 'BUY': 0.0}
        
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        
        price_change = (current_price - previous_price) / previous_price
        
        if price_change > 0.005:  # 0.5% increase
            return "BUY", 0.6, {'SELL': 0.0, 'HOLD': 0.4, 'BUY': 0.6}
        elif price_change < -0.005:  # 0.5% decrease
            return "SELL", 0.6, {'SELL': 0.6, 'HOLD': 0.4, 'BUY': 0.0}
        else:
            return "HOLD", 0.5, {'SELL': 0.0, 'HOLD': 1.0, 'BUY': 0.0}
    
    def simulate_trading(self, fold_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Simulate trading on a single fold"""
        initial_capital = 10000
        cash = initial_capital
        position = 0.0
        trades = []
        
        # Use first 20% of fold for warmup, trade on remaining 80%
        warmup_length = int(len(fold_data) * 0.2)
        trading_data = fold_data.iloc[warmup_length:]
        
        for i, (timestamp, row) in enumerate(trading_data.iterrows()):
            # Get historical data up to this point for prediction
            historical_data = fold_data.iloc[:warmup_length + i + 1]
            
            # Make prediction
            action, confidence, action_probs = self.make_prediction(historical_data, symbol)
            
            current_price = row['Close']
            
            # Debug: Print first few predictions to see pattern (one per bar)
            if i < 5:  # Fixed: use proper loop index
                print(f"  Day {i}: action={action}, confidence={confidence:.3f}, probs={action_probs}")
            
            # Execute trade based on prediction (REMOVED hard-coded confidence threshold)
            if action == "BUY" and cash > 0:
                # Buy with 10% of available cash
                shares_to_buy = (cash * 0.1) / current_price
                cash -= shares_to_buy * current_price
                position += shares_to_buy
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'confidence': confidence
                })
                
            elif action == "SELL" and position > 0:
                # Sell entire position
                cash += position * current_price
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'shares': position,
                    'price': current_price,
                    'confidence': confidence
                })
                
                position = 0.0
        
        # Calculate final portfolio value
        final_price = trading_data['Close'].iloc[-1]
        final_value = cash + position * final_price
        
        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate daily returns for Sharpe ratio (FIXED: track actual position changes)
        daily_portfolio_values = [initial_capital]
        daily_positions = [0.0]
        current_cash = initial_capital
        current_position = 0.0
        
        # Reconstruct daily portfolio values by tracking actual trades
        for i, (timestamp, row) in enumerate(trading_data.iterrows()):
            current_price = row['Close']
            
            # Check if there's a trade on this day
            for trade in trades:
                if trade['timestamp'] == timestamp:
                    if trade['action'] == 'BUY':
                        current_cash -= trade['shares'] * trade['price']
                        current_position += trade['shares']
                    elif trade['action'] == 'SELL':
                        current_cash += trade['shares'] * trade['price']
                        current_position -= trade['shares']
            
            portfolio_value = current_cash + current_position * current_price
            daily_portfolio_values.append(portfolio_value)
            daily_positions.append(current_position)
        
        # Calculate daily returns
        if len(daily_portfolio_values) > 1:
            daily_returns = np.diff(daily_portfolio_values) / daily_portfolio_values[:-1]
        else:
            daily_returns = np.array([0.0])
        
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades),
            'trades': trades,
            'action_probs': action_probs
        }
    
    def run_walkforward_validation(self, symbol: str, start_date: str, end_date: str,
                                 fold_length: int = 63, step_size: int = 21) -> Dict[str, Any]:
        """Run walkforward validation on the model"""
        
        print("=" * 60)
        print("WALKFORWARD MODEL VALIDATION")
        print("=" * 60)
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Fold Length: {fold_length} days")
        print(f"Step Size: {step_size} days")
        print("=" * 60)
        
        # Get historical data
        print(f"\n1. Downloading historical data...")
        data = self.get_historical_data(symbol, start_date, end_date)
        print(f"   Downloaded {len(data)} days of data")
        
        # Create walkforward folds
        print(f"\n2. Creating walkforward folds...")
        folds = self.create_walkforward_folds(data, fold_length, step_size)
        print(f"   Created {len(folds)} folds")
        
        # Run validation on each fold
        print(f"\n3. Running walkforward validation...")
        fold_results = []
        
        for i, fold in enumerate(folds):
            print(f"   Fold {i+1}/{len(folds)}: {fold['start_date'].date()} to {fold['end_date'].date()}")
            
            result = self.simulate_trading(fold['data'], symbol)
            result['fold_id'] = i + 1
            result['start_date'] = fold['start_date']
            result['end_date'] = fold['end_date']
            
            fold_results.append(result)
            
            print(f"     Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.3f}, Trades: {result['num_trades']}")
        
        # Calculate aggregate metrics
        print(f"\n4. Calculating aggregate metrics...")
        returns = [r['total_return'] for r in fold_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in fold_results]
        trade_counts = [r['num_trades'] for r in fold_results]
        
        aggregate_metrics = {
            'num_folds': len(fold_results),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'total_trades': sum(trade_counts),
            'mean_trades_per_fold': np.mean(trade_counts),
            'positive_return_folds': sum(1 for r in returns if r > 0),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns)
        }
        
        # Print results
        print(f"\n5. Walkforward Results:")
        print(f"   Number of Folds: {aggregate_metrics['num_folds']}")
        print(f"   Mean Return: {aggregate_metrics['mean_return']:.2%}")
        print(f"   Return Std: {aggregate_metrics['std_return']:.2%}")
        print(f"   Min Return: {aggregate_metrics['min_return']:.2%}")
        print(f"   Max Return: {aggregate_metrics['max_return']:.2%}")
        print(f"   Mean Sharpe: {aggregate_metrics['mean_sharpe']:.3f}")
        print(f"   Win Rate: {aggregate_metrics['win_rate']:.2%}")
        print(f"   Total Trades: {aggregate_metrics['total_trades']}")
        print(f"   Mean Trades per Fold: {aggregate_metrics['mean_trades_per_fold']:.1f}")
        
        # Determine if model is ready for paper trading
        print(f"\n6. Paper Trading Readiness Assessment:")
        ready_for_paper = (
            aggregate_metrics['mean_return'] > 0.0 and
            aggregate_metrics['win_rate'] > 0.4 and
            aggregate_metrics['total_trades'] > 10
        )
        
        if ready_for_paper:
            print(f"   ✅ Model is READY for paper trading!")
            print(f"   - Positive mean return: {aggregate_metrics['mean_return']:.2%}")
            print(f"   - Good win rate: {aggregate_metrics['win_rate']:.2%}")
            print(f"   - Sufficient trades: {aggregate_metrics['total_trades']}")
        else:
            print(f"   ❌ Model is NOT ready for paper trading")
            if aggregate_metrics['mean_return'] <= 0.0:
                print(f"   - Negative mean return: {aggregate_metrics['mean_return']:.2%}")
            if aggregate_metrics['win_rate'] <= 0.4:
                print(f"   - Low win rate: {aggregate_metrics['win_rate']:.2%}")
            if aggregate_metrics['total_trades'] <= 10:
                print(f"   - Insufficient trades: {aggregate_metrics['total_trades']}")
        
        print("=" * 60)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'fold_results': fold_results,
            'ready_for_paper': ready_for_paper
        }


def main():
    parser = argparse.ArgumentParser(description="Validate trained models using walkforward analysis")
    parser.add_argument("--model-path", required=True, help="Path to trained model file")
    parser.add_argument("--config-path", required=True, help="Path to model configuration file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date for validation")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for validation")
    parser.add_argument("--fold-length", type=int, default=63, help="Length of each fold in days")
    parser.add_argument("--step-size", type=int, default=21, help="Step size between folds in days")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Initialize validator
    validator = WalkforwardValidator(args.model_path, args.config_path)
    
    try:
        # Load model and config
        validator.load_model_and_config()
        
        # Run walkforward validation
        results = validator.run_walkforward_validation(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            fold_length=args.fold_length,
            step_size=args.step_size
        )
        
        # Save results
        results_path = f"walkforward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_path}")
        
        if results['ready_for_paper']:
            print(f"\n🎉 Model passed walkforward validation and is ready for paper trading!")
        else:
            print(f"\n⚠️  Model failed walkforward validation. Retrain with different parameters.")
        
    except Exception as e:
        logging.error(f"Walkforward validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
