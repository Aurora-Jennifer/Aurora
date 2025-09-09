#!/usr/bin/env python3
"""
Real Paper Trading Runner

This script runs paper trading using the ACTUAL trained neural network model.
It loads the real model and makes real predictions.
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
import time
from typing import Dict, Any, Optional
import pickle

import yfinance as yf
from core.utils import setup_logging


class RealPaperTradingAccount:
    """Paper trading account that simulates real trading"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> shares
        self.portfolio_value = initial_capital
        self.trades = []
        self.daily_returns = []
        self.trade_count = 0
        
    def get_position(self, symbol: str) -> float:
        """Get current position size for symbol"""
        return self.positions.get(symbol, 0.0)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                total_value += shares * current_prices[symbol]
        return total_value
    
    def execute_trade(self, symbol: str, action: str, price: float, 
                     max_position_size: float = 0.1, confidence: float = 0.5) -> bool:
        """Execute a trade with confidence-based position sizing"""
        current_shares = self.get_position(symbol)
        current_value = current_shares * price if current_shares > 0 else 0
        
        # Only trade if confidence is above threshold
        if confidence < 0.6:  # Require 60% confidence
            return False
        
        if action == "BUY":
            # Calculate position size based on confidence
            max_investment = self.portfolio_value * max_position_size * confidence
            shares_to_buy = max_investment / price
            
            if shares_to_buy > 0 and self.cash >= shares_to_buy * price:
                self.cash -= shares_to_buy * price
                self.positions[symbol] = current_shares + shares_to_buy
                self.trade_count += 1
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'value': shares_to_buy * price,
                    'confidence': confidence
                }
                self.trades.append(trade)
                return True
                
        elif action == "SELL":
            if current_shares > 0:
                self.cash += current_shares * price
                self.positions[symbol] = 0.0
                self.trade_count += 1
                
                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': current_shares,
                    'price': price,
                    'value': current_shares * price,
                    'confidence': confidence
                }
                self.trades.append(trade)
                return True
        
        return False  # HOLD or failed trade
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.trades:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'num_trades': 0}
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.daily_returns)):
            if self.daily_returns[i-1] > 0:
                ret = (self.daily_returns[i] - self.daily_returns[i-1]) / self.daily_returns[i-1]
                returns.append(ret)
        
        if not returns:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'num_trades': len(self.trades)}
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(self.trades)
        }


class RealPaperTradingRunner:
    """Real paper trading runner that uses actual trained models"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.account = None
        self.pipeline = None
        
    def load_model_and_config(self):
        """Load the actual trained model and configuration"""
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load the actual model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Initialize account
        paper_config = self.config.get('paper_trading', {})
        initial_capital = paper_config.get('initial_capital', 10000)
        self.account = RealPaperTradingAccount(initial_capital)
        
        logging.info(f"Loaded real model for {self.config.get('primary_symbol', 'UNKNOWN')}")
        logging.info(f"Model type: {self.config.get('model_type', 'unknown')}")
        logging.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Load the training pipeline for feature building
        try:
            from core.ml.reward_training_pipeline import RewardBasedTrainingPipeline, TrainingConfig
            
            # Recreate training config from saved metadata
            training_config_dict = self.config.get('training_config', {})
            training_config = TrainingConfig(**training_config_dict)
            
            self.pipeline = RewardBasedTrainingPipeline(training_config)
            logging.info("Loaded training pipeline for feature building")
        except Exception as e:
            logging.warning(f"Could not load training pipeline: {e}")
            self.pipeline = None
    
    def get_latest_data(self, symbol: str, lookback_days: int = 10) -> pd.DataFrame:
        """Get latest market data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        return data
    
    def make_trading_decision(self, data: pd.DataFrame, symbol: str) -> tuple[str, float]:
        """Make a trading decision using the actual trained model"""
        try:
            if self.pipeline is None:
                # Fallback to simple strategy
                return self._simple_strategy(data)
            
            # Build features using the training pipeline
            features = self.pipeline.market_analyzer.build_comprehensive_features(data)
            
            # Get the latest features
            if len(features) == 0:
                return "HOLD", 0.5
            
            latest_features = features.iloc[-1:].values
            
            # Make prediction using the actual model
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(latest_features)
            elif hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(latest_features)
            else:
                # Try to call the model directly
                prediction = self.model(latest_features)
            
            # Convert prediction to action
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 2:  # Probabilities
                    probs = prediction[0]
                    action_idx = np.argmax(probs)
                    confidence = float(probs[action_idx])
                else:  # Direct predictions
                    action_idx = int(prediction[0])
                    confidence = 0.7  # Default confidence
            else:
                action_idx = int(prediction)
                confidence = 0.7
            
            # Map to actions: 0=SELL, 1=HOLD, 2=BUY
            actions = ["SELL", "HOLD", "BUY"]
            if 0 <= action_idx < len(actions):
                action = actions[action_idx]
            else:
                action = "HOLD"
                confidence = 0.5
            
            return action, confidence
            
        except Exception as e:
            logging.error(f"Error making trading decision: {e}")
            return "HOLD", 0.5
    
    def _simple_strategy(self, data: pd.DataFrame) -> tuple[str, float]:
        """Simple fallback strategy"""
        if len(data) < 2:
            return "HOLD", 0.5
        
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        
        price_change = (current_price - previous_price) / previous_price
        
        if price_change > 0.005:  # 0.5% increase
            return "BUY", 0.6
        elif price_change < -0.005:  # 0.5% decrease
            return "SELL", 0.6
        else:
            return "HOLD", 0.5
    
    def run_paper_trading(self, duration_hours: int = 24, check_interval_minutes: int = 60):
        """Run paper trading for specified duration"""
        if not self.model or not self.config:
            raise ValueError("Model and config must be loaded first")
        
        primary_symbol = self.config.get('primary_symbol', 'SPY')
        max_position_size = self.config.get('paper_trading', {}).get('max_position_size', 0.1)
        
        print("=" * 60)
        print("REAL PAPER TRADING SESSION")
        print("=" * 60)
        print(f"Primary Symbol: {primary_symbol}")
        print(f"Model Type: {self.config.get('model_type', 'unknown')}")
        print(f"Duration: {duration_hours} hours")
        print(f"Check Interval: {check_interval_minutes} minutes")
        print(f"Initial Capital: ${self.account.initial_capital:,.2f}")
        print("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Get latest data
                data = self.get_latest_data(primary_symbol, lookback_days=10)
                if data.empty:
                    logging.warning("No data available, skipping this check")
                    time.sleep(check_interval_minutes * 60)
                    continue
                
                # Get latest price
                latest_price = data['Close'].iloc[-1]
                current_time = datetime.now()
                
                # Make trading decision using real model
                action, confidence = self.make_trading_decision(data, primary_symbol)
                
                # Execute trade
                trade_executed = self.account.execute_trade(
                    primary_symbol, action, latest_price, max_position_size, confidence
                )
                
                # Update portfolio value
                current_prices = {primary_symbol: latest_price}
                portfolio_value = self.account.get_portfolio_value(current_prices)
                self.account.portfolio_value = portfolio_value
                self.account.daily_returns.append(portfolio_value)
                
                # Log status
                position = self.account.get_position(primary_symbol)
                print(f"\n[{current_time.strftime('%H:%M:%S')}] {primary_symbol}: ${latest_price:.2f}")
                print(f"  Action: {action} (confidence: {confidence:.2f})")
                print(f"  Position: {position:.2f} shares")
                print(f"  Portfolio Value: ${portfolio_value:,.2f}")
                print(f"  Cash: ${self.account.cash:,.2f}")
                print(f"  Total Trades: {self.account.trade_count}")
                
                if trade_executed:
                    print(f"  ✅ Trade executed!")
                else:
                    print(f"  ⏸️  No trade (HOLD, low confidence, or insufficient funds)")
                
                # Wait for next check
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n\nPaper trading stopped by user")
                break
            except Exception as e:
                logging.error(f"Error during paper trading: {e}")
                time.sleep(check_interval_minutes * 60)
        
        # Final performance report
        self.print_final_report()
    
    def print_final_report(self):
        """Print final performance report"""
        print("\n" + "=" * 60)
        print("REAL PAPER TRADING FINAL REPORT")
        print("=" * 60)
        
        metrics = self.account.get_performance_metrics()
        
        print(f"Initial Capital: ${self.account.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.account.portfolio_value:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Number of Trades: {metrics['num_trades']}")
        
        print("\nTrade History:")
        for trade in self.account.trades[-10:]:  # Show last 10 trades
            print(f"  {trade['timestamp'].strftime('%H:%M:%S')} - {trade['action']} {trade['shares']:.2f} shares at ${trade['price']:.2f} (conf: {trade['confidence']:.2f})")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run real paper trading with a trained model")
    parser.add_argument("--model-path", required=True, help="Path to trained model file")
    parser.add_argument("--config-path", required=True, help="Path to model configuration file")
    parser.add_argument("--duration-hours", type=int, default=24, help="Trading duration in hours")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in minutes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Initialize runner
    runner = RealPaperTradingRunner(args.model_path, args.config_path)
    
    try:
        # Load model and config
        runner.load_model_and_config()
        
        # Run paper trading
        runner.run_paper_trading(
            duration_hours=args.duration_hours,
            check_interval_minutes=args.check_interval
        )
        
    except Exception as e:
        logging.error(f"Paper trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
