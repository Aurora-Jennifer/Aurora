#!/usr/bin/env python3
"""
Diagnose model predictions to understand why no trades are being made

This script analyzes the model's predictions in detail to identify
why the confidence threshold is preventing trades.
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
import pickle
from typing import Any

import yfinance as yf
from core.utils import setup_logging


class ModelDiagnostics:
    """Diagnose model prediction behavior"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        
    def load_model_and_config(self):
        """Load the trained model and configuration"""
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load the actual model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logging.info(f"Loaded model for {self.config.get('primary_symbol', 'UNKNOWN')}")
        logging.info(f"Model type: {self.config.get('model_type', 'unknown')}")
    
    def get_test_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get test data for diagnosis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        return data
    
    def analyze_predictions(self, data: pd.DataFrame, symbol: str) -> dict[str, Any]:
        """Analyze model predictions in detail"""
        print("=" * 60)
        print("MODEL PREDICTION DIAGNOSTICS")
        print("=" * 60)
        
        predictions = []
        confidences = []
        actions = []
        
        # Analyze predictions over the data
        for i in range(20, len(data)):  # Start after warmup period
            historical_data = data.iloc[:i+1]
            
            try:
                # Try to make prediction
                if hasattr(self.model, 'predict'):
                    prediction = self.model.predict(historical_data.values.reshape(1, -1))
                elif hasattr(self.model, 'predict_proba'):
                    prediction = self.model.predict_proba(historical_data.values.reshape(1, -1))
                else:
                    # Try direct call
                    prediction = self.model(historical_data.values.reshape(1, -1))
                
                # Analyze prediction
                if isinstance(prediction, np.ndarray):
                    if prediction.ndim == 2:  # Probabilities
                        probs = prediction[0]
                        max_prob = float(np.max(probs))
                        action_idx = int(np.argmax(probs))
                        confidence = max_prob
                    else:  # Direct predictions
                        action_idx = int(prediction[0])
                        confidence = 0.7  # Default
                        probs = np.array([0.0, 1.0, 0.0])  # Assume HOLD
                else:
                    action_idx = int(prediction)
                    confidence = 0.7
                    probs = np.array([0.0, 1.0, 0.0])
                
                # Map to actions
                action_names = ["SELL", "HOLD", "BUY"]
                action = action_names[action_idx] if 0 <= action_idx < len(action_names) else "HOLD"
                
                predictions.append({
                    'index': i,
                    'date': historical_data.index[-1],
                    'price': float(historical_data['Close'].iloc[-1]),
                    'action': action,
                    'confidence': confidence,
                    'probs': probs.tolist() if isinstance(probs, np.ndarray) else [0.0, 1.0, 0.0],
                    'action_idx': action_idx
                })
                
                confidences.append(confidence)
                actions.append(action)
                
            except Exception as e:
                logging.warning(f"Error making prediction at index {i}: {e}")
                continue
        
        # Analyze results
        print(f"\nAnalyzed {len(predictions)} predictions")
        
        # Confidence statistics
        confidences = np.array(confidences)
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std: {np.std(confidences):.3f}")
        print(f"  Min: {np.min(confidences):.3f}")
        print(f"  Max: {np.max(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        
        # Action distribution
        action_counts = pd.Series(actions).value_counts()
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            print(f"  {action}: {count} ({count/len(actions)*100:.1f}%)")
        
        # Confidence by action
        print(f"\nConfidence by Action:")
        for action in ["SELL", "HOLD", "BUY"]:
            action_confidences = [p['confidence'] for p in predictions if p['action'] == action]
            if action_confidences:
                print(f"  {action}: mean={np.mean(action_confidences):.3f}, std={np.std(action_confidences):.3f}")
        
        # Trades that would be made with different thresholds
        print(f"\nTrades with Different Confidence Thresholds:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            trades = [p for p in predictions if p['confidence'] > threshold and p['action'] != 'HOLD']
            print(f"  Threshold {threshold}: {len(trades)} trades")
        
        # Show some example predictions
        print(f"\nExample Predictions (last 10):")
        for pred in predictions[-10:]:
            print(f"  {pred['date'].strftime('%Y-%m-%d')}: {pred['action']} (conf: {pred['confidence']:.3f}) at ${pred['price']:.2f}")
        
        return {
            'predictions': predictions,
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            },
            'action_distribution': action_counts.to_dict(),
            'trades_by_threshold': {
                threshold: len([p for p in predictions if p['confidence'] > threshold and p['action'] != 'HOLD'])
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
        }
    
    def recommend_threshold(self, analysis_results: dict[str, Any]) -> float:
        """Recommend optimal confidence threshold"""
        trades_by_threshold = analysis_results['trades_by_threshold']
        confidence_stats = analysis_results['confidence_stats']
        
        print(f"\nThreshold Recommendations:")
        
        # Find threshold that gives reasonable number of trades
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            trades = trades_by_threshold[threshold]
            if trades > 5:  # Want at least 5 trades
                print(f"  Recommended threshold: {threshold} (would generate {trades} trades)")
                return threshold
        
        # If no threshold gives enough trades, use median confidence
        median_conf = confidence_stats['median']
        print(f"  No threshold gives enough trades. Using median confidence: {median_conf:.3f}")
        return median_conf


def main():
    parser = argparse.ArgumentParser(description="Diagnose model predictions")
    parser.add_argument("--model-path", required=True, help="Path to trained model file")
    parser.add_argument("--config-path", required=True, help="Path to model configuration file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test")
    parser.add_argument("--days", type=int, default=100, help="Days of data to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Initialize diagnostics
    diagnostics = ModelDiagnostics(args.model_path, args.config_path)
    
    try:
        # Load model and config
        diagnostics.load_model_and_config()
        
        # Get test data
        data = diagnostics.get_test_data(args.symbol, args.days)
        print(f"Loaded {len(data)} days of data for {args.symbol}")
        
        # Analyze predictions
        results = diagnostics.analyze_predictions(data, args.symbol)
        
        # Recommend threshold
        recommended_threshold = diagnostics.recommend_threshold(results)
        
        # Save results
        results['recommended_threshold'] = recommended_threshold
        results_path = f"model_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDiagnostics saved to: {results_path}")
        print(f"Recommended confidence threshold: {recommended_threshold:.3f}")
        
    except Exception as e:
        logging.error(f"Diagnostics failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
