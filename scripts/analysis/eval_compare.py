#!/usr/bin/env python3
"""
Model Comparison Evaluation Script

Compares performance of asset-specific models against universal baseline.
Implements Clearframe pass/fail gates for model validation.

Usage:
    python scripts/eval_compare.py \
      --symbols BTCUSDT ETHUSDT \
      --model-a models/crypto_v1.onnx \
      --model-b models/universal_v1.onnx \
      --window 2025-07-01:2025-08-15 \
      --metrics sharpe,vol_adj_return,maxdd,turnover \
      --out reports/crypto_v1_vs_universal.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
from core.utils import setup_logging
logger = setup_logging("logs/eval_compare.log", logging.INFO)


class ModelComparator:
    """Compare performance of two models on the same dataset."""
    
    def __init__(self):
        """Initialize comparator."""
        self.models = {}
        self.predictions = {}
        
    def load_model(self, model_path: str, model_name: str) -> None:
        """Load ONNX model for comparison."""
        try:
            import onnxruntime as ort
            
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            session = ort.InferenceSession(str(model_path))
            self.models[model_name] = session
            
            logger.info(f"✅ Loaded model {model_name} from {model_path}")
            
        except ImportError:
            logger.error("onnxruntime not available. Install with: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_data(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load evaluation data."""
        logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Check for existing data files
                data_file = Path(f"data/{symbol}_daily.parquet")
                if data_file.exists():
                    logger.info(f"Loading existing data for {symbol}")
                    df = pd.read_parquet(data_file)
                else:
                    logger.warning(f"No data file found for {symbol}, generating synthetic data")
                    df = self._generate_synthetic_data(symbol, start_date, end_date)
                
                # Filter date range
                df = df.loc[start_date:end_date].copy()
                
                # Add symbol column
                df['symbol'] = symbol
                all_data.append(df)
                
                logger.info(f"Loaded {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded for any symbols")
        
        # Combine all symbol data
        combined_data = pd.concat(all_data, ignore_index=False)
        logger.info(f"Combined dataset: {len(combined_data)} total rows")
        
        return combined_data
    
    def _generate_synthetic_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        logger.warning(f"Generating synthetic data for {symbol}")
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        n_days = len(date_range)
        
        # Set seed based on symbol for reproducibility
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate price series
        base_price = 50000 if 'BTC' in symbol else 3000
        returns = np.random.normal(0.001, 0.03, n_days)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, prices, strict=False)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(1e6, 1e7)
            
            data.append({
                'Open': open_price,
                'High': max(open_price, high, close),
                'Low': min(open_price, low, close),
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'Date'
        
        return df
    
    def generate_predictions(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Generate predictions from all loaded models."""
        logger.info("Generating predictions from models")
        
        # Build features (simplified for comparison)
        features_df = self._build_features(data)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Prepare features for ONNX
                feature_array = features_df.values.astype(np.float32)
                
                # Get predictions
                input_name = model.get_inputs()[0].name
                output = model.run(None, {input_name: feature_array})
                
                # Extract predictions
                pred_values = output[0].flatten()
                
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'prediction': pred_values,
                    'date': features_df.index,
                    'symbol': features_df['symbol'] if 'symbol' in features_df.columns else 'unknown'
                }).set_index('date')
                
                predictions[model_name] = pred_df
                
                logger.info(f"✅ Generated {len(pred_df)} predictions for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to generate predictions for {model_name}: {e}")
                continue
        
        return predictions
    
    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build simple features for model comparison."""
        
        features_list = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 20:
                continue
            
            # Simple technical features
            symbol_data['returns'] = symbol_data['Close'].pct_change()
            symbol_data['sma_5'] = symbol_data['Close'].rolling(5).mean()
            symbol_data['sma_20'] = symbol_data['Close'].rolling(20).mean()
            symbol_data['volatility'] = symbol_data['returns'].rolling(10).std()
            symbol_data['volume_ma'] = symbol_data['Volume'].rolling(5).mean()
            
            # Feature matrix
            features = pd.DataFrame(index=symbol_data.index)
            features['returns_1'] = symbol_data['returns'].shift(1)
            features['returns_5'] = symbol_data['returns'].rolling(5).mean().shift(1)
            features['sma_ratio'] = (symbol_data['Close'] / symbol_data['sma_20']).shift(1)
            features['volatility'] = symbol_data['volatility'].shift(1)
            features['volume_ratio'] = (symbol_data['Volume'] / symbol_data['volume_ma']).shift(1)
            
            # Symbol encoding
            features['symbol'] = symbol
            
            # Drop NaN rows
            features = features.dropna()
            
            features_list.append(features)
        
        if not features_list:
            raise ValueError("No features generated")
        
        # Combine features
        all_features = pd.concat(features_list, ignore_index=False)
        
        # One-hot encode symbols
        symbol_dummies = pd.get_dummies(all_features['symbol'], prefix='symbol')
        all_features = pd.concat([all_features.drop('symbol', axis=1), symbol_dummies], axis=1)
        
        # Fill any remaining NaNs
        all_features = all_features.fillna(0.0)
        
        return all_features
    
    def calculate_metrics(self, predictions: dict[str, pd.DataFrame], data: pd.DataFrame) -> dict[str, dict]:
        """Calculate comparison metrics for all models."""
        logger.info("Calculating performance metrics")
        
        metrics = {}
        
        for model_name, pred_df in predictions.items():
            try:
                # Align predictions with actual data
                aligned_data = []
                
                for symbol in data['symbol'].unique():
                    symbol_data = data[data['symbol'] == symbol].copy()
                    symbol_preds = pred_df[pred_df['symbol'] == symbol] if 'symbol' in pred_df.columns else pred_df
                    
                    # Align by date
                    common_dates = symbol_data.index.intersection(symbol_preds.index)
                    if len(common_dates) < 10:
                        continue
                    
                    aligned_symbol = pd.DataFrame({
                        'actual': symbol_data.loc[common_dates, 'Close'],
                        'predicted': symbol_preds.loc[common_dates, 'prediction'],
                        'symbol': symbol
                    })
                    
                    aligned_data.append(aligned_symbol)
                
                if not aligned_data:
                    logger.warning(f"No aligned data for {model_name}")
                    continue
                
                # Combine aligned data
                aligned_df = pd.concat(aligned_data, ignore_index=False)
                
                # Calculate metrics
                model_metrics = self._calculate_model_metrics(aligned_df)
                metrics[model_name] = model_metrics
                
                logger.info(f"✅ Calculated metrics for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to calculate metrics for {model_name}: {e}")
                continue
        
        return metrics
    
    def _calculate_model_metrics(self, aligned_df: pd.DataFrame) -> dict:
        """Calculate specific metrics for a model."""
        
        # Generate trading signals from predictions
        aligned_df['signal'] = np.sign(aligned_df['predicted'])
        aligned_df['returns'] = aligned_df['actual'].pct_change()
        aligned_df['strategy_returns'] = aligned_df['signal'].shift(1) * aligned_df['returns']
        
        # Remove NaN values
        strategy_returns = aligned_df['strategy_returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid returns'}
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Turnover (rough approximation)
        signal_changes = aligned_df['signal'].diff().fillna(0)
        turnover = (signal_changes != 0).sum() / len(aligned_df)
        
        # Hit rate
        hit_rate = (strategy_returns > 0).mean()
        
        # Volatility-adjusted return
        vol_adj_return = annualized_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'turnover': float(turnover),
            'hit_rate': float(hit_rate),
            'vol_adj_return': float(vol_adj_return),
            'n_trades': int((signal_changes != 0).sum()),
            'n_samples': len(strategy_returns)
        }
    
    def apply_gates(self, metrics: dict[str, dict], model_a: str, model_b: str) -> dict:
        """Apply Clearframe pass/fail gates."""
        logger.info(f"Applying quality gates: {model_a} vs {model_b}")
        
        gates = {
            'sharpe_improvement': {'threshold': 0.10, 'status': 'unknown'},
            'max_drawdown_control': {'threshold': 1.05, 'status': 'unknown'},
            'turnover_reasonable': {'threshold': 2.0, 'status': 'unknown'},
            'overall_pass': False
        }
        
        if model_a not in metrics or model_b not in metrics:
            logger.error("Missing metrics for comparison")
            return gates
        
        metrics_a = metrics[model_a]
        metrics_b = metrics[model_b]
        
        # Gate 1: Sharpe improvement
        sharpe_diff = metrics_a.get('sharpe', 0) - metrics_b.get('sharpe', 0)
        gates['sharpe_improvement']['value'] = sharpe_diff
        gates['sharpe_improvement']['status'] = 'pass' if sharpe_diff >= 0.10 else 'fail'
        
        # Gate 2: Max drawdown control
        dd_ratio = abs(metrics_a.get('max_drawdown', 0)) / max(abs(metrics_b.get('max_drawdown', 1e-6)), 1e-6)
        gates['max_drawdown_control']['value'] = dd_ratio
        gates['max_drawdown_control']['status'] = 'pass' if dd_ratio <= 1.05 else 'fail'
        
        # Gate 3: Turnover reasonable
        turnover_ratio = metrics_a.get('turnover', 0) / max(metrics_b.get('turnover', 1e-6), 1e-6)
        gates['turnover_reasonable']['value'] = turnover_ratio
        gates['turnover_reasonable']['status'] = 'pass' if turnover_ratio <= 2.0 else 'warn'
        
        # Overall pass
        critical_gates = ['sharpe_improvement', 'max_drawdown_control']
        gates['overall_pass'] = all(gates[gate]['status'] == 'pass' for gate in critical_gates)
        
        # Log results
        for gate_name, gate_info in gates.items():
            if gate_name != 'overall_pass':
                status = gate_info['status']
                value = gate_info.get('value', 'N/A')
                threshold = gate_info.get('threshold', 'N/A')
                logger.info(f"Gate {gate_name}: {status.upper()} (value={value:.4f}, threshold={threshold})")
        
        logger.info(f"Overall gate result: {'PASS' if gates['overall_pass'] else 'FAIL'}")
        
        return gates


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare model performance")
    
    parser.add_argument('--symbols', nargs='+', required=True,
                        help='Symbols to evaluate on')
    parser.add_argument('--model-a', required=True,
                        help='Path to first model (ONNX)')
    parser.add_argument('--model-b', required=True,
                        help='Path to second model (ONNX)')
    parser.add_argument('--window', required=True,
                        help='Date window (start:end)')
    parser.add_argument('--metrics', default='sharpe,vol_adj_return,maxdd,turnover',
                        help='Metrics to calculate')
    parser.add_argument('--out', required=True,
                        help='Output comparison report')
    
    args = parser.parse_args()
    
    try:
        # Parse date window
        start_date, end_date = args.window.split(':')
        
        # Initialize comparator
        comparator = ModelComparator()
        
        # Load models
        comparator.load_model(args.model_a, 'model_a')
        comparator.load_model(args.model_b, 'model_b')
        
        # Load data
        data = comparator.load_data(args.symbols, start_date, end_date)
        
        # Generate predictions
        predictions = comparator.generate_predictions(data)
        
        # Calculate metrics
        metrics = comparator.calculate_metrics(predictions, data)
        
        # Apply quality gates
        gates = comparator.apply_gates(metrics, 'model_a', 'model_b')
        
        # Create comparison report
        report = {
            'comparison_config': {
                'model_a_path': args.model_a,
                'model_b_path': args.model_b,
                'symbols': args.symbols,
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': datetime.now().isoformat(),
            },
            'metrics': metrics,
            'quality_gates': gates,
            'recommendation': 'deploy' if gates['overall_pass'] else 'reject'
        }
        
        # Save report
        report_path = Path(args.out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        if gates['overall_pass']:
            logger.info("✅ Model A PASSES quality gates - recommend deployment")
        else:
            logger.info("❌ Model A FAILS quality gates - reject deployment")
        
        logger.info(f"Comparison report saved: {args.out}")
        
        # Exit with appropriate code
        sys.exit(0 if gates['overall_pass'] else 1)
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
