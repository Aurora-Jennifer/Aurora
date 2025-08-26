#!/usr/bin/env python3
"""
Crypto Model Training Script

Trains asset-specific ONNX model for cryptocurrency trading.
Follows Clearframe methodology: deterministic, reproducible, with clear pass/fail gates.

Usage:
    python scripts/train_crypto.py \
      --symbols BTCUSDT ETHUSDT \
      --start 2021-01-01 --end 2025-08-15 \
      --out models/crypto_v1.onnx \
      --report reports/crypto_v1_metrics.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml.build_features import build_matrix

# Configure logging
from core.utils import setup_logging
logger = setup_logging("logs/train_crypto.log", logging.INFO)


class CryptoModelTrainer:
    """Trainer for cryptocurrency-specific models."""
    
    def __init__(self, config: Dict = None):
        """Initialize trainer with configuration."""
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        # Set deterministic seeds
        np.random.seed(self.config.get('random_seed', 42))
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for crypto training."""
        return {
            'random_seed': 42,
            'model_type': 'ridge',  # 'ridge', 'rf', 'xgb'
            'model_params': {
                'ridge': {'alpha': 1.0, 'random_state': 42},
                'rf': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
            },
            'features': {
                'lookback_periods': [5, 10, 20],
                'include_volume': True,
                'include_volatility': True,
                'include_momentum': True,
                'crypto_specific': True,  # Include crypto-specific features
            },
            'validation': {
                'n_splits': 5,
                'test_size': 0.2,
                'gap': 1,  # Gap between train and test
            },
            'quality_gates': {
                'min_r2': 0.01,
                'max_mse': 1.0,
                'min_samples': 1000,
            }
        }
    
    def load_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load and prepare crypto data."""
        logger.info(f"Loading crypto data for {symbols} from {start_date} to {end_date}")
        
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
                    df = self._generate_synthetic_crypto_data(symbol, start_date, end_date)
                
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
    
    def _generate_synthetic_crypto_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic crypto data for testing."""
        logger.warning(f"Generating synthetic data for {symbol} - use real data in production!")
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        n_days = len(date_range)
        
        # Crypto-like price dynamics (higher volatility)
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50000 if 'BTC' in symbol else 3000
        
        # Generate price series with crypto-like characteristics
        returns = np.random.normal(0.001, 0.05, n_days)  # Higher volatility than equities
        prices = [base_price]
        
        for i in range(1, n_days):
            # Add some momentum and mean reversion
            momentum = 0.1 * returns[i-1] if i > 0 else 0
            mean_reversion = -0.05 * (np.log(prices[-1] / base_price))
            
            daily_return = returns[i] + momentum + mean_reversion
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.02)))
            low = close * (1 - abs(np.random.normal(0, 0.02)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(1e6, 1e8)  # High volume for crypto
            
            data.append({
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'Date'
        
        return df
    
    def build_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Build features for crypto models."""
        logger.info("Building crypto-specific features")
        
        features_list = []
        targets_list = []
        
        # Process each symbol separately
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 50:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Build base OHLCV features using simple approach
            logger.info(f"Building features for {symbol}")
            X, y = self._build_simple_features(symbol_data)
            
            if X is None or len(X) == 0:
                logger.warning(f"No features generated for {symbol}")
                continue
            
            # Add crypto-specific features
            if self.config['features'].get('crypto_specific', True):
                X = self._add_crypto_features(X, symbol_data)
            
            # Add symbol identifier (one-hot encoded)
            X[f'symbol_{symbol}'] = 1.0
            
            features_list.append(X)
            targets_list.append(y)
        
        if not features_list:
            raise ValueError("No features generated for any symbols")
        
        # Combine features from all symbols
        X_combined = pd.concat(features_list, ignore_index=True)
        y_combined = pd.concat(targets_list, ignore_index=True)
        
        # Fill missing symbol columns
        symbol_cols = [col for col in X_combined.columns if col.startswith('symbol_')]
        for col in symbol_cols:
            X_combined[col] = X_combined[col].fillna(0.0)
        
        # Remove any remaining NaN values
        mask = ~(X_combined.isna().any(axis=1) | y_combined.isna())
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]
        
        logger.info(f"Features shape: {X_combined.shape}, targets: {len(y_combined)}")
        self.feature_names = list(X_combined.columns)
        
        return X_combined, y_combined
    
    def _add_crypto_features(self, X: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add cryptocurrency-specific features."""
        
        # 24-hour volatility (crypto markets are 24/7)
        if len(price_data) >= 24:
            vol_24h = price_data['close'].rolling(24).std()
            X['volatility_24h'] = vol_24h.reindex(X.index, method='ffill').fillna(0)
        
        # High-frequency momentum (crypto moves fast)
        if len(price_data) >= 4:
            momentum_4h = price_data['close'].pct_change(4)
            X['momentum_4h'] = momentum_4h.reindex(X.index, method='ffill').fillna(0)
        
        # Volume surge indicator (important for crypto)
        if len(price_data) >= 10:
            vol_avg = price_data['volume'].rolling(10).mean()
            vol_surge = price_data['volume'] / vol_avg
            X['volume_surge'] = vol_surge.reindex(X.index, method='ffill').fillna(1.0)
        
        # Price range expansion (crypto has big moves)
        if len(price_data) >= 5:
            price_range = (price_data['high'] - price_data['low']) / price_data['close']
            range_avg = price_range.rolling(5).mean()
            X['range_expansion'] = range_avg.reindex(X.index, method='ffill').fillna(0)
        
        return X
    
    def _build_simple_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Build simple features as fallback."""
        
        # Calculate basic technical indicators
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean() 
        data['volatility'] = data['returns'].rolling(10).std()
        data['volume_ma'] = data['volume'].rolling(5).mean()
        
        # Build feature matrix
        features = pd.DataFrame(index=data.index)
        features['returns_1'] = data['returns'].shift(1)
        features['returns_5'] = data['returns'].rolling(5).mean().shift(1)
        features['sma_ratio'] = (data['close'] / data['sma_20']).shift(1)
        features['volatility'] = data['volatility'].shift(1)
        features['volume_ratio'] = (data['volume'] / data['volume_ma']).shift(1)
        
        # Target: next period returns
        target = data['returns'].shift(-1)
        
        # Remove NaN rows
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        return features, target
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the crypto model with cross-validation."""
        logger.info(f"Training {self.config['model_type']} model")
        
        # Quality gate: minimum samples
        if len(X) < self.config['quality_gates']['min_samples']:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.config['quality_gates']['min_samples']}")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Initialize model
        model_type = self.config['model_type']
        model_params = self.config['model_params'].get(model_type, {})
        
        if model_type == 'ridge':
            self.model = Ridge(**model_params)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config['validation']['n_splits'],
            test_size=int(len(X) * self.config['validation']['test_size']),
            gap=self.config['validation']['gap']
        )
        
        cv_scores = []
        cv_r2_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train fold model
            fold_model = type(self.model)(**model_params)
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_scores.append(mse)
            cv_r2_scores.append(r2)
            
            logger.info(f"Fold {fold}: MSE={mse:.6f}, R²={r2:.6f}")
        
        # Final model on all data
        self.model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred_final = self.model.predict(X_scaled)
        final_mse = mean_squared_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        
        metrics = {
            'cv_mse_mean': float(np.mean(cv_scores)),
            'cv_mse_std': float(np.std(cv_scores)),
            'cv_r2_mean': float(np.mean(cv_r2_scores)),
            'cv_r2_std': float(np.std(cv_r2_scores)),
            'final_mse': float(final_mse),
            'final_r2': float(final_r2),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'model_type': model_type,
            'feature_names': self.feature_names
        }
        
        # Quality gates
        if final_r2 < self.config['quality_gates']['min_r2']:
            raise ValueError(f"Model R² too low: {final_r2:.6f} < {self.config['quality_gates']['min_r2']}")
        
        if final_mse > self.config['quality_gates']['max_mse']:
            raise ValueError(f"Model MSE too high: {final_mse:.6f} > {self.config['quality_gates']['max_mse']}")
        
        logger.info(f"✅ Model training successful: R²={final_r2:.6f}, MSE={final_mse:.6f}")
        
        return metrics
    
    def export_onnx(self, output_path: str) -> None:
        """Export trained model to ONNX format."""
        logger.info(f"Exporting model to {output_path}")
        
        try:
            import onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Create dummy input for ONNX conversion
            n_features = len(self.feature_names)
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert sklearn model to ONNX
            onnx_model = convert_sklearn(
                self.model, 
                initial_types=initial_type,
                target_opset=11
            )
            
            # Save ONNX model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"✅ ONNX model exported successfully to {output_path}")
            
        except ImportError as e:
            logger.error(f"ONNX export failed - missing dependencies: {e}")
            logger.info("Install with: pip install onnx skl2onnx")
            raise
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cryptocurrency-specific model")
    
    parser.add_argument('--symbols', nargs='+', required=True,
                        help='Crypto symbols to train on (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--start', required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--out', required=True,
                        help='Output ONNX model path')
    parser.add_argument('--report', required=True,
                        help='Output metrics report path')
    parser.add_argument('--config',
                        help='Training configuration file (YAML)')
    parser.add_argument('--model-type', choices=['ridge', 'rf'],
                        default='ridge', help='Model type to train')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Initialize trainer with default config
        trainer = CryptoModelTrainer(config)
        
        # Override model type if specified
        trainer.config['model_type'] = args.model_type
        

        
        # Load data
        data = trainer.load_data(args.symbols, args.start, args.end)
        
        # Build features
        X, y = trainer.build_features(data)
        
        # Train model
        metrics = trainer.train_model(X, y)
        
        # Export model
        trainer.export_onnx(args.out)
        
        # Save metrics report
        report = {
            'training_config': {
                'symbols': args.symbols,
                'start_date': args.start,
                'end_date': args.end,
                'model_type': args.model_type,
                'timestamp': datetime.now().isoformat(),
            },
            'metrics': metrics,
            'model_path': str(args.out),
            'feature_names': trainer.feature_names,
        }
        
        # Create report directory
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Training complete! Model: {args.out}, Report: {args.report}")
        logger.info(f"Model metrics: R²={metrics['final_r2']:.6f}, MSE={metrics['final_mse']:.6f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
