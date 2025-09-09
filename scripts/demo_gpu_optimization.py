#!/usr/bin/env python3
"""
Demo GPU Optimization - Hardware Performance Test

Tests the GPU-optimized training pipeline with synthetic data
to demonstrate hardware utilization and performance gains.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import sys
import os
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.xgb_gpu import XGBoostGPU, train_global_xgb_gpu
from scripts.train_scheduler import TaskScheduler, HardwareMonitor

# Try to import CatBoost, skip if not available
try:
    from ml.models.catboost_gpu import CatBoostGPU, train_global_catboost_gpu
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available, skipping CatBoost tests")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_panel_data(n_symbols: int = 100, 
                               n_days: int = 1000,
                               n_features: int = 50) -> pd.DataFrame:
    """Create synthetic panel data for testing."""
    logger.info(f"Creating synthetic panel data: {n_symbols} symbols, {n_days} days, {n_features} features")
    
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Create symbols
    symbols = [f"SYMBOL_{i:03d}" for i in range(n_symbols)]
    
    # Create panel data
    data = []
    
    for symbol in symbols:
        # Generate features
        features = {}
        
        # Returns
        features['ret1'] = np.random.normal(0, 0.02, n_days)
        features['ret5'] = np.random.normal(0, 0.03, n_days)
        features['ret10'] = np.random.normal(0, 0.04, n_days)
        
        # Volatility
        features['vol_5'] = np.random.uniform(0.01, 0.05, n_days)
        features['vol_20'] = np.random.uniform(0.01, 0.05, n_days)
        
        # Technical indicators
        features['bb_position'] = np.random.uniform(0, 1, n_days)
        features['rsi_14'] = np.random.uniform(20, 80, n_days)
        features['ma_fast'] = np.random.uniform(0.95, 1.05, n_days)
        features['ma_slow'] = np.random.uniform(0.95, 1.05, n_days)
        features['momentum_5'] = np.random.normal(0, 0.02, n_days)
        
        # Additional features
        for i in range(n_features - 9):
            features[f'feature_{i}'] = np.random.normal(0, 1, n_days)
        
        # Create target
        features['ret_fwd_5'] = np.random.normal(0, 0.01, n_days)
        
        # Add symbol and date
        features['symbol'] = symbol
        features['date'] = dates
        
        # Create DataFrame for this symbol
        symbol_df = pd.DataFrame(features)
        data.append(symbol_df)
    
    # Combine all symbols
    panel_df = pd.concat(data, ignore_index=True)
    
    logger.info(f"Created panel data: {len(panel_df)} rows, {len(panel_df.columns)} columns")
    
    return panel_df


def test_gpu_models(panel_df: pd.DataFrame, output_dir: Path) -> Dict[str, any]:
    """Test GPU-optimized models."""
    logger.info("Testing GPU-optimized models...")
    
    results = {}
    
    # XGBoost GPU config
    xgb_config = {
        'params': {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_bin': 512,
            'single_precision_histogram': True,
            'nthread': 8,
            'random_state': 42
        },
        'data': {
            'dtype': 'float32',
            'use_quantile_dmatrix': True,
            'cache_data': True
        }
    }
    
    # CatBoost GPU config
    catboost_config = {
        'params': {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 1000,
            'l2_leaf_reg': 3.0,
            'border_count': 128,
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'od_wait': 100,
            'random_seed': 42,
            'verbose': 100
        },
        'data': {
            'dtype': 'float32',
            'cache_data': True
        }
    }
    
    # Test XGBoost GPU
    try:
        logger.info("Testing XGBoost GPU...")
        start_time = time.time()
        
        xgb_results = train_global_xgb_gpu(panel_df, xgb_config, output_dir)
        xgb_time = time.time() - start_time
        
        results['xgboost_gpu'] = {
            **xgb_results,
            'training_time': xgb_time,
            'samples_per_second': xgb_results['n_samples'] / xgb_time
        }
        
        logger.info(f"XGBoost GPU: {xgb_time:.2f}s, {xgb_results['n_samples']/xgb_time:.0f} samples/s")
        
    except Exception as e:
        logger.error(f"XGBoost GPU failed: {e}")
        results['xgboost_gpu'] = {'error': str(e)}
    
    # Test CatBoost GPU (if available)
    if CATBOOST_AVAILABLE:
        try:
            logger.info("Testing CatBoost GPU...")
            start_time = time.time()
            
            catboost_results = train_global_catboost_gpu(panel_df, catboost_config, output_dir)
            catboost_time = time.time() - start_time
            
            results['catboost_gpu'] = {
                **catboost_results,
                'training_time': catboost_time,
                'samples_per_second': catboost_results['n_samples'] / catboost_time
            }
            
            logger.info(f"CatBoost GPU: {catboost_time:.2f}s, {catboost_results['n_samples']/catboost_time:.0f} samples/s")
            
        except Exception as e:
            logger.error(f"CatBoost GPU failed: {e}")
            results['catboost_gpu'] = {'error': str(e)}
    else:
        results['catboost_gpu'] = {'error': 'CatBoost not available'}
    
    return results


def test_cpu_models(panel_df: pd.DataFrame, output_dir: Path) -> Dict[str, any]:
    """Test CPU models for comparison."""
    logger.info("Testing CPU models for comparison...")
    
    results = {}
    
    # Simple Ridge regression for comparison
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    feature_cols = [col for col in panel_df.columns if col.startswith(('ret', 'vol', 'bb', 'rsi', 'ma', 'momentum'))]
    target_col = 'ret_fwd_5'
    
    X = panel_df[feature_cols].values
    y = panel_df[target_col].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Test Ridge
    try:
        logger.info("Testing Ridge CPU...")
        start_time = time.time()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Ridge
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_scaled, y)
        
        # Make predictions
        predictions = ridge.predict(X_scaled)
        
        # Calculate performance
        ic = np.corrcoef(predictions, y)[0, 1]
        mse = np.mean((predictions - y) ** 2)
        
        ridge_time = time.time() - start_time
        
        results['ridge_cpu'] = {
            'ic': float(ic),
            'mse': float(mse),
            'n_samples': len(y),
            'n_features': X.shape[1],
            'training_time': ridge_time,
            'samples_per_second': len(y) / ridge_time
        }
        
        logger.info(f"Ridge CPU: {ridge_time:.2f}s, {len(y)/ridge_time:.0f} samples/s")
        
    except Exception as e:
        logger.error(f"Ridge CPU failed: {e}")
        results['ridge_cpu'] = {'error': str(e)}
    
    return results


def demo_gpu_optimization():
    """Demonstrate GPU optimization capabilities."""
    logger.info("=== GPU Optimization Demo ===")
    
    # Create output directory
    output_dir = Path("models/gpu_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Monitor hardware
    monitor = HardwareMonitor()
    
    # Create synthetic data
    panel_df = create_synthetic_panel_data(n_symbols=50, n_days=500, n_features=30)
    
    # Test GPU models
    gpu_results = test_gpu_models(panel_df, output_dir)
    
    # Test CPU models
    cpu_results = test_cpu_models(panel_df, output_dir)
    
    # Combine results
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': {
            'cpu_cores': monitor.cpu_count,
            'memory_gb': monitor.memory_gb,
            'gpu_available': monitor.gpu_available
        },
        'data_info': {
            'n_symbols': panel_df['symbol'].nunique(),
            'n_days': panel_df['date'].nunique(),
            'n_samples': len(panel_df),
            'n_features': len([col for col in panel_df.columns if col.startswith(('ret', 'vol', 'bb', 'rsi', 'ma', 'momentum'))])
        },
        'gpu_models': gpu_results,
        'cpu_models': cpu_results
    }
    
    # Print results
    print("\n🚀 **GPU OPTIMIZATION DEMO RESULTS**")
    print("=" * 60)
    
    print(f"\n💻 **Hardware**")
    print(f"   CPU cores: {monitor.cpu_count}")
    print(f"   Memory: {monitor.memory_gb:.1f} GB")
    print(f"   GPU available: {monitor.gpu_available}")
    
    print(f"\n📊 **Data**")
    print(f"   Symbols: {all_results['data_info']['n_symbols']}")
    print(f"   Days: {all_results['data_info']['n_days']}")
    print(f"   Samples: {all_results['data_info']['n_samples']:,}")
    print(f"   Features: {all_results['data_info']['n_features']}")
    
    print(f"\n🚀 **GPU Models**")
    for model_name, result in gpu_results.items():
        if 'error' in result:
            print(f"   {model_name}: ❌ {result['error']}")
        else:
            print(f"   {model_name}:")
            print(f"     IC: {result['ic']:.4f}")
            print(f"     Time: {result['training_time']:.2f}s")
            print(f"     Speed: {result['samples_per_second']:.0f} samples/s")
    
    print(f"\n💻 **CPU Models**")
    for model_name, result in cpu_results.items():
        if 'error' in result:
            print(f"   {model_name}: ❌ {result['error']}")
        else:
            print(f"   {model_name}:")
            print(f"     IC: {result['ic']:.4f}")
            print(f"     Time: {result['training_time']:.2f}s")
            print(f"     Speed: {result['samples_per_second']:.0f} samples/s")
    
    # Calculate speedup
    if 'xgboost_gpu' in gpu_results and 'ridge_cpu' in cpu_results:
        if 'error' not in gpu_results['xgboost_gpu'] and 'error' not in cpu_results['ridge_cpu']:
            gpu_speed = gpu_results['xgboost_gpu']['samples_per_second']
            cpu_speed = cpu_results['ridge_cpu']['samples_per_second']
            speedup = gpu_speed / cpu_speed
            
            print(f"\n⚡ **Performance Comparison**")
            print(f"   GPU vs CPU speedup: {speedup:.1f}x")
            print(f"   GPU: {gpu_speed:.0f} samples/s")
            print(f"   CPU: {cpu_speed:.0f} samples/s")
    
    # Save results
    with open(output_dir / 'gpu_demo_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Demo results saved to {output_dir / 'gpu_demo_results.json'}")
    
    print(f"\n✅ **GPU Optimization Demo Complete**")
    print(f"   Ready for production-scale training")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Demo GPU Optimization')
    args = parser.parse_args()
    
    demo_gpu_optimization()


if __name__ == '__main__':
    main()
