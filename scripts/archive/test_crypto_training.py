#!/usr/bin/env python3
"""
Minimal Crypto Training Test

Quick test to validate the crypto training pipeline without full ONNX dependencies.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_crypto import CryptoModelTrainer

def test_crypto_training():
    """Test the crypto training pipeline."""
    print("🧪 Testing Crypto Training Pipeline")
    
    try:
        # Initialize trainer
        config = {
            'random_seed': 42,
            'model_type': 'ridge',
            'model_params': {'ridge': {'alpha': 1.0, 'random_state': 42}},
            'features': {
                'lookback_periods': [5, 10, 20],
                'include_volume': True,
                'include_volatility': True,
                'include_momentum': True,
                'crypto_specific': True,
            },
            'validation': {'n_splits': 3, 'test_size': 0.2, 'gap': 1},
            'quality_gates': {'min_r2': -1.0, 'max_mse': 10.0, 'min_samples': 10}
        }
        trainer = CryptoModelTrainer(config)
        
        # Create synthetic data for two symbols
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        symbols_data = []
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            # Generate realistic crypto price data
            np.random.seed(42 + hash(symbol) % 100)
            base_price = 50000 if 'BTC' in symbol else 3000
            
            returns = np.random.normal(0.001, 0.03, 100)
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))
            
            data = pd.DataFrame({
                'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': np.random.uniform(1e6, 1e7, 100),
                'symbol': symbol
            }, index=dates)
            
            symbols_data.append(data)
        
        # Combine data
        combined_data = pd.concat(symbols_data, ignore_index=False)
        print(f"✅ Created test data: {len(combined_data)} rows, {len(combined_data['symbol'].unique())} symbols")
        
        # Build features
        X, y = trainer.build_features(combined_data)
        print(f"✅ Built features: X.shape={X.shape}, y.shape={y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        # Train model
        metrics = trainer.train_model(X, y)
        print(f"✅ Model trained successfully!")
        print(f"   Final R²: {metrics['final_r2']:.4f}")
        print(f"   Final MSE: {metrics['final_mse']:.6f}")
        print(f"   CV R² mean: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        # Test prediction
        test_pred = trainer.model.predict(trainer.scaler.transform(X.iloc[:5]))
        print(f"✅ Model prediction test: {test_pred[:3]}")
        
        # Save simple model (pickle format for testing)
        import pickle
        simple_model = {
            'model': trainer.model,
            'scaler': trainer.scaler,
            'feature_names': trainer.feature_names,
            'metrics': metrics
        }
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        with open('models/crypto_test.pkl', 'wb') as f:
            pickle.dump(simple_model, f)
        
        print(f"✅ Simple model saved: models/crypto_test.pkl")
        
        # Test model loading and prediction
        with open('models/crypto_test.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        test_features = X.iloc[:3]
        scaled_features = loaded_model['scaler'].transform(test_features)
        predictions = loaded_model['model'].predict(scaled_features)
        
        print(f"✅ Model loading test successful: {predictions}")
        print(f"✅ Feature names: {len(loaded_model['feature_names'])} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_crypto_training()
    if success:
        print("\n🎉 Crypto training pipeline test PASSED!")
        print("Ready for real model training and ONNX export.")
    else:
        print("\n💥 Crypto training pipeline test FAILED!")
    
    sys.exit(0 if success else 1)
