#!/bin/bash
# Quick Start Production Training - 1 Hour Budget
# Optimized for 3080 + i7-11700K + 32GB

set -e

echo "🚀 PRODUCTION TRAINING QUICK START"
echo "=================================="

# Set environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export XGBOOST_NUM_THREADS=8

echo "✅ Environment variables set"

# Create output directory
OUTPUT_DIR="results/production_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found, GPU may not be available"
fi

# Check XGBoost GPU support
echo "🔍 Checking XGBoost GPU support..."
python -c "
import xgboost as xgb
print(f'XGBoost version: {xgb.__version__}')
try:
    import xgboost as xgb
    # Test GPU availability
    import numpy as np
    X = np.random.rand(100, 10).astype('float32')
    y = np.random.rand(100).astype('float32')
    dtrain = xgb.DMatrix(X, label=y)
    params = {'tree_method': 'gpu_hist', 'max_depth': 3, 'n_estimators': 10}
    model = xgb.train(params, dtrain, num_boost_round=10)
    print('✅ XGBoost GPU support confirmed')
except Exception as e:
    print(f'❌ XGBoost GPU test failed: {e}')
"

# Check CatBoost GPU support
echo "🔍 Checking CatBoost GPU support..."
python -c "
try:
    from catboost import CatBoostRegressor
    import numpy as np
    X = np.random.rand(100, 10).astype('float32')
    y = np.random.rand(100).astype('float32')
    model = CatBoostRegressor(task_type='GPU', iterations=10, verbose=False)
    model.fit(X, y)
    print('✅ CatBoost GPU support confirmed')
except Exception as e:
    print(f'❌ CatBoost GPU test failed: {e}')
"

echo ""
echo "🎯 Starting production training..."
echo "   Universe: 60 symbols"
echo "   Models: XGBoost GPU + CatBoost GPU + Ridge CPU"
echo "   Target time: ~1 hour"
echo ""

# Run production training
python scripts/train_production.py \
    --universe-cfg config/universe_production.yaml \
    --xgb-cfg config/grids/cs_xgb_gpu_hour.yaml \
    --catboost-cfg config/grids/cs_cat_gpu_light.yaml \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ Production training complete!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "🔍 Next steps:"
echo "   1. Check results: ls -la $OUTPUT_DIR"
echo "   2. Build ensemble: python scripts/demo_ensemble.py --input-dir $OUTPUT_DIR"
echo "   3. Create portfolio: python scripts/portfolio_aggregate.py --input-dir $OUTPUT_DIR"
echo "   4. Deploy: python scripts/deploy_phase3.py --out-dir deployment/production"
echo ""
