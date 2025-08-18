# Alpha v1 Pipeline Runbook

## Overview

The Alpha v1 pipeline provides a complete ML workflow for training and evaluating trading models with strict leakage guards and deterministic results.

## Quick Start

### 1. Train Alpha v1 Model
```bash
# Train on SPY, TSLA with 5-fold walkforward validation
python tools/train_alpha_v1.py --symbols SPY,TSLA --n-folds 5

# Train on larger universe
python tools/train_alpha_v1.py --symbols SPY,TSLA,AAPL,MSFT,GOOGL --n-folds 5
```

### 2. Validate Results
```bash
# Check if model meets promotion gates
python tools/validate_alpha.py reports/alpha_eval.json
```

### 3. Promote to Paper Trading
```bash
# If validation passes, bless the model
python tools/bless_model_inference.py

# Run smoke test
make smoke

# Update config to use linear_v1
# Edit config/base.yaml: models.selected = "linear_v1"
```

## Promotion Gates

Your model must meet these criteria to be promoted to paper trading:

- **IC (Spearman) ≥ 0.02** - Information coefficient (correlation between predictions and actual returns)
- **Hit Rate ≥ 0.52** - Directional accuracy (percentage of correct directional predictions)
- **Turnover ≤ 2.0** - Portfolio turnover (not excessive trading)
- **Total Predictions ≥ 100** - Sufficient sample size for evaluation

## Feature Engineering

### Current Features
- **Momentum**: 1d, 5d, 20d returns, SMA ratio (20d/50d - 1)
- **Volatility**: 10d, 20d rolling volatility
- **Oscillator**: 14-day RSI
- **Liquidity**: 20d z-scored volume

### Leakage Guards
- **Label Shift**: Target (`ret_fwd_1d`) is shifted forward by 1 day
- **Time-based Split**: Train/test split respects temporal ordering
- **Walkforward Validation**: No overlapping test periods

## Model Architecture

### Ridge Regression Pipeline
```python
Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
])
```

### Training Process
1. **Feature Building**: Download data, calculate features, apply leakage guards
2. **Time-based Split**: 80% train, 20% test (no overlap)
3. **Cross-validation**: TimeSeriesSplit for hyperparameter tuning
4. **Model Persistence**: Save as pickle file with metadata

## Evaluation Metrics

### Information Coefficient (IC)
- **Definition**: Spearman correlation between predictions and actual returns
- **Interpretation**: Higher IC = better predictive power
- **Threshold**: ≥ 0.02 (small but real alpha)

### Hit Rate
- **Definition**: Percentage of correct directional predictions
- **Calculation**: `sign(prediction) == sign(actual_return)`
- **Threshold**: ≥ 0.52 (better than random)

### Turnover
- **Definition**: Average position changes per period
- **Calculation**: `mean(abs(diff(positions)))`
- **Threshold**: ≤ 2.0 (not excessive trading)

### Return with Costs
- **Definition**: Net return after slippage and fees
- **Costs Applied**: 
  - Slippage: 5 bps per trade
  - Fees: 1 bp per trade
- **Interpretation**: Realistic performance estimate

## Walkforward Validation

### Process
1. **Expanding Window**: Train on increasing historical data
2. **Fixed Test Periods**: Test on non-overlapping future periods
3. **Multiple Folds**: 5-fold validation for robust estimates
4. **Out-of-Sample**: All test periods are strictly future

### Example Timeline
```
Fold 1: Train [2020-01-01, 2021-12-31] → Test [2022-01-01, 2022-06-30]
Fold 2: Train [2020-01-01, 2022-06-30] → Test [2022-07-01, 2022-12-31]
Fold 3: Train [2020-01-01, 2022-12-31] → Test [2023-01-01, 2023-06-30]
...
```

## Troubleshooting

### Common Issues

#### 1. Insufficient Data
```
ValueError: Insufficient data: 100 < 504
```
**Solution**: Use more symbols or longer date range
```bash
python tools/train_alpha_v1.py --symbols SPY,TSLA,AAPL,MSFT,GOOGL
```

#### 2. Low IC Score
```
IC 0.015 < 0.02 threshold
```
**Solutions**:
- Add more features (technical indicators, fundamental data)
- Try different hyperparameters
- Expand universe to more symbols
- Use longer training period

#### 3. Low Hit Rate
```
Hit rate 0.51 < 0.52 threshold
```
**Solutions**:
- Check for data quality issues
- Verify leakage guards are working
- Try different model architectures
- Adjust feature engineering

#### 4. High Turnover
```
Turnover 2.5 > 2.0 threshold
```
**Solutions**:
- Add position smoothing
- Use longer-term features
- Implement position limits
- Add transaction cost penalties

### Data Quality Checks

#### Verify Leakage Guards
```bash
# Run leakage tests
python -m pytest tests/ml/test_leakage_guards.py -v
```

#### Check Feature Distribution
```python
import pandas as pd
df = pd.read_parquet("artifacts/feature_store/SPY.parquet")
print(df.describe())
print(df.isnull().sum())
```

#### Validate Schema
```bash
# Validate evaluation results
python tools/validate_alpha.py reports/alpha_eval.json
```

## Iteration Process

### 1. Feature Engineering
- Add new technical indicators
- Include fundamental data
- Try different lookback periods
- Add cross-sectional features

### 2. Model Architecture
- Try different algorithms (LightGBM, XGBoost)
- Experiment with ensemble methods
- Add regularization techniques
- Implement feature selection

### 3. Hyperparameter Tuning
- Grid search over alpha values
- Try different CV strategies
- Optimize for different metrics
- Use Bayesian optimization

### 4. Universe Expansion
- Add more liquid symbols
- Include different asset classes
- Consider international markets
- Add sector-specific models

## Production Deployment

### Pre-deployment Checklist
- [ ] Model passes all promotion gates
- [ ] Leakage tests pass
- [ ] Schema validation passes
- [ ] Smoke test passes
- [ ] Model blessed for inference

### Monitoring
- Track IC decay over time
- Monitor hit rate stability
- Watch for turnover spikes
- Check for feature drift

### Rollback Plan
- Keep previous model version
- Monitor performance closely
- Have fallback to dummy model
- Document any issues

## Configuration

### Feature Configuration (`config/features.yaml`)
```yaml
features:
  ret_1d: "1-day return"
  ret_5d: "5-day return"
  # ... more features

params:
  min_history_bars: 60
  lookback_days: 50
  label_shift: 1
```

### Model Configuration (`config/models.yaml`)
```yaml
linear_v1:
  kind: "pickle"
  path: "artifacts/models/linear_v1.pkl"
  metadata:
    feature_order: ["ret_1d", "ret_5d", ...]
    zscore: true
    psi_threshold: 0.1
```

## Commands Reference

### Training
```bash
# Basic training
python tools/train_alpha_v1.py --symbols SPY,TSLA

# Custom parameters
python tools/train_alpha_v1.py \
  --symbols SPY,TSLA,AAPL,MSFT \
  --n-folds 5 \
  --random-state 42
```

### Validation
```bash
# Validate results
python tools/validate_alpha.py reports/alpha_eval.json

# Custom thresholds
python tools/validate_alpha.py \
  reports/alpha_eval.json \
  --ic-threshold 0.025 \
  --hit-rate-threshold 0.53
```

### Testing
```bash
# Run all tests
python -m pytest tests/ml/ -v

# Run specific test
python -m pytest tests/ml/test_leakage_guards.py -v
```

## Performance Benchmarks

### Expected Performance
- **IC**: 0.02 - 0.05 (good alpha)
- **Hit Rate**: 0.52 - 0.55 (better than random)
- **Turnover**: 0.1 - 0.5 (reasonable trading)
- **Training Time**: < 5 minutes for 5 symbols
- **Evaluation Time**: < 2 minutes for 5-fold validation

### Scaling Considerations
- **Memory**: ~1GB for 10 symbols, 3 years of data
- **Storage**: ~100MB for feature store
- **CPU**: Single-threaded, can parallelize by symbol
- **Network**: yfinance API calls for data download

## Next Steps

### Phase 2 Enhancements
1. **Cross-sectional Models**: Rank-based predictions
2. **Ensemble Methods**: Combine multiple models
3. **Feature Engineering**: Add more sophisticated indicators
4. **Risk Management**: Position sizing and portfolio construction

### Advanced Features
1. **Real-time Features**: Streaming data integration
2. **Market Regime**: Regime-aware models
3. **Multi-timeframe**: Combine daily and intraday signals
4. **Alternative Data**: News, sentiment, options flow
