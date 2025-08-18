# Multi-Asset ML Trading Analysis Guide

## 🎯 Overview

This guide provides step-by-step instructions for running comprehensive 5-year walkforward analysis on multiple assets including stocks and crypto using the ML trading system.

## 📊 Available Assets

The system supports analysis of the following assets:

| Asset | Symbol | Description | Type |
|-------|--------|-------------|------|
| SPY | SPY | S&P 500 ETF | Stock ETF |
| AAPL | AAPL | Apple Inc. | Stock |
| TSLA | TSLA | Tesla Inc. | Stock |
| GOOG | GOOG | Alphabet Inc. (Google) | Stock |
| BTC-USD | BTC-USD | Bitcoin | Cryptocurrency |

## 🚀 Quick Start Commands

### 1. Single Asset Analysis
```bash
# Run 5-year walkforward for SPY only
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy
```

### 2. Multiple Assets Analysis
```bash
# Run 5-year walkforward for stocks only
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl tsla goog

# Run 5-year walkforward for all assets including crypto
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl tsla goog btc
```

### 3. Custom Date Ranges
```bash
# Shorter period for testing
python scripts/run_multi_asset_walkforward.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl

# Longer period for comprehensive analysis
python scripts/run_multi_asset_walkforward.py \
  --start-date 2018-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl tsla goog btc
```

## 📈 Understanding the Analysis

### What Each Analysis Does

1. **Walkforward Testing**: Creates multiple train/test folds
   - Training period: 252 days (1 year)
   - Testing period: 63 days (1 quarter)
   - Step size: 63 days (moves forward each quarter)

2. **ML Learning**: In each training fold
   - Trains ML model on historical data
   - Records feature importance
   - Tracks learning progress

3. **Out-of-Sample Testing**: In each testing fold
   - Tests model on unseen data
   - Calculates performance metrics
   - Validates model robustness

4. **Warm-Start**: Between folds
   - Uses knowledge from previous folds
   - Improves convergence speed
   - Maintains learning continuity

### Expected Results

For a 5-year period (2019-2024), you should expect:
- **~20 folds** per asset (5 years × 4 quarters)
- **~100 total test periods** across all assets
- **Comprehensive performance metrics** for each asset
- **Feature importance analysis** across time periods
- **Cross-asset comparison** and ranking

## 📊 Results Structure

### Individual Asset Results
Each asset gets its own directory:
```
results/
├── ml_walkforward_spy/
│   ├── ml_walkforward_summary.md
│   ├── ml_walkforward_results.json
│   └── learning_progress.csv
├── ml_walkforward_aapl/
│   ├── ml_walkforward_summary.md
│   ├── ml_walkforward_results.json
│   └── learning_progress.csv
└── ...
```

### Comparison Report
```
results/
└── multi_asset_comparison.md
```

## 🔍 Analyzing Results

### 1. Check Individual Asset Performance
```bash
# View SPY results
cat results/ml_walkforward_spy/ml_walkforward_summary.md

# View AAPL results
cat results/ml_walkforward_aapl/ml_walkforward_summary.md

# View detailed JSON results
cat results/ml_walkforward_spy/ml_walkforward_results.json | jq '.overall_metrics'
```

### 2. Compare All Assets
```bash
# View comprehensive comparison
cat results/multi_asset_comparison.md
```

### 3. Check Learning Progress
```bash
# View learning progress for each asset
head -10 results/ml_walkforward_spy/learning_progress.csv
head -10 results/ml_walkforward_aapl/learning_progress.csv
```

## ⏱️ Time Estimates

### Processing Time by Asset Count
- **1 asset**: ~5-10 minutes
- **2-3 assets**: ~15-30 minutes
- **All 5 assets**: ~45-90 minutes

### Factors Affecting Speed
- **Date range**: Longer periods = more folds = more time
- **Asset volatility**: Crypto (BTC) may take longer due to data complexity
- **System resources**: CPU/memory availability
- **Network**: Data download speed for historical prices

## 🎯 Recommended Workflows

### Workflow 1: Quick Test (Recommended for First Run)
```bash
# Test with 2 assets, 2-year period
python scripts/run_multi_asset_walkforward.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl
```

### Workflow 2: Comprehensive Analysis
```bash
# Full 5-year analysis on all assets
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl tsla goog btc
```

### Workflow 3: Focus on Top Performers
```bash
# After identifying best assets, run detailed analysis
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl  # Focus on best performers
```

## 📊 Key Metrics to Monitor

### Performance Metrics
- **Average Test Return**: Expected return on out-of-sample data
- **Average Test Sharpe**: Risk-adjusted return
- **Win Rate**: Percentage of profitable test periods
- **Total Folds**: Number of train/test cycles completed

### Learning Metrics
- **Feature Importance Stability**: How consistent features are across time
- **Alpha Generation**: Features that consistently predict profits
- **Model Convergence**: How quickly the ML model learns

### Risk Metrics
- **Return Standard Deviation**: Volatility of returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Positive Folds**: Number of profitable test periods

## 🔧 Troubleshooting

### Common Issues

1. **"No folds generated"**
   - **Cause**: Date range too short
   - **Solution**: Use longer date range (at least 2 years)

2. **"Insufficient data" warnings**
   - **Cause**: Normal during warmup period
   - **Solution**: Ignore - system needs 252+ days for regime detection

3. **"RuntimeWarning: invalid value encountered"**
   - **Cause**: Normal numpy warnings during calculations
   - **Solution**: Ignore - doesn't affect results

4. **Slow performance**
   - **Cause**: Large date range or many assets
   - **Solution**: Run fewer assets or shorter period first

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/run_multi_asset_walkforward.py \
  --start-date 2023-01-01 \
  --end-date 2023-06-30 \
  --assets spy
```

## 📈 Next Steps After Analysis

### 1. Identify Best Performers
- Sort assets by Sharpe ratio
- Focus on consistent performers
- Note which assets work best in different market conditions

### 2. Feature Analysis
- Compare feature importance across assets
- Identify stable alpha factors
- Look for asset-specific patterns

### 3. Portfolio Construction
- Consider combining multiple assets
- Implement asset-specific risk controls
- Optimize position sizing per asset

### 4. Parameter Optimization
- Fine-tune parameters for each asset
- Test different fold lengths and step sizes
- Optimize ML learning parameters

## 🎯 Example Commands

### Quick Start (Recommended)
```bash
# Start with SPY only to test the system
python scripts/run_multi_asset_walkforward.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --assets spy

# Check results
cat results/multi_asset_comparison.md
```

### Full Analysis
```bash
# Run complete 5-year analysis on all assets
python scripts/run_multi_asset_walkforward.py \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --assets spy aapl tsla goog btc

# Monitor progress (check if files are being created)
ls -la results/ml_walkforward_*/

# View results when complete
cat results/multi_asset_comparison.md
```

### Custom Analysis
```bash
# Focus on tech stocks
python scripts/run_multi_asset_walkforward.py \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --assets aapl tsla goog

# Include crypto for diversification
python scripts/run_multi_asset_walkforward.py \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  --assets spy btc
```

## 📞 Support

- **Check logs**: Look for error messages in terminal output
- **Verify data**: Ensure historical data is available for selected assets
- **Monitor progress**: Check if result files are being created
- **Review results**: Use the generated reports to understand performance

The system is designed to be robust and provide comprehensive analysis across multiple assets and time periods. Start with a small test and gradually expand to full analysis as you become comfortable with the process.
