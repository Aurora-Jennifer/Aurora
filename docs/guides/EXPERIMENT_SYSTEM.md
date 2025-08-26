# Aurora Professional Experiment System

## Overview

The Aurora experiment system provides **professional-grade configuration testing** with statistical rigor. This is your "config lab" for systematic signal discovery across different features, models, and market conditions.

## Key Features

### 🔬 **Statistical Rigor**
- **Timeline splits**: Discovery vs. confirmation holdout (prevents data snooping)
- **IC validation**: Information Coefficient with HAC standard errors & t-tests
- **Block bootstrap**: Confidence intervals that respect time series structure
- **Multiple testing controls**: Deflated Sharpe ratios and promotion gates

### ⚡ **Easy Operation**
- **One-command experiments**: `python scripts/experiment_runner.py momentum_discovery`
- **Pre-built profiles**: 6 ready-to-use experimental designs
- **Parallel execution**: Automatic multi-core processing
- **Full provenance**: Every result tracked with complete metadata

### 🎯 **Professional Quality**
- **Hypothesis-driven**: Each experiment tests a specific market hypothesis
- **Trial budgets**: Prevents infinite p-hacking
- **Promotion gates**: Only statistically significant results advance
- **Regime analysis**: Performance across different market conditions

---

## Quick Start

### 1. List Available Experiments
```bash
python scripts/experiment_runner.py list
```

### 2. Run a Quick Test
```bash
# Dry run to see what configs would be generated
python scripts/experiment_runner.py quick_validation --dry-run

# Run discovery phase only (safer for testing)
python scripts/experiment_runner.py quick_validation --discovery-only

# Full experiment (discovery + confirmation)
python scripts/experiment_runner.py quick_validation
```

### 3. Run a Larger Experiment
```bash
# Test momentum features across multiple assets
python scripts/experiment_runner.py momentum_discovery

# Cross-asset momentum (includes crypto)
python scripts/experiment_runner.py cross_asset_momentum

# Conservative validation (strict thresholds)
python scripts/experiment_runner.py conservative_validation
```

---

## Understanding the Process

### Phase 1: Discovery
- Tests many configurations on historical data
- Computes IC statistics with HAC standard errors
- Filters by statistical significance (t-stat > 2.0)
- Selects top candidates based on deflated Sharpe ratio

### Phase 2: Confirmation (One-Shot)
- Tests selected candidates on **holdout data**
- Cannot be repeated (prevents confirmation bias)
- Only promotes configs that maintain performance
- Generates final deployment recommendations

### Example Output
```
🔍 DISCOVERY PHASE
Timeline: 2020-01-01 to 2023-06-30
🔧 Generated 150 configurations
✅ [150/150] Complete | Mean IC: 0.023 | Best Sharpe: 1.2

🎯 CANDIDATE SELECTION
Discovery configs: 150
Passed filters: 12
Selected for confirmation: 5

🔒 CONFIRMATION PHASE
Timeline: 2023-07-01 to 2024-12-31
✅ Confirmation IC: 0.019 (t=2.4) | Sharpe: 0.8

🚀 PROMOTED: 2 configurations ready for paper trading
```

---

## Experiment Profiles

### Built-in Profiles

| Profile | Hypothesis | Assets | Budget | Thresholds |
|---------|------------|--------|--------|------------|
| `quick_validation` | Rapid development testing | SPY | 25 | Lenient (IC > 0.01) |
| `momentum_discovery` | Multi-timeframe momentum signals | SPY, QQQ, IWM | 150 | Standard (IC > 0.02) |
| `volatility_regimes` | Vol-based features across regimes | SPY, VIX | 100 | Standard (IC > 0.025) |
| `cross_asset_momentum` | Momentum across asset classes | Equities + Crypto | 200 | Moderate (IC > 0.03) |
| `high_frequency_signals` | Short-term technical patterns | SPY | 75 | Lenient (IC > 0.015) |
| `conservative_validation` | Institutional-grade validation | 5 ETFs | 300 | Strict (IC > 0.05) |

### Creating Custom Profiles

Add to `config/experiment_profiles.yaml`:

```yaml
my_custom_experiment:
  name: "my_custom_experiment"
  hypothesis: "Custom feature set provides alpha in volatile markets"
  feature_families: ["momentum_extended", "volatility_focus"]
  tickers: ["SPY", "QQQ"]
  discovery_start: "2020-01-01"
  discovery_end: "2023-06-30"
  confirmation_start: "2023-07-01"
  confirmation_end: "2024-12-31"
  trial_budget: 100
  cost_bps: 5.0
  min_ic_threshold: 0.025
  min_sharpe_threshold: 0.6
  random_seed: 42
  parallel_jobs: 4
```

---

## Advanced Usage

### Direct Config Sweep
```bash
# Run config sweep directly with custom profile
python scripts/config_sweep.py config/experiment_profiles.yaml momentum_discovery

# Discovery only
python scripts/config_sweep.py config/experiment_profiles.yaml momentum_discovery --discovery-only

# Generate configs without running
python scripts/config_sweep.py config/experiment_profiles.yaml momentum_discovery --dry-run
```

### IC Analysis Integration
The experiment system automatically integrates with the IC analysis framework:

```bash
# After running experiments, analyze specific results
python scripts/ic_analysis.py --experiment <exp_id>

# List all experiments with IC quality scores
python scripts/experiments.py list
```

---

## Interpreting Results

### IC Quality Classifications
- **Weak**: |IC| < 0.02 (may not be profitable after costs)
- **Moderate**: 0.02 ≤ |IC| < 0.05 (potentially profitable)
- **Strong**: 0.05 ≤ |IC| < 0.10 (likely profitable)
- **Exceptional**: |IC| ≥ 0.10 (investigate for data leakage)

### Statistical Significance
- **t-stat > 2.0**: Statistically significant at 95% confidence
- **Hit rate > 0.55**: Predictions directionally correct more than half the time
- **Confidence interval**: Bootstrap CI excludes zero

### Promotion Criteria
A configuration is promoted if it passes ALL of:
1. **Statistical significance**: t-stat > 2.0
2. **Economic significance**: IC > threshold
3. **Risk-adjusted performance**: Sharpe > threshold  
4. **Drawdown control**: Max drawdown < 30%
5. **Confirmation validation**: Maintains performance on holdout

---

## Integration with Trading System

### 1. Paper Trading Deployment
```bash
# After getting promoted configs, deploy to paper trading
python scripts/paper_runner.py --config <promoted_config_path>
```

### 2. Model Training Pipeline
```bash
# Use promoted features for production training
python scripts/train.py --features <promoted_features> --model <promoted_model>
```

### 3. Live Trading (After Paper Success)
```bash
# Only after 4-6 weeks of successful paper trading
python scripts/live_runner.py --config <validated_config>
```

---

## Best Practices

### Experimental Design
1. **Hypothesis first**: Always start with a clear market hypothesis
2. **Timeline discipline**: Never reuse confirmation data
3. **Budget limits**: Stick to trial budgets to prevent p-hacking
4. **Cost realism**: Include realistic transaction costs

### Feature Engineering
1. **Domain knowledge**: Use financial intuition to guide feature families
2. **Regime awareness**: Test across different market conditions  
3. **Stability**: Prefer features that work across multiple assets
4. **Leakage prevention**: Never use future information

### Statistical Validation
1. **Significance testing**: Always require t-stat > 2.0
2. **Block bootstrap**: Use for time series confidence intervals
3. **Multiple testing**: Apply deflated Sharpe or other corrections
4. **Regime analysis**: Check performance across vol/trend regimes

---

## Common Workflows

### Research & Development
```bash
# 1. Quick iteration on new features
python scripts/experiment_runner.py quick_validation --discovery-only

# 2. Deeper investigation of promising signals
python scripts/experiment_runner.py momentum_discovery

# 3. Cross-asset validation
python scripts/experiment_runner.py cross_asset_momentum
```

### Production Validation
```bash
# 1. Conservative thresholds for live deployment
python scripts/experiment_runner.py conservative_validation

# 2. Analyze promoted configs
python scripts/ic_analysis.py --experiment <promoted_exp_id>

# 3. Deploy to paper trading
python scripts/paper_runner.py --config <promoted_config>
```

### Troubleshooting
```bash
# No configs promoted? Check discovery results:
python scripts/experiments.py list

# Examine specific failed experiment:
python scripts/ic_analysis.py --experiment <exp_id>

# Lower thresholds for development:
# Edit config/experiment_profiles.yaml, reduce min_ic_threshold
```

---

## System Architecture

```
experiments/
├── config_sweep.py          # Core sweep runner
├── experiment_runner.py     # User-friendly CLI
├── ic_analysis.py          # Statistical validation
└── experiments.py          # Result management

core/stats/
├── ic_validator.py         # IC computation with HAC/bootstrap
└── __init__.py

config/
└── experiment_profiles.yaml # Pre-built experimental designs

reports/experiments/
├── <experiment_name>_<timestamp>/
│   ├── discovery_results.json
│   ├── confirmation_results.json
│   └── final_report.json
```

---

## Next Steps

After running experiments:

1. **Review promoted configurations** in experiment reports
2. **Deploy to paper trading** for live validation
3. **Monitor performance** for 4-6 weeks minimum
4. **Consider live deployment** only after paper success
5. **Iterate** with new hypotheses and features

The experiment system gives you the tools to **systematically discover profitable signals** while maintaining statistical rigor. Use it to build confidence in your strategies before risking real capital.
