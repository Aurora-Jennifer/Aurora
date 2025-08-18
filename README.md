# 🚀 Advanced Trading System with Composer

![Smoke](https://github.com/<you>/<repo>/actions/workflows/smoke.yml/badge.svg)

### Smoke Run (CI)
- Runs on push/PR (fast deterministic check)
- Artifacts: `docs/analysis/walkforward_smoke_*.md`, `reports/smoke_run.json`
- Fails on: 0 folds, no trades, NaN/inf metrics, runtime >60s, DataSanity (CI enforce)

A config-driven, asset-class-aware trading system with a two-level composer for strategy selection and performance optimization.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Current Status](#current-status)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

## 🎯 System Overview

This trading system implements a **two-level composer architecture**:

### **Level 1: Strategy Selection**
- **Asset Class Detection**: Automatic crypto/ETF/equity classification
- **Market Regime Detection**: Trend, chop, volatile market identification
- **Strategy Weighting**: Dynamic strategy selection based on market conditions
- **Softmax Blending**: Intelligent combination of multiple strategies

### **Level 2: Performance Weight Optimization**
- **Composite Scoring**: Multi-metric evaluation (CAGR, Sharpe, Win Rate, Avg Return)
- **Automated Optimization**: Walkforward-based parameter tuning
- **Asset-Specific Tuning**: Different optimizations per asset class

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Config System │    │  Composer Core  │    │  Optimization   │
│                 │    │                 │    │                 │
│ • Base Config   │───▶│ • Strategy      │───▶│ • Weight Tuner  │
│ • Profile       │    │ • Regime        │    │ • Composite     │
│ • Asset-Specific│    │ • Composer      │    │ • Walkforward   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Entry Points   │    │  Integration    │    │  Validation     │
│                 │    │                 │    │                 │
│ • CLI Backtest  │    │ • Engine Hook   │    │ • Data Sanity   │
│ • Walkforward   │    │ • Composer      │    │ • Risk Guards   │
│ • Training      │    │ • Asset Class   │    │ • Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Basic Backtest**
```bash
# Conservative approach
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --profile risk_low --symbols SPY --fast

# Balanced approach
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --profile risk_balanced --symbols SPY TSLA

# Aggressive approach
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --profile risk_strict --symbols BTC-USD --initial-capital 50000
```

### **Asset-Specific Configuration**
```bash
# Crypto with higher volatility tolerance
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --asset BTC-USD --profile risk_balanced

# ETF with conservative settings
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --asset SPY --profile risk_low

# Equity with custom capital
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --asset TSLA --initial-capital 200000
```

### **Walkforward Analysis**
```bash
# Standard walkforward
python scripts/walkforward_framework.py --symbol SPY --profile risk_balanced --start-date 2020-01-01 --end-date 2023-12-31

# Custom parameters
python scripts/walkforward_framework.py --symbol BTC-USD --train-len 180 --test-len 45 --stride 15
```

### **Composer System Training**
```bash
# Train on multiple assets
python scripts/train_composer_system.py --symbols SPY TSLA BTC-USD --start-date 2020-01-01 --end-date 2023-12-31 --max-trials 10

# Train with specific profile
python scripts/train_composer_system.py --symbols SPY --profile risk_balanced --max-trials 5
```

## ⚙️ Configuration Guide

### **Configuration Hierarchy**
```
Base Config (config/base.json)
    ↓
Profile Override (risk_low/balanced/strict)
    ↓
Asset Override (crypto/etf/equity specific)
    ↓
CLI Override (--symbols, --initial-capital, etc.)
```

### **Base Configuration (`config/base.json`)**
```json
{
  "symbols": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN", "BTC-USD", "ETH-USD"],
  "risk_params": {
    "max_gross_exposure_pct": 50.0,
    "max_daily_loss_pct": 2.0,
    "max_position_size": 0.1,
    "max_drawdown_pct": 15.0
  },
  "composer": {
    "use_composer": false,
    "composer_mode": "softmax_blender",
    "eligible_strategies": ["momentum", "mean_reversion", "breakout"]
  }
}
```

### **Risk Profiles**
- **`risk_low`**: Conservative (25% exposure, 8% max DD)
- **`risk_balanced`**: Moderate (50% exposure, 15% max DD)
- **`risk_strict`**: Aggressive (75% exposure, 25% max DD)

### **Asset Classes**
- **Crypto**: Higher volatility, all strategies enabled
- **ETF**: Lower volatility, momentum + mean_reversion
- **Equity**: Moderate volatility, momentum + breakout

## 📊 Usage Examples

### **Example 1: Multi-Asset Backtest**
```bash
python cli/backtest.py \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --symbols SPY TSLA BTC-USD \
  --profile risk_balanced \
  --initial-capital 100000
```

### **Example 2: Custom Configuration**
```bash
# Create custom config
python -c "
from core.config_loader import create_profile_config
create_profile_config('my_profile', {
    'risk_params': {'max_drawdown_pct': 10.0},
    'composer': {'use_composer': True}
})
"

# Use custom config
python cli/backtest.py \
  --config config/my_profile.json \
  --start 2024-01-01 \
  --end 2024-01-10 \
  --symbols SPY
```

### **Example 3: Training Optimization**
```bash
# Train composer system
python scripts/train_composer_system.py \
  --symbols SPY TSLA \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --profile risk_balanced \
  --max-trials 20

# Use optimized config
python cli/backtest.py \
  --config config/optimized_composer.json \
  --start 2024-01-01 \
  --end 2024-01-10 \
  --symbols SPY
```

## 🔄 How It Works

### **Two-Level Composer System**

#### **Level 1: Strategy Selection**
```
Market Data → Asset Class Detection → Regime Detection → Strategy Selection → Weighted Blending
```

1. **Asset Class Detection**:
   - `BTC-USD` → `crypto` (higher volatility, all strategies)
   - `SPY` → `etf` (lower volatility, momentum + mean_reversion)
   - `TSLA` → `equity` (moderate volatility, momentum + breakout)

2. **Regime Detection**:
   - **Trend**: Linear regression slope > threshold
   - **Chop**: ADX-like choppiness > threshold
   - **Volatile**: High volatility, unclear direction

3. **Strategy Weighting**:
   - **Trending markets**: Favor momentum strategies
   - **Choppy markets**: Favor mean reversion strategies
   - **Volatile markets**: Equal weighting

#### **Level 2: Performance Weight Optimization**
```
Walkforward Results → Composite Scoring → Weight Optimization → Best Configuration
```

1. **Composite Scoring**:
   ```python
   score = (α × CAGR + β × Sharpe + γ × WinRate + δ × AvgReturn) - penalties
   ```

2. **Weight Optimization**:
   - Test different weight combinations
   - Evaluate via walkforward analysis
   - Select weights that maximize composite score

### **Data Flow**
```
1. Load Configuration → Merge Overrides → Validate
2. Initialize Composer System → Register Strategies → Build Components
3. Load Market Data → Extract Features → Detect Regime
4. Generate Strategy Predictions → Apply Composer Weights → Final Signal
5. Execute Trades → Track Performance → Update Metrics
6. Evaluate Results → Optimize Weights → Save Best Config
```

## 📈 Current Status

### ✅ **Completed Features**
- [x] Comprehensive configuration system
- [x] Two-level composer architecture
- [x] Asset-class-aware logic
- [x] Risk profile system
- [x] Walkforward integration
- [x] Training infrastructure
- [x] CLI refactoring

### ⚠️ **Critical Issues (Preventing Paper Trading)**
- [ ] DataSanityValidator method name conflicts
- [ ] Numba compilation errors in simulation
- [ ] Composer integration issues in walkforward
- [ ] Missing production validation features

### 🔄 **In Progress**
- [ ] Phase 1: Critical fixes
- [ ] Data validation standardization
- [ ] Performance optimization

## 🎯 Next Steps

### **Phase 1: Fix Critical Issues (Week 1)**
1. **Fix DataSanityValidator**
   - Standardize interface across all validation points
   - Add validation at data loading, composer decisions, trade execution

2. **Fix Numba Compilation**
   - Fix variable scoping in simulation functions
   - Add fallback to pure Python
   - Test all numba functions

3. **Fix Composer Integration**
   - Fix config propagation
   - Add comprehensive error handling
   - Test composer in walkforward

### **Phase 2: Add Production Features (Week 2)**
1. **Position Sizing & Risk Management**
   - Real-time position size validation
   - Risk limit enforcement
   - Dynamic position sizing based on volatility

2. **Transaction Cost Modeling**
   - Commission modeling
   - Slippage estimation
   - Market impact modeling

3. **Real-time Monitoring**
   - Performance monitoring
   - Risk monitoring
   - Alert system for violations

### **Phase 3: Validation & Testing (Week 3)**
1. **Comprehensive Testing**
   - Unit tests for all new components
   - Integration tests for composer system
   - End-to-end testing with real data

2. **Paper Trading Validation**
   - Test with IBKR paper trading
   - Validate all risk controls
   - Monitor performance and stability

### **Phase 4: Production Deployment (Week 4)**
1. **Production Configuration**
   - Production risk profiles
   - Production composer settings
   - Production monitoring setup

2. **Documentation & Training**
   - User manual
   - Troubleshooting guide
   - Performance tuning guide

## 🔧 Troubleshooting

### **Common Issues**

#### **1. Composer Not Working**
```bash
# Check if composer is enabled
python scripts/composer_flow_demo.py

# Verify config propagation
python core/config_loader.py
```

#### **2. DataSanity Errors**
```bash
# Skip validation temporarily
python scripts/walkforward_framework.py --validate-data false
```

#### **3. Numba Compilation Errors**
```bash
# Use pure Python fallback
export NUMBA_DISABLE_JIT=1
python scripts/train_composer_system.py
```

#### **4. Configuration Issues**
```bash
# Validate configuration
python -c "
from core.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_config(profile='risk_balanced')
print('Config valid:', loader.validate_config(config))
"
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python cli/backtest.py --start 2024-01-01 --end 2024-01-10 --symbols SPY
```

### **Performance Profiling**
```bash
# Profile composer system
python -m cProfile -o profile.stats scripts/train_composer_system.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 📚 Additional Resources

### **Configuration Files**
- `config/base.json` - Base configuration
- `config/risk_low.json` - Conservative risk profile
- `config/risk_balanced.json` - Balanced risk profile
- `config/risk_strict.json` - Aggressive risk profile

### **Key Scripts**
- `cli/backtest.py` - Main backtest interface
- `scripts/walkforward_framework.py` - Walkforward analysis
- `scripts/train_composer_system.py` - Composer training
- `scripts/composer_flow_demo.py` - Composer demonstration

### **Core Modules**
- `core/config_loader.py` - Configuration management
- `core/composer/` - Composer system
- `core/metrics/` - Performance metrics
- `core/engine/` - Trading engine integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

Aurora Analytics — Proprietary License. See the LICENSE and NOTICE files for details.

---

**⚠️ Important**: This system is currently in development and not ready for live trading. Complete Phase 1 fixes before using in production.
