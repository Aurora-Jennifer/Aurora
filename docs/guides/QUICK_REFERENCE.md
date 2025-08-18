# Quick Reference Guide

**Date**: 2025-08-16
**System Status**: ⚠️ PARTIALLY WORKING
**Paper Trading Ready**: ❌ NO

## 🚨 Immediate Issues (Fix First)

### 1. IBKR Broker Error
```bash
# Error: name 'IB' is not defined
# Fix: Add to brokers/ibkr_broker.py
from ib_insync import IB, Ticker, Contract
```

### 2. Logging Error
```bash
# Error: cannot import name 'get_logger'
# Fix: Add to core/enhanced_logging.py
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

### 3. Missing Environment Variables
```bash
# Create .env file:
MAX_POSITION_PCT=0.15
MAX_GROSS_LEVERAGE=2.0
DAILY_LOSS_CUT_PCT=0.03
STRUCTURED_LOGS=1
```

## ✅ Working Components

- Core backtesting engine
- Walkforward analysis (without DataSanity)
- Regime detection
- Strategy execution
- Basic performance metrics

## ❌ Broken Components

- IBKR broker integration
- Structured logging
- DataSanity validation in walkforward
- Risk management enforcement

## 🔧 Quick Fix Commands

### Fix IBKR Broker
```bash
# Add import to brokers/ibkr_broker.py
echo "from ib_insync import IB, Ticker, Contract" >> brokers/ibkr_broker.py
```

### Fix Logging
```bash
# Add function to core/enhanced_logging.py
echo "def get_logger(name: str) -> logging.Logger: return logging.getLogger(name)" >> core/enhanced_logging.py
```

### Set Environment Variables
```bash
export MAX_POSITION_PCT=0.15
export MAX_GROSS_LEVERAGE=2.0
export DAILY_LOSS_CUT_PCT=0.03
export STRUCTURED_LOGS=1
```

## 🧪 Test Commands

### Test Core Functionality
```bash
python scripts/preflight.py
```

### Test Go/No-Go Gate
```bash
STRUCTURED_LOGS=1 RUN_ID=$(date +%Y%m%d-%H%M%S) MAX_POSITION_PCT=0.15 python scripts/go_nogo.py
```

### Test Backtesting
```bash
python cli/backtest.py --start 2024-01-01 --end 2024-01-31 --symbols SPY --fast
```

### Test Walkforward
```bash
python scripts/walkforward_framework.py --symbol SPY --train-len 60 --test-len 20
```

### Test All Imports
```bash
python -c "
from core.engine.paper import PaperTradingEngine
from core.engine.backtest import BacktestEngine
from strategies.regime_aware_ensemble import RegimeAwareEnsembleStrategy
from core.regime_detector import RegimeDetector
print('✅ Core imports work')
"
```

## 📊 Current Performance

### Backtest Results (1 week)
- Total Return: -0.00%
- Sharpe Ratio: -0.96
- Max Drawdown: -0.03%
- Trades: 1

### Walkforward Results (36 folds)
- Mean Sharpe: 0.212
- Out-of-sample Sharpe: 0.31
- Win Rate: 29.7% (in-sample), 37% (out-of-sample)

## 📁 Key Files

### Configuration
- `config/enhanced_paper_trading_config.json` - Main config
- `config/data_sanity.yaml` - DataSanity settings
- `config/ibkr_config.json` - IBKR settings

### Core Components
- `core/engine/paper.py` - Paper trading engine
- `core/engine/backtest.py` - Backtesting engine
- `core/regime_detector.py` - Regime detection
- `strategies/regime_aware_ensemble.py` - Main strategy

### Scripts
- `scripts/preflight.py` - System validation
- `scripts/go_nogo.py` - Production readiness check
- `scripts/walkforward_framework.py` - Walkforward analysis

### Tests
- `tests/` - Test suite (45 failed out of 448)
- `scripts/falsification_tests.py` - Data validation tests

## 🚀 Next Steps

### Phase 1: Critical Fixes (1-2 hours)
1. Fix IBKR broker import error
2. Fix logging system
3. Configure environment variables
4. Test core functionality

### Phase 2: DataSanity Integration (1-2 hours)
1. Fix timezone validation issues
2. Align performance metrics
3. Test walkforward with DataSanity

### Phase 3: Production Readiness (1-2 hours)
1. Set up IBKR Gateway
2. Configure monitoring
3. Run comprehensive tests

## 📋 Status Checklist

- [ ] IBKR broker working
- [ ] Logging system working
- [ ] Environment variables set
- [ ] DataSanity integration working
- [ ] Performance metrics complete
- [ ] Test suite passing
- [ ] Risk management configured
- [ ] IBKR Gateway set up

## 🔍 Error Codes

| Error | Status | Fix |
|-------|--------|-----|
| `name 'IB' is not defined` | ❌ | Add import |
| `cannot import name 'get_logger'` | ❌ | Add function |
| `MAX_POSITION_PCT not set` | ❌ | Set env var |
| `Naive timezone not allowed` | ⚠️ | Add timezone |
| `Insufficient data` | ⚠️ | Reduce requirements |

## 📞 Emergency Contacts

- **System Issues**: Check `SYSTEM_READINESS_REPORT.md`
- **Error Details**: Check `ERROR_ANALYSIS_REPORT.md`
- **Component Status**: Check `COMPONENT_ANALYSIS_REPORT.md`
- **Refactoring Guide**: Check `REFACTORING_GUIDE.md`

## 🎯 Success Criteria

System is ready for paper trading when:
- [ ] All critical errors resolved
- [ ] Preflight test passes
- [ ] Go/No-Go gate shows "GO ✅"
- [ ] Walkforward with DataSanity works
- [ ] Test suite >90% passing
- [ ] Risk management configured
- [ ] IBKR Gateway connected

**Estimated Time to Ready**: 4-6 hours
