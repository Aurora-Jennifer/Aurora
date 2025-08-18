# 🚀 Trading System Overview - Context for AI Prompting

## **System Identity**
- **Name**: Advanced Trading System with Composer
- **Type**: Config-driven, asset-class-aware algorithmic trading system
- **Architecture**: Two-level composer for strategy selection and optimization
- **Status**: Production-ready with ongoing improvements
- **Language**: Python 3.11+
- **Key Features**: ML learning, regime detection, walkforward analysis, data validation

## **Core Architecture**

### **Two-Level Composer System**
```
Level 1: Strategy Selection
├── Asset Class Detection (crypto/ETF/equity)
├── Market Regime Detection (trend/chop/volatile)
├── Strategy Weighting (dynamic selection)
└── Softmax Blending (intelligent combination)

Level 2: Performance Optimization
├── Composite Scoring (CAGR, Sharpe, Win Rate, Avg Return)
├── Automated Optimization (walkforward-based tuning)
└── Asset-Specific Tuning (per asset class)
```

### **Key Components**
- **Core Engine**: Backtest and paper trading engines
- **Strategy Selector**: ML-based strategy selection with contextual bandit
- **Regime Detector**: Market regime identification (trend, chop, volatile)
- **Portfolio Manager**: Position and risk management
- **Risk System**: Multi-layer risk controls and guardrails
- **ML System**: Machine learning with continual learning (19,088+ trades processed)
- **Walkforward Framework**: Time-based cross-validation (20-32x performance improvement)
- **DataSanity**: Data integrity and lookahead contamination prevention

## **Current System Status**

### **✅ Working Components**
- Core backtesting engine
- Walkforward analysis (without DataSanity)
- Regime detection
- Strategy execution
- ML learning system
- Technical indicators (28 enhanced indicators)
- Basic performance metrics

### **⚠️ Partially Working Components**
- Paper trading engine (needs IBKR configuration)
- Preflight validation
- Go/No-Go gate
- Performance reporting

### **❌ Critical Issues (Must Fix)**
- Memory leak in composer integration
- Configuration file loading failures
- Timezone handling edge cases
- Non-deterministic test behavior
- Error message inconsistencies

## **Performance Metrics**
- **Test Success Rate**: 94% (245/261 tests passing)
- **Walkforward Speed**: 20-32x improvement
- **ML Trades Processed**: 19,088+
- **Code Quality**: 200+ issues resolved

## **Configuration System**
- **Base Config**: `config/base.yaml` with core settings
- **Risk Profiles**: `risk_low.yaml`, `risk_balanced.yaml`, `risk_strict.yaml`
- **Asset-Specific**: Different configurations per asset class
- **Deep Merging**: Hierarchical configuration loading

## **Key Technologies**
- **Data Source**: yfinance with explicit auto_adjust control
- **ML Framework**: Custom contextual bandit with Thompson sampling
- **Validation**: DataSanity for data integrity checks
- **Risk Management**: Multi-layer with position sizing, drawdown limits, daily loss limits
- **Performance**: Optimized walkforward framework with caching

## **Development Philosophy**
- **No Hardcoded Values**: All runtime knobs come from config
- **Safety First**: Never index empty arrays, always check lengths
- **Warmup Discipline**: Enforce min_history_bars before decisions
- **One Log Per Cause**: No per-bar error spam, fold-level summaries
- **Idempotent Changes**: Minimal, surgical diffs only

## **Current Development Focus**
- **Critical Error Fixes**: Memory leaks, config loading, timezone handling
- **Production Readiness**: Full system integration testing
- **Performance Optimization**: Large dataset handling
- **Documentation**: API docs and deployment guides

## **Next Session Priorities**
1. Fix memory leak in composer integration
2. Validate all configuration file loading
3. Test timezone handling with various formats
4. Ensure test determinism
5. Verify error message consistency

## **Success Criteria**
- Memory usage <2GB for large datasets
- Execution time <30min for 5-year backtest
- 95%+ test success rate
- All configuration combinations load successfully
- No critical error logs in production simulation
