# 🔧 Repo-Wide Refactor Report

**Date**: December 2024
**Status**: Analysis Complete - Ready for Implementation

## 📊 **File Size Analysis**

### **Top 30 Largest .py Files (LOC)**

| Rank | File | Lines | Status | Action Required |
|------|------|-------|--------|----------------|
| 1 | `tests/test_data_sanity_enforcement.py` | 1,773 | 🚨 **CRITICAL** | Split into multiple test modules |
| 2 | `core/data_sanity.py` | 1,395 | 🚨 **CRITICAL** | Split into core + validators |
| 3 | `tests/test_data_integrity.py` | 1,229 | 🚨 **CRITICAL** | Split into multiple test modules |
| 4 | `core/engine/backtest.py` | 972 | 🚨 **CRITICAL** | Split into core + helpers |
| 5 | `analysis_viz.py` | 889 | 🚨 **CRITICAL** | Split into multiple viz modules |
| 6 | `core/engine/paper.py` | 800 | 🚨 **CRITICAL** | Split into core + helpers |
| 7 | `attic/root_tests/test_core_functionality.py` | 769 | ⚠️ **LARGE** | Legacy - consider cleanup |
| 8 | `scripts/walkforward_framework.py` | 750 | 🚨 **CRITICAL** | Split into framework + utils |
| 9 | `core/ml/visualizer.py` | 676 | 🚨 **CRITICAL** | Split into multiple viz modules |
| 10 | `scripts/readiness_check.py` | 650 | 🚨 **CRITICAL** | Split into multiple check modules |
| 11 | `risk/overlay.py` | 643 | 🚨 **CRITICAL** | Split into core + metrics |
| 12 | `scripts/ml_walkforward.py` | 595 | 🚨 **CRITICAL** | Split into core + utils |
| 13 | `ml/train.py` | 595 | 🚨 **CRITICAL** | Split into core + helpers |
| 14 | `core/ml/profit_learner.py` | 591 | 🚨 **CRITICAL** | Split into core + helpers |
| 15 | `strategies/regime_aware_ensemble.py` | 586 | 🚨 **CRITICAL** | Split into core + components |
| 16 | `signals/condition.py` | 560 | 🚨 **CRITICAL** | Split into core + metrics |
| 17 | `features/feature_engine.py` | 537 | 🚨 **CRITICAL** | Split into core + helpers |
| 18 | `core/regime_detector.py` | 535 | 🚨 **CRITICAL** | Split into core + indicators |
| 19 | `scripts/generate_simple_signal_templates.py` | 531 | 🚨 **CRITICAL** | Split into core + generators |
| 20 | `core/feature_reweighter.py` | 504 | 🚨 **CRITICAL** | Split into core + helpers |
| 21 | `experiments/persistence.py` | 503 | 🚨 **CRITICAL** | Split into core + analysis |
| 22 | `attic/root_tests/test_data_integrity.py` | 485 | ⚠️ **LARGE** | Legacy - consider cleanup |
| 23 | `scripts/generate_signal_templates.py` | 474 | 🚨 **CRITICAL** | Split into core + generators |
| 24 | `attic/root_tests/final_ml_validation.py` | 473 | ⚠️ **LARGE** | Legacy - consider cleanup |
| 25 | `tests/test_v02_modules.py` | 470 | ⚠️ **LARGE** | Split into multiple test modules |
| 26 | `features/ensemble.py` | 465 | 🚨 **CRITICAL** | Split into core + components |
| 27 | `core/telemetry/snapshot.py` | 459 | 🚨 **CRITICAL** | Split into core + serializers |
| 28 | `attic/root_tests/test_meaningful_validation.py` | 455 | ⚠️ **LARGE** | Legacy - consider cleanup |
| 29 | `brokers/data_provider.py` | 450 | 🚨 **CRITICAL** | Split into core + providers |
| 30 | `core/portfolio.py` | 448 | 🚨 **CRITICAL** | Split into core + helpers |

### **Size Categories**
- **🚨 CRITICAL** (>500 LOC): 20 files
- **⚠️ LARGE** (400-500 LOC): 5 files
- **✅ ACCEPTABLE** (<400 LOC): 5 files

## 🎯 **Refactoring Priorities**

### **Phase 1: Critical Files (>500 LOC)**
1. **Core Engine Files** - `backtest.py`, `paper.py`, `data_sanity.py`
2. **ML Components** - `profit_learner.py`, `visualizer.py`, `train.py`
3. **Feature Engineering** - `feature_engine.py`, `regime_detector.py`
4. **Risk & Signals** - `overlay.py`, `condition.py`
5. **Scripts** - `walkforward_framework.py`, `readiness_check.py`

### **Phase 2: Large Files (400-500 LOC)**
1. **Test Files** - Split into focused test modules
2. **Legacy Files** - Clean up attic/root_tests
3. **Remaining Core** - `portfolio.py`, `data_provider.py`

## 🔧 **Shared Utility Modules to Create**

### **1. utils/indicators.py**
```python
# Technical indicators and calculations
- rolling_mean, rolling_std, rolling_median
- zscore, winsorize, normalize
- rsi, macd, atr, bollinger_bands
- pct_change, lag, lead, diff
- momentum, volatility calculations
```

### **2. utils/timeseries.py**
```python
# Time series operations
- is_monotonic, is_sorted
- align_dataframes, resample_data
- safe_forward_fill, safe_backward_fill
- split_train_test, split_walkforward
- date_range_helpers
```

### **3. utils/metrics.py**
```python
# Performance and risk metrics
- sharpe_ratio, sortino_ratio, calmar_ratio
- max_drawdown, var, cvar
- turnover, hit_rate, profit_factor
- return dataclasses for structured output
```

### **4. utils/logging.py**
```python
# Centralized logging
- get_logger(name) with file+console output
- single formatter for consistency
- log levels and configuration
```

### **5. utils/timing.py**
```python
# Performance timing
- @time_block("stage") decorator
- context manager for timing blocks
- performance tracking utilities
```

### **6. utils/config.py**
```python
# Configuration management
- YAML loader with dataclass schemas
- defaults and validation
- environment variable support
```

## 📋 **Refactoring Action Plan**

### **Step 1: Create Shared Utilities**
1. Create `utils/` directory
2. Implement all utility modules
3. Add comprehensive tests for utilities

### **Step 2: Split Critical Files**
For each file >500 LOC:
1. Identify distinct responsibilities
2. Create `*_core.py` (public API)
3. Create `*_impl.py` (private helpers)
4. Move shared code to utils/
5. Update imports and maintain compatibility

### **Step 3: Standardize Interfaces**
1. Create model adapters for consistency
2. Standardize DataFrame interfaces
3. Implement thin, import-friendly APIs
4. Add deprecation warnings for old APIs

### **Step 4: Clean Up Legacy**
1. Review attic/root_tests files
2. Remove or refactor legacy code
3. Update documentation

## 🎯 **Success Metrics**

### **Targets**
- **Module size**: ≤400 LOC (warn if >300)
- **Function size**: ≤50 LOC
- **Cyclomatic complexity**: ≤C (radon)
- **Duplicate code**: Zero warnings (pylint R0801)
- **Runtime performance**: Equal or faster on existing tests

### **Validation**
- All existing tests pass
- No behavior changes
- Performance maintained or improved
- Code coverage maintained

## 📊 **Expected Benefits**

### **Maintainability**
- Smaller, focused modules
- Reduced cognitive load
- Easier testing and debugging
- Better code reuse

### **Performance**
- Reduced memory usage
- Faster imports
- Better caching
- Optimized algorithms

### **Developer Experience**
- Faster IDE performance
- Better autocomplete
- Clearer module boundaries
- Easier onboarding

---

**Next Steps**: Begin with Phase 1 critical files, starting with the largest offenders and creating shared utilities to eliminate code duplication.
