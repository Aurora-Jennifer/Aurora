# Cleanup Session Context

## 🎯 **Session Overview**

**Date**: 2025-08-18  
**Status**: Ready for Phase 1 Cleanup Execution  
**Goal**: Remove 45 old ML system files safely while preserving Alpha v1 and core system functionality

---

## 📊 **Current System State**

### **Alpha v1 Status** ✅ **WORKING**
- **Training**: `python tools/train_alpha_v1.py --symbols SPY,TSLA` ✅
- **Validation**: `python tools/validate_alpha.py reports/alpha_eval.json` ✅
- **Walkforward**: `python scripts/walkforward_alpha_v1.py --symbols SPY --train-len 50 --test-len 20 --stride 10 --warmup 10` ✅
- **Comparison**: `python scripts/compare_walkforward.py --symbols SPY` ✅

### **Documentation Status** ✅ **COMPLETE**
- **`docs/SYSTEM_AUDIT_DOCUMENTATION.md`** - Comprehensive folder-by-folder analysis
- **`docs/AUDIT_SUMMARY.md`** - Concise audit summary and recommendations
- **`docs/FINAL_CLEANUP_PLAN.md`** - Complete execution plan with specific commands
- **`docs/ALPHA_V1_SYSTEM_OVERVIEW.md`** - Alpha v1 system documentation
- **`docs/ALPHA_V1_DEPENDENCIES.md`** - Alpha v1 dependency mapping

### **File Classification** ✅ **COMPLETE**
- **🟢 SAFE TO REMOVE**: 45 files (old ML system)
- **🟡 KEEP WITH CAUTION**: 25 files (legacy components)
- **🔴 MUST KEEP**: 10 files (core system dependencies)

---

## 🗑️ **Phase 1: Safe Removals (45 files)**

### **Old ML Components (15 files)**
```bash
# Remove old ML components
rm core/ml/profit_learner.py
rm core/ml/visualizer.py
rm core/ml/warm_start.py
rm ml/profit_learner.py
rm ml/visualizer.py
rm ml/warm_start.py
rm ml/runtime.py
rm scripts/ml_walkforward.py
rm scripts/generate_ml_plots.py
rm scripts/auto_ml_analysis.py
rm scripts/test_ml_trading.py
rm scripts/test_ml_recording.py
rm scripts/train_with_persistence.py
rm scripts/analyze_ml_learning.py
rm tests/ml/test_tripwires.py
```

### **Old Configuration Files (20 files)**
```bash
# Remove old configuration files
rm config/ml_backtest_config.json
rm config/ml_backtest_spy.json
rm config/ml_backtest_aapl.json
rm config/ml_backtest_btc.json
rm config/ml_backtest_goog.json
rm config/ml_backtest_tsla.json
rm config/ml_backtest_test_config.json
rm config/ml_backtest_unified.json
rm config/ml_config.yaml
rm config/paper_config.json
rm config/paper_trading_config.json
rm config/enhanced_paper_trading_config.json
rm config/enhanced_paper_trading_config_unified.json
rm config/enhanced_paper_trading.yaml
rm config/ibkr_config.json
rm config/live_config_ibkr.json
rm config/live_config.json
rm config/live_profile.json
rm config/strategies_config.json
rm config/strategies.yaml
```

### **Old Test Files (10 files)**
```bash
# Remove old test files
rm tests/ml/test_model_golden.py
rm tests/ml/test_feature_stats.py
rm tests/ml/test_model_runtime.py
rm tests/ml/test_score_mapping.py
rm tests/walkforward/test_repro_and_parallel.py
rm tests/walkforward/test_no_lookahead.py
rm tests/walkforward/test_data_sanity_integration.py
rm tests/walkforward/test_fold_integrity.py
rm tests/walkforward/test_metrics_consistency.py
rm tests/sanity/test_cases.py
```

---

## ⚠️ **Components to Keep (35 files)**

### **Legacy Components (25 files) - KEEP WITH CAUTION**
```
strategies/base.py                 # Used by composer system
strategies/ensemble_strategy.py    # Used by composer system
strategies/regime_aware_ensemble.py # Used by composer system
strategies/factory.py              # Used by composer system
strategies/__init__.py             # Package initialization
strategies/README.md               # Documentation
signals/condition.py               # Used by strategies
signals/__init__.py                # Package initialization
signals/README.md                  # Documentation
features/ensemble.py               # Used by strategies
scripts/walkforward_framework.py   # Used by multiple scripts and tests
scripts/paper_runner.py            # Used by Makefile and daily maintenance
scripts/canary_runner.py           # Used by Makefile and CI/CD
scripts/monitor_performance.py     # Used by monitoring system
scripts/health_check.py            # Used by health monitoring
scripts/check_data_sources.py      # Used by data validation
scripts/check_ibkr_connection.py   # Used by IBKR validation
scripts/flatten_positions.py       # Used by position management
tests/unit/test_returns_properties.py # Used by test framework
tests/meta/test_meta_core.py      # Used by test framework
tests/backtest/test_*.py          # Used by backtest testing
tests/brokers/test_*.py           # Used by broker testing
tests/live/test_*.py              # Used by live testing
tests/utils/test_*.py             # Used by utility testing
tests/test_v02_modules.py         # Used by module testing
```

### **Core System Components (10 files) - MUST KEEP**
```
core/engine/paper.py               # Uses strategies and signals
core/factory.py                    # Uses IBKR config
core/engine/backtest.py            # Uses old ML profit learner (but safely)
tools/self_check.py                # Uses IBKR config
tools/guardrails.py                # Uses IBKR config
tools/daily_maintenance.py         # Uses paper_runner and canary_runner
Makefile                           # Uses paper_runner and canary_runner
.github/workflows/smoke.yml        # Uses canary_runner
scripts/run_multi_asset_walkforward.py # Uses ml_backtest configs
scripts/start_performance_analysis.sh # Uses ml_backtest configs
```

---

## 🔍 **Alpha v1 Core Components (PROTECTED)**

### **Training & Evaluation**
```
tools/train_alpha_v1.py           # Alpha v1 training script
tools/validate_alpha.py           # Alpha v1 validation script
ml/trainers/train_linear.py       # Ridge regression trainer
ml/eval/alpha_eval.py             # Evaluation logic
ml/features/build_daily.py        # Feature engineering
```

### **Testing & Validation**
```
scripts/walkforward_alpha_v1.py   # Walkforward testing
scripts/compare_walkforward.py    # Results comparison
tests/ml/test_leakage_guards.py   # Leakage prevention tests
tests/ml/test_alpha_eval_contract.py # Evaluation contract tests
```

### **Core System Dependencies**
```
core/engine/backtest.py           # Backtesting engine
core/walk/ml_pipeline.py          # ML pipeline integration
core/walk/folds.py                # Walkforward fold generation
core/sim/simulate.py              # Trading simulation
core/metrics/stats.py             # Performance metrics
core/data_sanity.py               # Data validation
```

### **Configuration**
```
config/features.yaml              # Feature definitions
config/models.yaml                # Model configurations
config/base.yaml                  # Base configuration
config/data_sanity.yaml           # Data validation config
config/guardrails.yaml            # System guardrails
```

---

## 🚀 **Execution Plan**

### **Step 1: Execute Safe Removals**
```bash
# Run the removal commands from Phase 1 above
# This will remove 45 old ML system files
```

### **Step 2: Validate Alpha v1**
```bash
# Test Alpha v1 functionality
python tools/train_alpha_v1.py --symbols SPY,TSLA
python tools/validate_alpha.py reports/alpha_eval.json
python scripts/walkforward_alpha_v1.py --symbols SPY --train-len 50 --test-len 20 --stride 10 --warmup 10
python scripts/compare_walkforward.py --symbols SPY
```

### **Step 3: Validate Core System**
```bash
# Test core system functionality
python scripts/paper_runner.py --symbols SPY --poll-sec 1 --steps 2
python scripts/canary_runner.py --symbols SPY --poll-sec 1 --steps 2
```

### **Step 4: Update Documentation**
```bash
# Update documentation to remove references to old components
# Update MASTER_DOCUMENTATION.md
# Update README.md
# Update guides and runbooks
```

---

## 📋 **Validation Checklist**

### **Pre-Cleanup Validation**
- [ ] Alpha v1 training works
- [ ] Alpha v1 validation works
- [ ] Alpha v1 walkforward works
- [ ] Alpha v1 comparison works
- [ ] Core system scripts work
- [ ] CI/CD pipeline works

### **Post-Cleanup Validation**
- [ ] Alpha v1 training still works
- [ ] Alpha v1 validation still works
- [ ] Alpha v1 walkforward still works
- [ ] Alpha v1 comparison still works
- [ ] Core system scripts still work
- [ ] CI/CD pipeline still works
- [ ] No broken imports or references

### **Documentation Validation**
- [ ] MASTER_DOCUMENTATION.md updated
- [ ] README.md updated
- [ ] Guides updated
- [ ] Runbooks updated
- [ ] No broken links or references

---

## 🎯 **Success Criteria**

### **Immediate Success**
- ✅ 45 old ML system files removed
- ✅ Alpha v1 functionality preserved
- ✅ Core system functionality preserved
- ✅ CI/CD pipeline preserved
- ✅ Documentation updated

### **Long-term Success**
- ✅ Cleaner codebase architecture
- ✅ Reduced maintenance burden
- ✅ Focused development on Alpha v1
- ✅ Easier onboarding for new developers

---

## 📚 **Reference Documents**

### **Audit Documentation**
- `docs/SYSTEM_AUDIT_DOCUMENTATION.md` - Comprehensive analysis
- `docs/AUDIT_SUMMARY.md` - Quick reference
- `docs/FINAL_CLEANUP_PLAN.md` - Execution plan

### **Alpha v1 Documentation**
- `docs/ALPHA_V1_SYSTEM_OVERVIEW.md` - System overview
- `docs/ALPHA_V1_DEPENDENCIES.md` - Dependency mapping
- `MASTER_DOCUMENTATION.md` - Master documentation

### **System Documentation**
- `README.md` - Main project documentation
- `docs/guides/` - User guides
- `docs/runbooks/` - Operations runbooks

---

## 🔧 **Troubleshooting**

### **If Alpha v1 Breaks**
1. Check if any Alpha v1 components were accidentally removed
2. Verify all Alpha v1 dependencies are intact
3. Check import statements in Alpha v1 files
4. Restore from git if necessary

### **If Core System Breaks**
1. Check if any core system components were accidentally removed
2. Verify all core system dependencies are intact
3. Check import statements in core system files
4. Restore from git if necessary

### **If CI/CD Breaks**
1. Check if any CI/CD components were accidentally removed
2. Verify all CI/CD dependencies are intact
3. Check CI/CD configuration files
4. Restore from git if necessary

---

## 📞 **Next Session Actions**

1. **Execute Phase 1 removals** using the provided commands
2. **Validate all functionality** using the validation checklist
3. **Update documentation** to reflect changes
4. **Commit changes** with clear commit message
5. **Plan Phase 2** (future legacy component replacement)

---

**Status**: ✅ **READY FOR EXECUTION**  
**Risk Level**: 🟢 **LOW** - Clear plan and validation procedures  
**Confidence**: 🎯 **HIGH** - Alpha v1 fully protected  
**Next Action**: 🚀 **Execute Phase 1 safe removals**
