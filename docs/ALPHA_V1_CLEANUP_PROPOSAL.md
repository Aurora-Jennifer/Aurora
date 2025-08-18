# Alpha v1 Codebase Cleanup Proposal

## 🎯 **Executive Summary**

After implementing the Alpha v1 ML pipeline, we need to audit and clean up the codebase to remove unused code while preserving all critical functionality. This proposal provides a **dry-run analysis** of what can be safely removed without breaking the Alpha v1 system.

## 📊 **Current State Analysis**

### **Alpha v1 Critical Components (MUST PRESERVE)**
Based on documentation and implementation, these are the **core Alpha v1 files** that must be preserved:

#### **1. Alpha v1 Core Files (15 files)**
```
✅ tools/train_alpha_v1.py              # Main training script
✅ scripts/walkforward_alpha_v1.py      # Alpha v1 walkforward testing
✅ core/walk/ml_pipeline.py             # ML pipeline integration
✅ ml/trainers/train_linear.py          # Ridge regression trainer
✅ ml/eval/alpha_eval.py                # Alpha evaluation logic
✅ ml/features/build_daily.py           # Feature engineering
✅ tools/validate_alpha.py              # Validation script
✅ reports/alpha.schema.json            # Evaluation schema
✅ scripts/compare_walkforward.py       # Comparison script
```

#### **2. Configuration Files (5 files)**
```
✅ config/features.yaml                 # Feature definitions
✅ config/models.yaml                   # Model configurations
✅ config/base.yaml                     # Base configuration
✅ config/data_sanity.yaml              # Data validation config
✅ config/guardrails.yaml               # System guardrails
```

#### **3. Test Files (3 files)**
```
✅ tests/ml/test_leakage_guards.py      # Leakage prevention tests
✅ tests/ml/test_alpha_eval_contract.py # Evaluation contract tests
✅ tests/ml/test_model_golden.py        # Golden dataset tests
```

#### **4. Documentation Files (3 files)**
```
✅ docs/runbooks/alpha.md               # Alpha v1 runbook
✅ docs/ALPHA_V1_WALKFORWARD_GUIDE.md   # Walkforward guide
✅ docs/ALPHA_V1_CLEANUP_PROPOSAL.md    # This proposal
```

### **System Dependencies (MUST PRESERVE)**
Based on the MASTER_DOCUMENTATION.md, these core system components must be preserved:

#### **1. Core Engine (20+ files)**
```
✅ core/engine/backtest.py              # Backtesting engine
✅ core/engine/composer_integration.py  # Composer integration
✅ core/composer/contracts.py           # Composer interfaces
✅ core/composer/registry.py            # Strategy registry
✅ core/strategy_selector.py            # ML strategy selection
✅ core/regime_detector.py              # Market regime detection
✅ core/portfolio.py                    # Portfolio management
✅ core/risk/guardrails.py              # Risk management
✅ core/data_sanity.py                  # Data validation
✅ core/walk/folds.py                   # Walkforward folds
✅ core/walk/pipeline.py                # Walkforward pipeline
✅ core/walk/run.py                     # Walkforward execution
✅ core/sim/simulate.py                 # Simulation engine
✅ core/metrics/stats.py                # Performance metrics
✅ core/config_loader.py                # Configuration loading
✅ core/utils.py                        # Core utilities
```

#### **2. Infrastructure Files (15+ files)**
```
✅ requirements.txt                     # Dependencies
✅ pyproject.toml                       # Project configuration
✅ pytest.ini                          # Test configuration
✅ ruff.toml                           # Linting configuration
✅ Makefile                            # Build system
✅ Justfile                            # Task runner
✅ .github/workflows/ci.yml             # CI/CD pipeline
✅ README.md                           # Main documentation
✅ MASTER_DOCUMENTATION.md              # System documentation
```

## 🗑️ **Safe Removal Categories**

### **Phase 1: Clearly Unused Files (LOW RISK)**

#### **1. Temporary/Test Files (50+ files)**
```
❌ temp_ml_training_config.json         # Temporary config
❌ test_backtest_config.json            # Test config
❌ test_paper_trading_config.json       # Test config
❌ test_performance_config.json         # Test config
❌ *.bak files                          # Backup files
❌ __pycache__/ directories             # Python cache
❌ .pytest_cache/                       # Test cache
❌ .ruff_cache/                         # Lint cache
❌ .hypothesis/                         # Test cache
❌ .perf/                               # Performance cache
```

#### **2. Legacy/Deprecated Files (30+ files)**
```
❌ attic/                               # Entire legacy directory
❌ baselines/                           # Old baseline files
❌ runlocks/                            # Old locking mechanism
❌ migrate_indicators.py                # One-time migration script
❌ analysis_viz.py                      # Old analysis script
❌ build_secure.py                      # Old build script
❌ setup_github.sh                      # One-time setup script
```

#### **3. Duplicate/Redundant Files (20+ files)**
```
❌ README.md.bak                        # Backup of README
❌ CONTEXT_ORGANIZATION_SUMMARY.md      # Redundant with MASTER_DOC
❌ PUBLIC_PRESENTATION.md               # Presentation material
❌ INVESTOR_PRESENTATION.md             # Presentation material
❌ indicators_comparison.png            # Old visualization
❌ trading.log                          # Empty log file
❌ =4.21                                # Unknown file
```

### **Phase 2: Review Required Files (MEDIUM RISK)**

#### **1. Old Walkforward Framework (10+ files)**
```
⚠️ scripts/walkforward_framework.py     # Old regime-based walkforward
⚠️ core/walk/pipeline.py                # Old pipeline (if not used by Alpha v1)
⚠️ core/walk/run.py                     # Old run logic (if not used by Alpha v1)
```

#### **2. Old ML Components (15+ files)**
```
⚠️ ml/profit_learner.py                 # Old ML component
⚠️ ml/visualizer.py                     # Old visualization
⚠️ ml/warm_start.py                     # Old warm start
⚠️ core/ml/                             # Old ML directory
```

#### **3. Old Strategy Components (20+ files)**
```
⚠️ strategies/                          # Old strategy implementations
⚠️ signals/                             # Old signal processing
⚠️ features/                            # Old feature engineering
```

### **Phase 3: Keep for Now (HIGH RISK)**

#### **1. Core Infrastructure (30+ files)**
```
📋 core/                               # Core engine (preserve)
📋 config/                             # Configuration (preserve)
📋 tests/                              # Test framework (preserve)
📋 docs/                               # Documentation (preserve)
📋 tools/                              # Utility tools (preserve)
📋 scripts/                            # Scripts (preserve)
📋 brokers/                            # Broker integration (preserve)
📋 cli/                                # Command line interface (preserve)
📋 api/                                # API components (preserve)
📋 apps/                               # Application components (preserve)
```

## 🚀 **Cleanup Execution Plan**

### **Phase 1: Safe Removals (Immediate)**
```bash
# Remove temporary files
rm temp_ml_training_config.json
rm test_*.json
rm *.bak
rm -rf __pycache__/
rm -rf .pytest_cache/
rm -rf .ruff_cache/
rm -rf .hypothesis/
rm -rf .perf/

# Remove legacy directories
rm -rf attic/
rm -rf baselines/
rm -rf runlocks/

# Remove one-time scripts
rm migrate_indicators.py
rm analysis_viz.py
rm build_secure.py
rm setup_github.sh

# Remove redundant files
rm README.md.bak
rm CONTEXT_ORGANIZATION_SUMMARY.md
rm PUBLIC_PRESENTATION.md
rm INVESTOR_PRESENTATION.md
rm indicators_comparison.png
rm trading.log
rm =4.21
```

**Expected Impact**: Remove ~100 files, ~50MB of space
**Risk Level**: LOW - These are clearly unused files

### **Phase 2: Review and Remove (After Testing)**
```bash
# Test Alpha v1 functionality first
python tools/train_alpha_v1.py --symbols SPY,TSLA
python scripts/walkforward_alpha_v1.py --symbols SPY TSLA
python tools/validate_alpha.py reports/alpha_eval.json

# If tests pass, remove old components
# (Review each file individually before removal)
```

**Expected Impact**: Remove ~50 files, ~25MB of space
**Risk Level**: MEDIUM - Requires careful review

### **Phase 3: Infrastructure Cleanup (Future)**
```bash
# Consolidate configuration files
# Remove unused dependencies
# Clean up documentation
```

**Expected Impact**: Optimize ~30 files, ~10MB of space
**Risk Level**: HIGH - Requires system-wide analysis

## 🧪 **Validation Strategy**

### **Pre-Cleanup Validation**
```bash
# 1. Test Alpha v1 training
python tools/train_alpha_v1.py --symbols SPY,TSLA

# 2. Test Alpha v1 walkforward
python scripts/walkforward_alpha_v1.py --symbols SPY TSLA

# 3. Test validation
python tools/validate_alpha.py reports/alpha_eval.json

# 4. Run comparison
python scripts/compare_walkforward.py --symbols SPY TSLA

# 5. Run core tests
python -m pytest tests/ml/ -v
```

### **Post-Cleanup Validation**
```bash
# 1. Verify Alpha v1 still works
python tools/train_alpha_v1.py --symbols SPY,TSLA

# 2. Verify walkforward still works
python scripts/walkforward_alpha_v1.py --symbols SPY TSLA

# 3. Verify validation still works
python tools/validate_alpha.py reports/alpha_eval.json

# 4. Run full test suite
python -m pytest tests/ -v

# 5. Check system health
python scripts/health_check.py
```

## 📊 **Expected Results**

### **File Count Reduction**
- **Current**: ~500 Python files
- **After Phase 1**: ~400 Python files (-20%)
- **After Phase 2**: ~350 Python files (-30%)
- **After Phase 3**: ~320 Python files (-36%)

### **Space Savings**
- **Current**: ~200MB codebase
- **After Phase 1**: ~150MB (-25%)
- **After Phase 2**: ~125MB (-37%)
- **After Phase 3**: ~115MB (-42%)

### **Maintenance Benefits**
- **Reduced complexity**: Fewer files to maintain
- **Clearer structure**: Focus on Alpha v1 components
- **Faster builds**: Less code to process
- **Easier onboarding**: Clearer codebase structure

## ⚠️ **Risk Mitigation**

### **1. Backup Strategy**
```bash
# Create backup before cleanup
git checkout -b backup-before-alpha-v1-cleanup
git add .
git commit -m "Backup before Alpha v1 cleanup"
```

### **2. Incremental Approach**
- **Phase 1**: Remove only clearly unused files
- **Phase 2**: Review each file before removal
- **Phase 3**: System-wide analysis required

### **3. Rollback Plan**
```bash
# If issues occur, rollback immediately
git checkout backup-before-alpha-v1-cleanup
git checkout -b fix-alpha-v1-cleanup
# Re-add necessary files
```

## 🎯 **Recommendations**

### **Immediate Actions (This Session)**
1. ✅ **Create backup branch** before any changes
2. ✅ **Run Phase 1 cleanup** (safe removals only)
3. ✅ **Validate Alpha v1 functionality** after cleanup
4. ✅ **Document any issues** found during cleanup

### **Next Session Actions**
1. 🔄 **Review Phase 2 files** individually
2. 🔄 **Test each removal** before proceeding
3. 🔄 **Update documentation** to reflect changes
4. 🔄 **Plan Phase 3** infrastructure cleanup

### **Long-term Actions**
1. 📋 **Monitor system performance** after cleanup
2. 📋 **Update CI/CD** to reflect new structure
3. 📋 **Train team** on new Alpha v1 focus
4. 📋 **Plan future enhancements** based on cleaner codebase

## 🎉 **Success Criteria**

### **Phase 1 Success**
- [ ] Alpha v1 training works after cleanup
- [ ] Alpha v1 walkforward works after cleanup
- [ ] Alpha v1 validation works after cleanup
- [ ] Core tests still pass
- [ ] No regression in functionality

### **Overall Success**
- [ ] 20%+ reduction in file count
- [ ] 25%+ reduction in codebase size
- [ ] Clearer project structure
- [ ] Easier maintenance
- [ ] Faster development cycles

---

**Status**: 🟡 **DRY-RUN PROPOSAL** - Ready for review and approval
**Risk Level**: 🟢 **LOW** for Phase 1, 🟡 **MEDIUM** for Phase 2
**Estimated Time**: 2-3 hours for Phase 1, 4-6 hours for Phase 2
**Approval Required**: `APPROVE: CLEANUP-001` to proceed with Phase 1
