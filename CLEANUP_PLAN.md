# REPOSITORY CLEANUP PLAN - REVISED (CONSERVATIVE)

## **CRITICAL FINDINGS - MUST KEEP THESE SCRIPTS**

After thorough analysis of imports, Makefile usage, and CI dependencies, here are the scripts that are **ACTUALLY USED** and must be preserved:

### **CORE TRAINING & EXECUTION (KEEP ALL)**
- `scripts/train.py` - Main training script
- `scripts/train_linear.py` - Used in Makefile targets (train-e0, train-e1, train-e2)
- `scripts/train_crypto.py` - Used by tests/test_crypto_contract_and_determinism.py
- `scripts/paper_runner.py` - Core paper trading (used by live_trading_workflow.py)
- `scripts/paper_broker.py` - Used by tests/test_paper_broker.py and benchmarks
- `scripts/canary_runner.py` - Used by tests/live/test_canary_runner_smoke.py
- `scripts/e2d.py` - Used by Makefile e2d target
- `scripts/runner.py` - Used by Makefile
- `scripts/backtest.py` - Used by Makefile backtest targets

### **WALKFORWARD FRAMEWORK (CRITICAL - KEEP ALL)**
- `scripts/multi_walkforward_report.py` - **MAIN walkforward script** (used in CI smoke)
- `scripts/walk_core.py` - Core walkforward implementation
- `scripts/walkforward_framework.py` - **DEPRECATED but still imported by 20+ test files**
- `scripts/walkforward_framework_compat.py` - **CRITICAL compatibility layer**
- `scripts/walkforward_with_composer.py` - Used by tests/test_composer_end_to_end.py
- `scripts/walk_forward.py` - Alternative walkforward entry point

### **CONFIGURATION & EXPERIMENTS (KEEP)**
- `scripts/config_sweep.py` - Used by experiment_runner.py and quick_sweep_demo.py
- `scripts/experiment_runner.py` - Used by Makefile

### **VALIDATION & GATES (KEEP)**
- `scripts/validate_run_report.py` - Used in CI workflow
- `scripts/gate_e2d.py` - Used by Makefile e2d-gate target
- `scripts/go_nogo.py` - Used by Makefile go-nogo targets

---

## **📊 BASELINE ESTABLISHED (2025-01-27 19:32)**

### **Repository Metrics**
- **Total Python files**: 544
- **Total lines of code**: 186,282 (Python only)
- **Total repository size**: 329,811 lines across all languages

### **Duplication Analysis**
- **Logging hits found**: 189 lines across all Python files
- **Validator hits found**: 11,154 lines (very high!)
- **Files with logging setup**: ~30+ files identified

### **Test Status**
- **6 collection errors** (mostly import issues)
- **127 warnings** (mostly pytest mark warnings)
- **Main issues**: Missing `_last`, `_safe_len` functions in `core.utils`

### **Baseline Files Created**
- `artifacts/consolidation_baseline/cloc.txt` - Code size metrics
- `artifacts/consolidation_baseline/py_file_count.txt` - Python file count
- `artifacts/consolidation_baseline/logging_hits.txt` - Logging setup locations
- `artifacts/consolidation_baseline/validator_hits.txt` - Data validation locations
- `artifacts/consolidation_baseline/pytest.txt` - Test status (with errors)
- `artifacts/consolidation_baseline/baseline_summary.md` - Comprehensive summary

### **Tools Created**
- `scripts/consolidation_report.py` - Automated consolidation status reporting
- `scripts/ci/check_logging_consolidation.sh` - CI gate for logging consolidation

---

## **🎯 DEDUPLICATION & CONSOLIDATION OPPORTUNITIES**

### **1. LOGGING SETUP CONSOLIDATION (HIGH IMPACT)**
**Problem**: 30+ duplicate logging setup functions across scripts
**Files affected**: 
- `scripts/fetch_yfinance.py` (lines 50-52)
- `scripts/live_trading_workflow.py` (lines 35-38)
- `scripts/train_crypto.py` (line 39)
- `scripts/multi_walkforward_report.py` (line 506)
- `scripts/falsify_data_sanity.py` (line 25)
- `scripts/eval_compare.py` (line 32)
- `scripts/perf_gate.py` (line 295)
- `scripts/fetch_corporate_actions.py` (line 16)
- `analysis_viz.py` (line 37)
- `serve/adapter.py` (line 21)
- `signals/condition.py` (line 499)
- `core/metrics/weight_tuner.py` (line 297)
- `risk/overlay.py` (line 589)
- `brokers/realtime_feed.py` (line 180)
- And 15+ more files...

**Solution**: 
```python
# Use existing centralized logging (already exists in core/utils.py)
from core.utils import setup_logging
logger = setup_logging('logs/script_name.log', logging.INFO)
```

**Impact**: **~200 lines of duplicate code removed**

### **2. DATA VALIDATION CONSOLIDATION (HIGH IMPACT)**
**Problem**: Multiple duplicate data validation functions
**Files affected**:
- `scripts/fetch_yfinance.py` (lines 121-175) - `clean_and_validate()`
- `scripts/fetch_yfinance.py` (lines 177-200) - `run_datasanity_validation()`
- `core/data_sanity.py` (lines 430-687) - `DataSanityValidator`
- `core/data_sanity/main.py` (lines 149-429) - `DataSanityValidator`

**Solution**: 
```python
# Use centralized DataSanityValidator
from core.data_sanity import DataSanityValidator
validator = DataSanityValidator()
clean_data, result = validator.validate_and_repair(data, symbol)
```

**Impact**: **~300 lines of duplicate code removed**

### **3. TRADE VALIDATION CONSOLIDATION (MEDIUM IMPACT)**
**Problem**: Multiple `validate_trade` functions
**Files affected**:
- `core/utils.py` (lines 116-176) - Main implementation
- `core/risk/guardrails.py` (lines 141-180) - Risk-specific validation
- `attic/_quarantine_20250819/guardrails.py` (lines 24-50) - Old implementation

**Solution**: 
```python
# Use unified validate_trade from core/utils.py
from core.utils import validate_trade
is_valid, reason = validate_trade(symbol, quantity, price, cash, action, positions, risk_limits)
```

**Impact**: **~100 lines of duplicate code removed**

### **4. SIMULATION ENGINE CONSOLIDATION (MEDIUM IMPACT)**
**Problem**: Duplicate simulation functions
**Files affected**:
- `core/sim/simulate.py` (lines 5-200) - Main `simulate_orders_numba()`
- `scripts/walkforward_with_composer.py` (lines 118-300) - `simulate_orders_python()`

**Solution**: 
```python
# Use unified simulation from core/sim/simulate.py
from core.sim.simulate import simulate_orders_numba
```

**Impact**: **~150 lines of duplicate code removed**

### **5. TRAINING SCRIPT CONSOLIDATION (HIGH IMPACT)**
**Problem**: Multiple large training scripts with overlapping functionality
**Files affected**:
- `scripts/train.py` (602 lines) - Main training
- `scripts/train_linear.py` (722 lines) - Linear model training
- `scripts/train_crypto.py` (482 lines) - Crypto training
- `scripts/live_trading_workflow.py` (389 lines) - Live training workflow

**Solution**: 
```python
# Create unified training framework
# scripts/train_unified.py - Single entry point for all training types
# Move model-specific logic to core/ml/trainers/
```

**Impact**: **~1000 lines consolidated, ~500 lines removed**

### **6. WALKFORWARD CONSOLIDATION (HIGH IMPACT)**
**Problem**: Multiple walkforward implementations
**Files affected**:
- `scripts/multi_walkforward_report.py` (706 lines) - Main implementation
- `scripts/walkforward_with_composer.py` (583 lines) - Composer integration
- `scripts/walk_core.py` - Core logic
- `scripts/walkforward_framework.py` - Deprecated framework

**Solution**: 
```python
# Consolidate into single walkforward engine
# scripts/walkforward_engine.py - Unified walkforward runner
# Move composer logic to core/walkforward/composer.py
```

**Impact**: **~800 lines consolidated, ~400 lines removed**

### **7. UTILITY FUNCTION CONSOLIDATION (MEDIUM IMPACT)**
**Problem**: Scattered utility functions
**Files affected**:
- `core/utils.py` - Core utilities
- `utils/logging.py` - Logging utilities
- `core/utils/__init__.py` - More utilities
- Multiple script files with utility functions

**Solution**: 
```python
# Consolidate all utilities into core/utils/
# Create specialized modules: core/utils/{logging,validation,calculation}.py
```

**Impact**: **~300 lines consolidated**

---

## **CONSOLIDATION IMPLEMENTATION PLAN**

### **PHASE 1: LOGGING CONSOLIDATION (SAFE)**
1. **Audit all logging setups** - Identify all duplicate logging configurations
2. **Update scripts** - Replace with `from core.utils import setup_logging`
3. **Test** - Verify all scripts still work
4. **Remove duplicates** - Delete duplicate logging code

### **PHASE 2: DATA VALIDATION CONSOLIDATION (SAFE)**
1. **Audit data validation functions** - Map all validation logic
2. **Enhance DataSanityValidator** - Add missing validation features
3. **Update scripts** - Replace custom validation with centralized validator
4. **Test** - Verify data validation still works correctly

### **PHASE 3: TRAINING CONSOLIDATION (MEDIUM RISK)**
1. **Create unified training framework** - `scripts/train_unified.py`
2. **Extract model-specific logic** - Move to `core/ml/trainers/`
3. **Update Makefile targets** - Point to unified trainer
4. **Test thoroughly** - Verify all training still works

### **PHASE 4: WALKFORWARD CONSOLIDATION (MEDIUM RISK)**
1. **Create unified walkforward engine** - `scripts/walkforward_engine.py`
2. **Extract composer logic** - Move to `core/walkforward/composer.py`
3. **Update imports** - Point all walkforward code to unified engine
4. **Test thoroughly** - Verify all walkforward functionality works

### **PHASE 5: UTILITY CONSOLIDATION (SAFE)**
1. **Audit utility functions** - Map all scattered utilities
2. **Consolidate into core/utils/** - Organize by function type
3. **Update imports** - Point all scripts to consolidated utilities
4. **Test** - Verify all functionality preserved

---

## **EXPECTED IMPACT**

### **Lines of Code Reduction**
- **Logging consolidation**: ~200 lines removed
- **Data validation consolidation**: ~300 lines removed  
- **Trade validation consolidation**: ~100 lines removed
- **Simulation consolidation**: ~150 lines removed
- **Training consolidation**: ~500 lines removed
- **Walkforward consolidation**: ~400 lines removed
- **Utility consolidation**: ~300 lines consolidated

**TOTAL**: **~1,950 lines of duplicate code removed**

### **File Count Reduction**
- **Current**: 96 scripts
- **After consolidation**: ~60 scripts
- **Reduction**: ~36 scripts (38% reduction)

### **Maintainability Improvements**
- **Single source of truth** for common functions
- **Consistent patterns** across all scripts
- **Easier debugging** - centralized logging and validation
- **Reduced bugs** - fewer places for bugs to hide
- **Faster development** - reuse existing patterns

---

## **REVISED SAFE CLEANUP STRATEGY**

### **PHASE 1: ORGANIZE WITHOUT DELETING (SAFE)**

**Create subdirectories and move files (no deletion):**

```bash
# Create organization structure
mkdir -p scripts/{core,training,walkforward,validation,experiments,archive}

# Move core execution scripts
mv scripts/{paper_runner.py,paper_broker.py,canary_runner.py,e2d.py,runner.py,backtest.py} scripts/core/

# Move training scripts  
mv scripts/{train.py,train_linear.py,train_crypto.py} scripts/training/

# Move walkforward scripts (CRITICAL - keep all)
mv scripts/{multi_walkforward_report.py,walk_core.py,walkforward_framework.py,walkforward_framework_compat.py,walkforward_with_composer.py,walk_forward.py} scripts/walkforward/

# Move validation scripts
mv scripts/{validate_run_report.py,gate_e2d.py,go_nogo.py} scripts/validation/

# Move experiment scripts
mv scripts/{config_sweep.py,experiment_runner.py} scripts/experiments/

# Move everything else to archive (SAFE - no deletion)
mv scripts/*.py scripts/archive/
```

### **PHASE 2: DEDUPLICATE (SAFE)**
1. **Logging consolidation** - Replace 30+ duplicate logging setups
2. **Data validation consolidation** - Use centralized DataSanityValidator
3. **Trade validation consolidation** - Use unified validate_trade function
4. **Utility consolidation** - Organize scattered utility functions

### **PHASE 3: CONSOLIDATE (MEDIUM RISK)**
1. **Training consolidation** - Create unified training framework
2. **Walkforward consolidation** - Create unified walkforward engine
3. **Simulation consolidation** - Use single simulation engine

### **PHASE 4: VERIFICATION**

**Test that everything still works:**

```bash
# Test core functionality
make smoke  # Should still work
make e2d    # Should still work
make paper  # Should still work

# Test imports
python -c "import scripts.paper_runner; print('paper_runner OK')"
python -c "import scripts.walkforward_framework; print('walkforward_framework OK')"
python -c "import scripts.train; print('train OK')"

# Run critical tests
pytest tests/test_paper_broker.py -v
pytest tests/live/test_canary_runner_smoke.py -v
pytest tests/walkforward/test_smoke_contract.py -v
```

---

## **WHAT THIS ACHIEVES**

1. **SAFETY FIRST**: No files are deleted, only consolidated
2. **MASSIVE REDUCTION**: ~2,000 lines of duplicate code removed
3. **BETTER ORGANIZATION**: Single source of truth for common functions
4. **IMPROVED MAINTAINABILITY**: Consistent patterns across all scripts
5. **FASTER DEVELOPMENT**: Reuse existing patterns instead of rewriting

## **NEXT STEPS**

1. **Fix test collection issues** (missing `_last`, `_safe_len` functions)
2. **Start Phase 1** (logging consolidation - SAFE)
3. **Execute Phase 2** (data validation consolidation - SAFE)
4. **Execute Phase 3** (training/walkforward consolidation - MEDIUM RISK)
5. **Verify everything works**

## **WHY THIS IS BETTER**

- **No deletion**: Everything is preserved, just consolidated
- **Massive reduction**: ~2,000 lines of duplicate code removed
- **Better architecture**: Single source of truth for common functions
- **Easier maintenance**: Consistent patterns across all scripts
- **Faster development**: Reuse existing patterns instead of rewriting

**This approach gives you both organization AND massive code reduction without risk.**
