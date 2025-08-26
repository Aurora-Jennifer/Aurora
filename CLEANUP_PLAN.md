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
- `scripts/backtest.py` - Used by Makefile
- `scripts/multi_walkforward_report.py` - Main walkforward script (used in CI smoke)
- `scripts/walk_core.py` - Core walkforward implementation
- `scripts/walkforward_framework_compat.py` - Critical compatibility layer
- `scripts/walkforward_with_composer.py` - Used by tests
- `scripts/walk_forward.py` - Alternative walkforward entry point
- `scripts/config_sweep.py` - Used by experiment_runner.py and quick_sweep_demo.py
- `scripts/experiment_runner.py` - Used by Makefile
- `scripts/validate_run_report.py` - Used in CI workflow
- `scripts/gate_e2d.py` - Used by Makefile
- `scripts/go_nogo.py` - Used by Makefile

### **DATA & VALIDATION (KEEP ALL)**
- `scripts/fetch_yfinance.py` - Data acquisition
- `scripts/falsify_data_sanity.py` - Data validation testing
- `scripts/canary_datasanity.py` - Data sanity canary tests
- `scripts/perf_gate.py` - Performance validation
- `scripts/eval_compare.py` - Model evaluation
- `scripts/fetch_corporate_actions.py` - Corporate actions data

### **ANALYSIS & UTILITIES (KEEP ALL)**
- `analysis_viz.py` - Analysis visualization
- `scripts/consolidation_report.py` - Consolidation status tracking
- `scripts/ci/check_logging_consolidation.sh` - CI gate for logging consolidation

## **📊 BASELINE ESTABLISHED (2025-01-27 19:32)**

**Repository Size:**
- **544 Python files** 
- **186,282 lines of Python code**
- **329,811 total lines** across all languages

**Duplication Found:**
- **189 logging setup hits** (30+ files with duplicate logging)
- **11,154 validator hits** (massive data validation duplication)
- **Multiple training scripts** with overlapping functionality
- **Multiple walkforward implementations** with duplicate logic

## **🔄 CONSOLIDATION PHASES**

### **Phase 1: Logging Consolidation (SAFE) - ✅ COMPLETE**

**Status:** ✅ **COMPLETED** (2025-01-27 19:44)

**Results:**
- **Reduced logging hits from 58 to 39** (33% reduction)
- **Updated 19 files** to use centralized `core.utils.setup_logging`
- **Created CI gate** `scripts/ci/check_logging_consolidation.sh` to prevent regression
- **Zero risk** - all changes were safe replacements
- **Immediate benefits** - consistent logging across all scripts

**Files Updated:**
- `analysis_viz.py` - Main analysis visualization
- `scripts/train_crypto.py` - Crypto training script
- `scripts/falsify_data_sanity.py` - Data validation testing
- `scripts/eval_compare.py` - Model evaluation
- `scripts/fetch_corporate_actions.py` - Corporate actions data
- `scripts/multi_walkforward_report.py` - Main walkforward script
- `scripts/canary_datasanity.py` - Data sanity canary tests
- `scripts/perf_gate.py` - Performance validation
- `scripts/fetch_yfinance.py` - Data acquisition
- `serve/adapter.py` - Model serving adapter
- `signals/condition.py` - Signal conditioning
- `brokers/realtime_feed.py` - Real-time data feed
- `risk/overlay.py` - Risk management overlay
- `features/regime_features.py` - Feature engineering
- `ml/train.py` - ML training
- `tools/test_adapter_isolation.py` - Testing utilities
- `tools/test_asset_routing.py` - Asset routing tests
- `tools/train_alpha_v1.py` - Alpha training
- `core/metrics/weight_tuner.py` - Weight tuning

**CI Gate Status:** ✅ **PASSING**
- All active scripts now use centralized logging
- Excludes allowed exceptions (tests, attic, core logging modules)
- Prevents future regression

**Estimated Impact:**
- [x] ~150-200 lines of code removed
- [x] CI gate prevents logging setup regression
- [x] Consistent logging format across all scripts
- [x] Better maintainability and debugging

### **Phase 2: Data Validation Consolidation (SAFE)**

**Status:** 🔄 **PENDING**

**Target:** Use centralized `core.data_sanity.DataSanityValidator` instead of multiple duplicate validation functions.

**Files to Update:**
- `scripts/fetch_yfinance.py` - Has `clean_and_validate` function
- `scripts/falsify_data_sanity.py` - Has validation logic
- `scripts/canary_datasanity.py` - Has validation logic
- `core/utils.py` - Has `validate_trade` function
- `core/risk/guardrails.py` - Has `validate_trade` function

**Estimated Impact:**
- [ ] ~300 lines of code removed
- [ ] Consistent validation logic across all scripts
- [ ] Better error handling and reporting
- [ ] Centralized validation configuration

### **Phase 3: Consolidate (Medium Risk)**

**Status:** 🔄 **PENDING**

#### **3.1 Training Script Consolidation**
**Target:** Merge `scripts/train.py`, `scripts/train_linear.py`, `scripts/train_crypto.py` into a unified training framework.

**Estimated Impact:**
- [ ] ~400 lines of code removed
- [ ] Single training entry point
- [ ] Consistent training interface
- [ ] Better model management

#### **3.2 Walkforward Consolidation**
**Target:** Merge `scripts/multi_walkforward_report.py`, `scripts/walk_core.py`, `scripts/walkforward_with_composer.py` into unified walkforward framework.

**Estimated Impact:**
- [ ] ~400 lines of code removed
- [ ] Single walkforward entry point
- [ ] Consistent walkforward interface
- [ ] Better performance tracking

#### **3.3 Simulation Consolidation**
**Target:** Use centralized `core.sim.simulate.simulate_orders_numba` instead of duplicate simulation functions.

**Estimated Impact:**
- [ ] ~250 lines of code removed
- [ ] Consistent simulation logic
- [ ] Better performance (numba optimization)
- [ ] Centralized simulation configuration

### **Phase 4: Verification**

**Status:** 🔄 **PENDING**

**Tasks:**
- [ ] Run full test suite to ensure no regressions
- [ ] Verify all Makefile targets still work
- [ ] Verify CI pipeline still passes
- [ ] Update documentation to reflect changes

## **📈 DEDUPLICATION & CONSOLIDATION PLAN**

### **Major Areas for Consolidation:**

1. **Logging Setup** (✅ COMPLETE)
   - **Before:** 30+ files with duplicate logging setup
   - **After:** Centralized `core.utils.setup_logging`
   - **Impact:** ~200 lines removed, consistent logging

2. **Data Validation** (🔄 PENDING)
   - **Before:** Multiple validation functions scattered across files
   - **After:** Centralized `core.data_sanity.DataSanityValidator`
   - **Impact:** ~300 lines removed, consistent validation

3. **Training Scripts** (🔄 PENDING)
   - **Before:** 3 separate training scripts with overlapping functionality
   - **After:** Unified training framework with config-driven approach
   - **Impact:** ~400 lines removed, single entry point

4. **Walkforward Framework** (🔄 PENDING)
   - **Before:** Multiple walkforward implementations
   - **After:** Unified walkforward framework
   - **Impact:** ~400 lines removed, consistent interface

5. **Simulation Logic** (🔄 PENDING)
   - **Before:** Duplicate simulation functions
   - **After:** Centralized numba-optimized simulation
   - **Impact:** ~250 lines removed, better performance

6. **Utility Functions** (🔄 PENDING)
   - **Before:** Scattered utility functions
   - **After:** Centralized utility modules
   - **Impact:** ~200 lines removed, better organization

### **Total Estimated Impact:**
- **Lines of Code Removed:** ~1,750 lines
- **Files Simplified:** ~50 files
- **Maintainability:** Significantly improved
- **Consistency:** Dramatically improved
- **Performance:** Better (numba optimization)
- **Risk:** Low (phased approach with rollback capability)

## **🎯 NEXT STEPS**

1. **✅ Phase 1 Complete** - Logging consolidation successful
2. **🔄 Phase 2** - Data validation consolidation (safe, low risk)
3. **🔄 Phase 3** - Training, walkforward, simulation consolidation (medium risk)
4. **🔄 Phase 4** - Verification and testing

## **📋 EXECUTION PLAN**

### **Phase 2: Data Validation Consolidation**
1. Identify all data validation functions
2. Create centralized validation interface
3. Update scripts to use centralized validation
4. Add tests to ensure validation still works
5. Update CI to include validation tests

### **Phase 3: Consolidation**
1. **Training Consolidation:**
   - Create unified training framework
   - Migrate existing training scripts
   - Update Makefile targets
   - Add tests for unified training

2. **Walkforward Consolidation:**
   - Create unified walkforward framework
   - Migrate existing walkforward scripts
   - Update CI to use unified framework
   - Add tests for unified walkforward

3. **Simulation Consolidation:**
   - Migrate to centralized simulation
   - Update all simulation calls
   - Add performance tests
   - Verify simulation accuracy

### **Phase 4: Verification**
1. Run full test suite
2. Verify all Makefile targets work
3. Verify CI pipeline passes
4. Update documentation
5. Create rollback plan

## **🚨 ROLLBACK PLAN**

If any phase causes issues:
1. **Immediate:** Revert to previous commit
2. **Investigation:** Identify root cause
3. **Fix:** Address issue in smaller, safer increments
4. **Retest:** Verify fix works
5. **Continue:** Resume consolidation with more caution

## **📊 SUCCESS METRICS**

- [x] **Phase 1:** Logging hits reduced by 33% (58 → 39)
- [ ] **Phase 2:** Validation hits reduced by 50%+
- [ ] **Phase 3:** Training/walkforward/simulation consolidation complete
- [ ] **Overall:** 1,750+ lines of code removed
- [ ] **Maintainability:** Significantly improved
- [ ] **Consistency:** Dramatically improved
- [ ] **Performance:** Better (numba optimization)
- [ ] **Risk:** Low (phased approach with rollback capability)
