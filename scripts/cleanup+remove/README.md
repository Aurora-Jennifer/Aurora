# CLEANUP + REMOVE ORGANIZATION

This folder contains files staged for removal from the codebase, organized by risk level.

## 📁 PHASE 1: IMMEDIATE (100% Safe to Remove)

**5 files** - Completely unused, no references anywhere:

- `README.md` - Documentation file not referenced
- `accept_l0.sh` - Shell script not used
- `pre_push_smoke.sh` - Shell script not used  
- `setup_cron.sh` - Shell script not used
- `start_performance_analysis.sh` - Shell script not used

**Status:** ✅ Ready for immediate deletion
**Risk:** None - these files are not referenced anywhere

## 📁 PHASE 2: CONSERVATIVE (90% Safe to Remove)

**29 files** - Experimental/utility files not used in Makefile or CI:

### Test Files (4):
- `test_crypto_training.py` - Test file, may be superseded
- `test_determinism.py` - Test file, may be superseded
- `test_identical_closes.py` - Test file, may be superseded  
- `test_leakage_guards.py` - Test file, may be superseded

### Experimental/Utility Files (25):
- `SUPER_EASY.py` - Appears experimental
- `experiments.py` - Experimental code
- `falsification_tests.py` - Test utilities
- `feature_first_upgrade.py` - Upgrade script
- `final_error_report.py` - Error reporting utility
- `iterate_backtest.py` - Backtesting utility
- `metrics_server.py` - Server utility
- `multi_symbol_test.py` - Test utility
- `oms_run.py` - OMS utility
- `parity_doctor.py` - Parity utility
- `preflight.py` - Preflight utility
- `process_kaggle_data.py` - Data processing utility
- `production_banner.py` - Banner utility
- `quick_momentum_upgrade.py` - Upgrade script
- `readiness_check.py` - Check utility
- `retrain_momentum_model.py` - Retraining script
- `rolling_windows.py` - Window utility
- `run_data_sanity_tests.py` - Test runner
- `run_enhanced_data_sanity_tests.py` - Enhanced test runner
- `smart_xgb_upgrade.py` - Upgrade script
- `trace_bad_prices.py` - Tracing utility
- `trace_lookahead.py` - Tracing utility
- `use_xgboost.py` - XGBoost utility
- `verification_summary.py` - Summary utility
- `walkforward_alpha_v1.py` - Alpha version

**Status:** 🔍 Review recommended before deletion
**Risk:** Low - these files are not in Makefile or CI, but may have edge case usage

## 📁 REMAINING IN ARCHIVE (Keep These)

**21 files** - Actively used in Makefile, CI, or critical functionality:

### Makefile/CI Critical (15):
- `eval_oof.py` - Used in Makefile
- `fix_golden_snapshot.py` - Used in Makefile
- `gate_data.py` - Used in Makefile
- `gate_pnl.py` - Used in Makefile
- `gate_signal.py` - Used in Makefile
- `gate_significance.py` - Used in Makefile
- `gate_wf.py` - Used in Makefile
- `hash_snapshot.py` - Used in Makefile
- `make_dashboard.py` - Used in Makefile
- `onnx_parity.py` - Used in Makefile
- `rollback.py` - Used in Makefile
- `validate_snapshot.py` - Used in Makefile
- `validate_run_hashes.py` - Used in CI
- `validate_schema.py` - Used in CI

### Gate/Validation Files (7):
- `gate_ablation.py` - Gate functionality
- `gate_parity_live.py` - Gate functionality
- `gate_promote.py` - Gate functionality
- `validate_metrics.py` - Validation functionality
- `walkforward_framework.py` - Framework functionality

## 🚀 EXECUTION PLAN

### Phase 1 (Immediate):
```bash
# Delete Phase 1 files (100% safe)
rm -rf scripts/cleanup+remove/phase1_immediate/
```

### Phase 2 (After Review):
```bash
# Delete Phase 2 files (after manual review)
rm -rf scripts/cleanup+remove/phase2_conservative/
```

## 📊 IMPACT SUMMARY

- **Total files staged for removal:** 34 files
- **Phase 1 (immediate):** 5 files
- **Phase 2 (conservative):** 29 files
- **Remaining in archive:** 21 files
- **Total reduction:** 62% of archive files

## 🔄 ROLLBACK

If any issues arise, files can be restored from this staging area:
```bash
# Restore Phase 1 files
mv scripts/cleanup+remove/phase1_immediate/* scripts/archive/

# Restore Phase 2 files  
mv scripts/cleanup+remove/phase2_conservative/* scripts/archive/
```

## ✅ VALIDATION

After each phase, run:
```bash
make smoke  # Verify core functionality
python scripts/tools/analyze_archive_usage.py  # Re-analyze usage
```
