# Next Session Quick Reference

## 🎯 **What to Do Next**

### **1. Execute Safe Removals (45 files)**
```bash
# Old ML Components (15 files)
rm core/ml/profit_learner.py core/ml/visualizer.py core/ml/warm_start.py
rm ml/profit_learner.py ml/visualizer.py ml/warm_start.py ml/runtime.py
rm scripts/ml_walkforward.py scripts/generate_ml_plots.py scripts/auto_ml_analysis.py
rm scripts/test_ml_trading.py scripts/test_ml_recording.py scripts/train_with_persistence.py
rm scripts/analyze_ml_learning.py tests/ml/test_tripwires.py

# Old Configuration Files (20 files)
rm config/ml_backtest_*.json config/ml_config.yaml
rm config/paper_config.json config/paper_trading_config.json
rm config/enhanced_paper_trading_*.json config/enhanced_paper_trading.yaml
rm config/ibkr_config.json config/live_config_*.json config/live_profile.json
rm config/strategies_config.json config/strategies.yaml

# Old Test Files (10 files)
rm tests/ml/test_model_golden.py tests/ml/test_feature_stats.py
rm tests/ml/test_model_runtime.py tests/ml/test_score_mapping.py
rm tests/walkforward/test_*.py tests/sanity/test_cases.py
```

### **2. Validate Alpha v1**
```bash
python tools/train_alpha_v1.py --symbols SPY,TSLA
python tools/validate_alpha.py reports/alpha_eval.json
python scripts/walkforward_alpha_v1.py --symbols SPY --train-len 50 --test-len 20 --stride 10 --warmup 10
python scripts/compare_walkforward.py --symbols SPY
```

### **3. Validate Core System**
```bash
python scripts/paper_runner.py --symbols SPY --poll-sec 1 --steps 2
python scripts/canary_runner.py --symbols SPY --poll-sec 1 --steps 2
```

## 📊 **Current Status**
- ✅ **Alpha v1**: Working (training, validation, walkforward, comparison)
- ✅ **Documentation**: Complete (audit docs, cleanup plan, context file)
- ✅ **File Classification**: Complete (45 safe to remove, 35 keep)
- 🚀 **Ready**: Execute Phase 1 cleanup

## 📚 **Key Documents**
- `CLEANUP_SESSION_CONTEXT.md` - Full context and plan
- `docs/FINAL_CLEANUP_PLAN.md` - Detailed execution plan
- `docs/SYSTEM_AUDIT_DOCUMENTATION.md` - Complete analysis
- `docs/AUDIT_SUMMARY.md` - Quick summary

## 🎯 **Success Criteria**
- ✅ 45 old ML files removed
- ✅ Alpha v1 still works
- ✅ Core system still works
- ✅ CI/CD still works
- ✅ Documentation updated

**Next Action**: Execute the removal commands above, then validate everything works!
