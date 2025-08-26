# FOLDER ORGANIZATION PLAN - REDUCE IDE BLOAT

## **🎯 GOAL**
Organize 102 scripts into logical folders to reduce IDE cognitive load and improve navigation.

## **📊 CURRENT STATE**
- **102 Python scripts** in single `scripts/` directory
- **IDE bloat** - hard to find specific scripts
- **Cognitive overhead** - too many files in one view
- **Import confusion** - unclear dependencies

## **🏗️ PROPOSED FOLDER STRUCTURE**

```
scripts/
├── core/                    # Core execution scripts (8 files)
│   ├── paper_runner.py
│   ├── paper_broker.py
│   ├── canary_runner.py
│   ├── e2d.py
│   ├── runner.py
│   ├── backtest.py
│   ├── easy_trade.py
│   └── e2e_pipeline.py
│
├── training/                # Training scripts (6 files)
│   ├── train.py
│   ├── train_linear.py
│   ├── train_crypto.py
│   ├── live_trading_workflow.py
│   ├── demo_onnx_export.py
│   └── demo_asset_routing.py
│
├── walkforward/             # Walkforward scripts (6 files)
│   ├── multi_walkforward_report.py
│   ├── walk_core.py
│   ├── walkforward_framework_compat.py
│   ├── walkforward_with_composer.py
│   ├── walk_forward.py
│   └── e2e_sanity_check.py
│
├── validation/              # Validation & testing scripts (8 files)
│   ├── validate_run_report.py
│   ├── gate_e2d.py
│   ├── go_nogo.py
│   ├── canary_datasanity.py
│   ├── check_datasanity.py
│   ├── check_non_positive_prices.py
│   ├── falsify_data_sanity.py
│   └── perf_gate.py
│
├── data/                    # Data acquisition scripts (4 files)
│   ├── fetch_yfinance.py
│   ├── fetch_corporate_actions.py
│   ├── market_scanner.py
│   └── debug_lookahead.py
│
├── analysis/                # Analysis & evaluation scripts (6 files)
│   ├── analyze_metrics.py
│   ├── eval_compare.py
│   ├── detect_lookahead_contamination.py
│   ├── ablate.py
│   ├── bench_infer.py
│   └── aurora_audit.py
│
├── experiments/             # Experiment scripts (4 files)
│   ├── config_sweep.py
│   ├── experiment_runner.py
│   ├── quick_sweep_demo.py
│   └── ic_analysis.py
│
├── tools/                   # Utility tools (6 files)
│   ├── consolidation_report.py
│   ├── test_adapter_isolation.py
│   ├── test_asset_routing.py
│   ├── train_alpha_v1.py
│   └── setup_portfolio_demo.py
│
└── archive/                 # Everything else (54 files)
    ├── [all remaining scripts]
    └── [organized by type if needed]
```

## **📈 BENEFITS**

### **IDE Performance**
- **Before:** 102 files in single directory
- **After:** 8 organized folders + archive
- **Navigation:** 90% reduction in file list size
- **Search:** Faster file finding

### **Cognitive Load**
- **Before:** Overwhelming file list
- **After:** Clear logical organization
- **Discovery:** Easy to find related scripts
- **Maintenance:** Clear ownership boundaries

### **Development Workflow**
- **Core scripts:** Easy access to main execution
- **Training:** All training in one place
- **Validation:** All testing/validation together
- **Data:** All data acquisition together

## **🔧 IMPLEMENTATION PLAN**

### **Phase 1: Create Folder Structure**
```bash
mkdir -p scripts/{core,training,walkforward,validation,data,analysis,experiments,tools,archive}
```

### **Phase 2: Move Critical Scripts**
Move the 38 most important scripts to organized folders:

**Core (8 files):**
```bash
mv scripts/{paper_runner.py,paper_broker.py,canary_runner.py,e2d.py,runner.py,backtest.py,easy_trade.py,e2e_pipeline.py} scripts/core/
```

**Training (6 files):**
```bash
mv scripts/{train.py,train_linear.py,train_crypto.py,live_trading_workflow.py,demo_onnx_export.py,demo_asset_routing.py} scripts/training/
```

**Walkforward (6 files):**
```bash
mv scripts/{multi_walkforward_report.py,walk_core.py,walkforward_framework_compat.py,walkforward_with_composer.py,walk_forward.py,e2e_sanity_check.py} scripts/walkforward/
```

**Validation (8 files):**
```bash
mv scripts/{validate_run_report.py,gate_e2d.py,go_nogo.py,canary_datasanity.py,check_datasanity.py,check_non_positive_prices.py,falsify_data_sanity.py,perf_gate.py} scripts/validation/
```

**Data (4 files):**
```bash
mv scripts/{fetch_yfinance.py,fetch_corporate_actions.py,market_scanner.py,debug_lookahead.py} scripts/data/
```

**Analysis (6 files):**
```bash
mv scripts/{analyze_metrics.py,eval_compare.py,detect_lookahead_contamination.py,ablate.py,bench_infer.py,aurora_audit.py} scripts/analysis/
```

**Experiments (4 files):**
```bash
mv scripts/{config_sweep.py,experiment_runner.py,quick_sweep_demo.py,ic_analysis.py} scripts/experiments/
```

**Tools (6 files):**
```bash
mv scripts/{consolidation_report.py,test_adapter_isolation.py,test_asset_routing.py,train_alpha_v1.py,setup_portfolio_demo.py} scripts/tools/
```

### **Phase 3: Move Remaining Scripts**
```bash
mv scripts/*.py scripts/archive/
```

### **Phase 4: Create Symlinks for Backward Compatibility**
```bash
# Create symlinks for critical scripts that might be imported
ln -s core/paper_runner.py scripts/paper_runner.py
ln -s core/paper_broker.py scripts/paper_broker.py
ln -s walkforward/multi_walkforward_report.py scripts/multi_walkforward_report.py
ln -s training/train.py scripts/train.py
ln -s validation/validate_run_report.py scripts/validate_run_report.py
```

### **Phase 5: Update Makefile**
Update Makefile to use new paths:
```makefile
# Update paths in Makefile
smoke: python scripts/walkforward/multi_walkforward_report.py --smoke ...
paper: python scripts/core/paper_runner.py ...
train: python scripts/training/train.py ...
```

## **🚨 BACKWARD COMPATIBILITY**

### **Import Safety**
- Create symlinks for critical scripts
- Update any hardcoded imports
- Test all Makefile targets

### **CI Safety**
- Update CI workflows to use new paths
- Test all CI jobs still work
- Update documentation

## **📋 EXECUTION CHECKLIST**

- [ ] Create folder structure
- [ ] Move critical scripts (38 files)
- [ ] Move remaining scripts to archive (64 files)
- [ ] Create symlinks for backward compatibility
- [ ] Update Makefile paths
- [ ] Test all Makefile targets
- [ ] Test CI workflows
- [ ] Update documentation
- [ ] Verify IDE navigation improvement

## **🎯 SUCCESS METRICS**

- **IDE Performance:** 90% reduction in file list size
- **Navigation:** Find any script in <3 clicks
- **Organization:** Clear logical grouping
- **Maintenance:** Easier to understand codebase structure
- **Backward Compatibility:** All existing workflows still work

## **🚀 READY TO IMPLEMENT**

This organization will dramatically reduce IDE bloat while maintaining all functionality. The archive folder keeps everything accessible while the organized folders make the most important scripts easy to find.

**Ready to proceed with folder organization?**
