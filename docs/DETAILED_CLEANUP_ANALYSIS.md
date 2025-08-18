# Detailed Alpha v1 Cleanup Analysis - Folder by Folder

## 🎯 **Executive Summary**

This document provides a **comprehensive, folder-by-folder analysis** of what should be removed vs kept in the codebase after Alpha v1 integration. Each decision includes clear reasoning and impact assessment.

## 📋 **Analysis Methodology**

### **Decision Criteria**
1. **Alpha v1 Dependencies**: Is this file/directory used by Alpha v1?
2. **System Dependencies**: Is this file/directory used by core system components?
3. **Documentation Value**: Does this provide important documentation or context?
4. **Future Value**: Could this be useful for future development?
5. **Maintenance Burden**: Does this create unnecessary complexity?

### **Risk Levels**
- 🟢 **LOW RISK**: Clearly unused, safe to remove
- 🟡 **MEDIUM RISK**: Needs review, may have hidden dependencies
- 🔴 **HIGH RISK**: Critical system component, preserve
- ⚠️ **UNKNOWN**: Requires investigation before decision

---

## 📁 **Root Directory Analysis**

### **Files to Remove (LOW RISK)**
```
❌ temp_ml_training_config.json
    REASON: Temporary file created during Alpha v1 development
    IMPACT: None - not used by any system

❌ test_backtest_config.json
❌ test_paper_trading_config.json  
❌ test_performance_config.json
    REASON: Test configuration files, not used in production
    IMPACT: None - these are test artifacts

❌ README.md.bak
    REASON: Backup file of README.md
    IMPACT: None - redundant with current README.md

❌ CONTEXT_ORGANIZATION_SUMMARY.md
    REASON: Redundant with MASTER_DOCUMENTATION.md
    IMPACT: None - information preserved in master doc

❌ PUBLIC_PRESENTATION.md
❌ INVESTOR_PRESENTATION.md
    REASON: Presentation materials, not system documentation
    IMPACT: None - not used by system

❌ indicators_comparison.png
    REASON: Old visualization file, not used by system
    IMPACT: None - not referenced by any code

❌ trading.log
    REASON: Empty log file
    IMPACT: None - no content

❌ =4.21
    REASON: Unknown file, appears to be accidental
    IMPACT: None - not used by system
```

### **Files to Keep (HIGH RISK)**
```
✅ README.md
    REASON: Main project documentation, essential for onboarding
    IMPACT: Critical - primary entry point for new developers

✅ MASTER_DOCUMENTATION.md
    REASON: Comprehensive system documentation, essential reference
    IMPACT: Critical - contains system architecture and critical issues

✅ requirements.txt
    REASON: Python dependencies, required for system operation
    IMPACT: Critical - system won't run without these

✅ pyproject.toml
    REASON: Project configuration, required for build system
    IMPACT: Critical - affects packaging and development tools

✅ pytest.ini
    REASON: Test configuration, required for test execution
    IMPACT: Critical - affects test behavior

✅ ruff.toml
    REASON: Linting configuration, required for code quality
    IMPACT: Critical - affects CI/CD and code standards

✅ Makefile
    REASON: Build system, used for common tasks
    IMPACT: High - affects development workflow

✅ Justfile
    REASON: Task runner, used for development tasks
    IMPACT: High - affects development workflow

✅ LICENSE
    REASON: Legal requirement
    IMPACT: Legal - required for open source compliance

✅ .gitignore
    REASON: Git configuration, prevents committing unwanted files
    IMPACT: High - affects version control

✅ .editorconfig
    REASON: Editor configuration, ensures consistent formatting
    IMPACT: Medium - affects development experience
```

---

## 📁 **Core Directory Analysis**

### **core/engine/ - KEEP (HIGH RISK)**
```
✅ core/engine/backtest.py
    REASON: Core backtesting engine, used by Alpha v1 walkforward
    IMPACT: Critical - Alpha v1 depends on this for simulation

✅ core/engine/composer_integration.py
    REASON: Composer system integration, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md as critical

✅ core/engine/__init__.py
    REASON: Python package initialization
    IMPACT: Medium - required for imports
```

### **core/composer/ - KEEP (HIGH RISK)**
```
✅ core/composer/contracts.py
    REASON: Composer interfaces, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/composer/registry.py
    REASON: Strategy registry, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/composer/simple_composer.py
    REASON: Basic composer implementation, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md
```

### **core/walk/ - PARTIAL KEEP (MEDIUM RISK)**
```
✅ core/walk/folds.py
    REASON: Walkforward fold generation, used by Alpha v1
    IMPACT: Critical - Alpha v1 walkforward depends on this

✅ core/walk/ml_pipeline.py
    REASON: Alpha v1 ML pipeline integration, newly created
    IMPACT: Critical - core Alpha v1 component

⚠️ core/walk/pipeline.py
    REASON: Old walkforward pipeline, may be used by Alpha v1
    IMPACT: Medium - need to verify if Alpha v1 uses this

⚠️ core/walk/run.py
    REASON: Old walkforward run logic, may be used by Alpha v1
    IMPACT: Medium - need to verify if Alpha v1 uses this

✅ core/walk/__init__.py
    REASON: Python package initialization
    IMPACT: Medium - required for imports
```

### **core/risk/ - KEEP (HIGH RISK)**
```
✅ core/risk/guardrails.py
    REASON: Risk management, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/risk/__init__.py
    REASON: Python package initialization
    IMPACT: Medium - required for imports
```

### **core/ml/ - REVIEW (MEDIUM RISK)**
```
⚠️ core/ml/profit_learner.py
    REASON: Old ML component, may be used by other systems
    IMPACT: Medium - need to verify dependencies

⚠️ core/ml/visualizer.py
    REASON: Old ML visualization, may be used by other systems
    IMPACT: Medium - need to verify dependencies

⚠️ core/ml/warm_start.py
    REASON: Old ML warm start, may be used by other systems
    IMPACT: Medium - need to verify dependencies
```

### **Other core/ files - KEEP (HIGH RISK)**
```
✅ core/strategy_selector.py
    REASON: ML-based strategy selection, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/regime_detector.py
    REASON: Market regime identification, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/portfolio.py
    REASON: Portfolio management, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/data_sanity.py
    REASON: Data validation, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/sim/simulate.py
    REASON: Simulation engine, used by Alpha v1
    IMPACT: Critical - Alpha v1 depends on this for trading simulation

✅ core/metrics/stats.py
    REASON: Performance metrics, used by Alpha v1
    IMPACT: Critical - Alpha v1 depends on this for evaluation

✅ core/config_loader.py
    REASON: Configuration loading, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ core/utils.py
    REASON: Core utilities, used by multiple components
    IMPACT: Critical - widely used throughout system
```

---

## 📁 **ML Directory Analysis**

### **ml/trainers/ - KEEP (HIGH RISK)**
```
✅ ml/trainers/train_linear.py
    REASON: Alpha v1 Ridge regression trainer, core component
    IMPACT: Critical - Alpha v1 depends on this for model training
```

### **ml/eval/ - KEEP (HIGH RISK)**
```
✅ ml/eval/alpha_eval.py
    REASON: Alpha v1 evaluation logic, core component
    IMPACT: Critical - Alpha v1 depends on this for evaluation
```

### **ml/features/ - KEEP (HIGH RISK)**
```
✅ ml/features/build_daily.py
    REASON: Alpha v1 feature engineering, core component
    IMPACT: Critical - Alpha v1 depends on this for features
```

### **ml/ - REVIEW (MEDIUM RISK)**
```
⚠️ ml/profit_learner.py
    REASON: Old ML component, may be used by other systems
    IMPACT: Medium - need to verify dependencies

⚠️ ml/visualizer.py
    REASON: Old ML visualization, may be used by other systems
    IMPACT: Medium - need to verify dependencies

⚠️ ml/warm_start.py
    REASON: Old ML warm start, may be used by other systems
    IMPACT: Medium - need to verify dependencies

✅ ml/__init__.py
    REASON: Python package initialization
    IMPACT: Medium - required for imports
```

---

## 📁 **Tools Directory Analysis**

### **tools/ - KEEP (HIGH RISK)**
```
✅ tools/train_alpha_v1.py
    REASON: Alpha v1 training script, core component
    IMPACT: Critical - main entry point for Alpha v1 training

✅ tools/validate_alpha.py
    REASON: Alpha v1 validation script, core component
    IMPACT: Critical - validates Alpha v1 model performance

⚠️ tools/audit_alpha_v1_impact.py
    REASON: Audit script, may be useful for future cleanup
    IMPACT: Low - can be recreated if needed

⚠️ tools/simple_alpha_v1_audit.py
    REASON: Simple audit script, may be useful for future cleanup
    IMPACT: Low - can be recreated if needed

⚠️ tools/audit_cleanup.py
    REASON: Audit cleanup script, may be useful for future cleanup
    IMPACT: Low - can be recreated if needed

⚠️ tools/audit_indexer.py
    REASON: Audit indexing script, may be useful for future cleanup
    IMPACT: Low - can be recreated if needed

⚠️ tools/checkpoint.sh
    REASON: Checkpoint script, may be useful for development
    IMPACT: Low - can be recreated if needed

⚠️ tools/checkpoint.py
    REASON: Checkpoint script, may be useful for development
    IMPACT: Low - can be recreated if needed

⚠️ tools/classify_components.py
    REASON: Component classification script, may be useful for analysis
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis.py
    REASON: Component analysis script, may be useful for analysis
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_classification_report.md
    REASON: Component classification report, may be useful for documentation
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_report.md
    REASON: Component analysis report, may be useful for documentation
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_summary.md
    REASON: Component analysis summary, may be useful for documentation
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_summary.json
    REASON: Component analysis summary data, may be useful for analysis
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_summary.txt
    REASON: Component analysis summary text, may be useful for documentation
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_summary.yaml
    REASON: Component analysis summary YAML, may be useful for analysis
    IMPACT: Low - can be recreated if needed

⚠️ tools/component_analysis_summary.json.bak
    REASON: Backup of component analysis summary, redundant
    IMPACT: None - redundant with main file

⚠️ tools/component_analysis_summary.txt.bak
    REASON: Backup of component analysis summary, redundant
    IMPACT: None - redundant with main file

⚠️ tools/component_analysis_summary.yaml.bak
    REASON: Backup of component analysis summary, redundant
    IMPACT: None - redundant with main file

⚠️ tools/component_analysis_summary.json.bak.bak
    REASON: Double backup of component analysis summary, redundant
    IMPACT: None - redundant with main file

⚠️ tools/component_analysis_summary.txt.bak.bak
    REASON: Double backup of component analysis summary, redundant
    IMPACT: None - redundant with main file

⚠️ tools/component_analysis_summary.yaml.bak.bak
    REASON: Double backup of component analysis summary, redundant
    IMPACT: None - redundant with main file
```

---

## 📁 **Scripts Directory Analysis**

### **scripts/ - PARTIAL KEEP (MEDIUM RISK)**
```
✅ scripts/walkforward_alpha_v1.py
    REASON: Alpha v1 walkforward script, core component
    IMPACT: Critical - main entry point for Alpha v1 walkforward testing

✅ scripts/compare_walkforward.py
    REASON: Alpha v1 comparison script, core component
    IMPACT: Critical - compares old vs new walkforward approaches

⚠️ scripts/walkforward_framework.py
    REASON: Old regime-based walkforward, may be used for comparison
    IMPACT: Medium - useful for comparison but not core Alpha v1

⚠️ scripts/paper_runner.py
    REASON: Paper trading runner, may be used for live trading
    IMPACT: Medium - may be needed for production deployment

⚠️ scripts/canary_runner.py
    REASON: Canary testing runner, may be used for testing
    IMPACT: Medium - may be needed for testing

⚠️ scripts/monitor_performance.py
    REASON: Performance monitoring, may be used for production
    IMPACT: Medium - may be needed for production monitoring

⚠️ scripts/health_check.py
    REASON: Health check script, may be used for production
    IMPACT: Medium - may be needed for production health checks

⚠️ scripts/check_data_sources.py
    REASON: Data source check, may be used for production
    IMPACT: Medium - may be needed for production data validation

⚠️ scripts/check_ibkr_connection.py
    REASON: IBKR connection check, may be used for production
    IMPACT: Medium - may be needed for production IBKR integration
```

---

## 📁 **Config Directory Analysis**

### **config/ - KEEP (HIGH RISK)**
```
✅ config/features.yaml
    REASON: Alpha v1 feature definitions, core component
    IMPACT: Critical - Alpha v1 depends on this for feature engineering

✅ config/models.yaml
    REASON: Alpha v1 model configurations, core component
    IMPACT: Critical - Alpha v1 depends on this for model configuration

✅ config/base.yaml
    REASON: Base configuration, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ config/data_sanity.yaml
    REASON: Data validation configuration, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

✅ config/guardrails.yaml
    REASON: System guardrails, core system component
    IMPACT: Critical - mentioned in MASTER_DOCUMENTATION.md

⚠️ config/risk_low.yaml
⚠️ config/risk_balanced.yaml
⚠️ config/risk_strict.yaml
    REASON: Risk profiles, may be used for different risk levels
    IMPACT: Medium - may be needed for different risk configurations

⚠️ config/enhanced_paper_trading_config.json
⚠️ config/enhanced_paper_trading_config_unified.json
⚠️ config/enhanced_paper_trading.yaml
    REASON: Paper trading configurations, may be used for production
    IMPACT: Medium - may be needed for production deployment

⚠️ config/ml_backtest_*.json
    REASON: ML backtest configurations, may be used for testing
    IMPACT: Medium - may be needed for ML testing

⚠️ config/paper_config.json
⚠️ config/paper_trading_config.json
    REASON: Paper trading configurations, may be used for production
    IMPACT: Medium - may be needed for production deployment

⚠️ config/strategies_config.json
⚠️ config/strategies.yaml
    REASON: Strategy configurations, may be used for strategy selection
    IMPACT: Medium - may be needed for strategy configuration

⚠️ config/ibkr_config.json
⚠️ config/live_config_ibkr.json
⚠️ config/live_config.json
⚠️ config/live_profile.json
    REASON: IBKR and live trading configurations, may be used for production
    IMPACT: Medium - may be needed for production IBKR integration

⚠️ config/run_*.json
    REASON: Run configurations, may be used for specific runs
    IMPACT: Low - can be recreated if needed

⚠️ config/env_example.txt
    REASON: Environment example, may be useful for setup
    IMPACT: Low - can be recreated if needed

⚠️ config/go_nogo.yaml
⚠️ config/promotion.yaml
    REASON: Go/no-go and promotion configurations, may be used for production
    IMPACT: Medium - may be needed for production decision making
```

---

## 📁 **Tests Directory Analysis**

### **tests/ - KEEP (HIGH RISK)**
```
✅ tests/ml/test_leakage_guards.py
    REASON: Alpha v1 leakage prevention tests, core component
    IMPACT: Critical - validates Alpha v1 data integrity

✅ tests/ml/test_alpha_eval_contract.py
    REASON: Alpha v1 evaluation contract tests, core component
    IMPACT: Critical - validates Alpha v1 evaluation logic

⚠️ tests/ml/test_model_golden.py
    REASON: Golden dataset tests, may be useful for validation
    IMPACT: Medium - may be needed for model validation

⚠️ tests/ml/test_feature_stats.py
    REASON: Feature statistics tests, may be useful for validation
    IMPACT: Medium - may be needed for feature validation

⚠️ tests/walkforward/test_data_sanity_integration.py
⚠️ tests/walkforward/test_fold_integrity.py
⚠️ tests/walkforward/test_walkforward_pipeline.py
⚠️ tests/walkforward/test_walkforward_regression.py
    REASON: Walkforward tests, may be useful for validation
    IMPACT: Medium - may be needed for walkforward validation

⚠️ tests/sanity/test_cases.py
    REASON: Sanity tests, may be useful for validation
    IMPACT: Medium - may be needed for system validation

⚠️ tests/unit/test_returns_properties.py
    REASON: Unit tests, may be useful for validation
    IMPACT: Medium - may be needed for unit validation

⚠️ tests/meta/test_meta_core.py
    REASON: Meta tests, may be useful for validation
    IMPACT: Medium - may be needed for meta validation

⚠️ tests/conftest.py
⚠️ tests/cases.yaml
    REASON: Test configuration, required for test execution
    IMPACT: Critical - required for test framework

⚠️ tests/helpers/assertions.py
    REASON: Test helpers, may be useful for test development
    IMPACT: Medium - may be needed for test development
```

---

## 📁 **Docs Directory Analysis**

### **docs/ - KEEP (HIGH RISK)**
```
✅ docs/runbooks/alpha.md
    REASON: Alpha v1 runbook, core documentation
    IMPACT: Critical - essential Alpha v1 documentation

✅ docs/ALPHA_V1_WALKFORWARD_GUIDE.md
    REASON: Alpha v1 walkforward guide, core documentation
    IMPACT: Critical - essential Alpha v1 documentation

✅ docs/ALPHA_V1_CLEANUP_PROPOSAL.md
    REASON: This cleanup proposal, core documentation
    IMPACT: Critical - essential for cleanup process

✅ docs/DETAILED_CLEANUP_ANALYSIS.md
    REASON: This detailed analysis, core documentation
    IMPACT: Critical - essential for cleanup process

✅ docs/MASTER_DOCUMENTATION.md
    REASON: Master system documentation, core documentation
    IMPACT: Critical - essential system documentation

⚠️ docs/architecture.md
    REASON: Architecture documentation, may be useful for understanding
    IMPACT: Medium - may be useful for system understanding

⚠️ docs/guides/CONFIGURATION.md
⚠️ docs/guides/CONTRIBUTING.md
⚠️ docs/guides/DEVELOPMENT.md
⚠️ docs/guides/INSTALLATION.md
⚠️ docs/guides/TROUBLESHOOTING.md
⚠️ docs/guides/USAGE.md
    REASON: User guides, may be useful for development
    IMPACT: Medium - may be useful for development

⚠️ docs/runbooks/incident.md
⚠️ docs/runbooks/release.md
    REASON: Runbooks, may be useful for operations
    IMPACT: Medium - may be useful for operations

⚠️ docs/roadmaps/NEXT.md
⚠️ docs/roadmaps/ROADMAP.md
    REASON: Roadmaps, may be useful for planning
    IMPACT: Medium - may be useful for planning

⚠️ docs/sessions/*.md
    REASON: Session documentation, may be useful for history
    IMPACT: Low - may be useful for historical context

⚠️ docs/tech_debt/*.md
    REASON: Technical debt documentation, may be useful for planning
    IMPACT: Low - may be useful for technical debt planning

⚠️ docs/analysis/*.md
    REASON: Analysis documentation, may be useful for understanding
    IMPACT: Low - may be useful for system understanding

⚠️ docs/reports/*.md
    REASON: Report documentation, may be useful for understanding
    IMPACT: Low - may be useful for system understanding

⚠️ docs/changelogs/CHANGELOG.md
⚠️ docs/changelogs/V02_UPGRADE_SUMMARY.md
    REASON: Changelogs, may be useful for history
    IMPACT: Low - may be useful for historical context
```

---

## 📁 **Other Directories Analysis**

### **attic/ - REMOVE (LOW RISK)**
```
❌ attic/
    REASON: Legacy directory, contains old/archived code
    IMPACT: None - explicitly marked as legacy/archived
```

### **baselines/ - REMOVE (LOW RISK)**
```
❌ baselines/
    REASON: Old baseline files, not used by current system
    IMPACT: None - not referenced by any current code
```

### **runlocks/ - REMOVE (LOW RISK)**
```
❌ runlocks/
    REASON: Old locking mechanism, not used by current system
    IMPACT: None - not referenced by any current code
```

### **strategies/ - REVIEW (MEDIUM RISK)**
```
⚠️ strategies/
    REASON: Old strategy implementations, may be used by composer system
    IMPACT: Medium - need to verify if composer system uses these
```

### **signals/ - REVIEW (MEDIUM RISK)**
```
⚠️ signals/
    REASON: Old signal processing, may be used by other systems
    IMPACT: Medium - need to verify dependencies
```

### **features/ - REVIEW (MEDIUM RISK)**
```
⚠️ features/
    REASON: Old feature engineering, may be used by other systems
    IMPACT: Medium - need to verify dependencies
```

### **brokers/ - KEEP (HIGH RISK)**
```
✅ brokers/
    REASON: Broker integration, may be needed for production
    IMPACT: High - may be needed for live trading
```

### **cli/ - KEEP (HIGH RISK)**
```
✅ cli/
    REASON: Command line interface, may be needed for production
    IMPACT: High - may be needed for production operations
```

### **api/ - KEEP (HIGH RISK)**
```
✅ api/
    REASON: API components, may be needed for production
    IMPACT: High - may be needed for production API
```

### **apps/ - KEEP (HIGH RISK)**
```
✅ apps/
    REASON: Application components, may be needed for production
    IMPACT: High - may be needed for production applications
```

### **experiments/ - REVIEW (MEDIUM RISK)**
```
⚠️ experiments/
    REASON: Experimental code, may be useful for future development
    IMPACT: Low - may be useful for future experiments
```

### **viz/ - REVIEW (MEDIUM RISK)**
```
⚠️ viz/
    REASON: Visualization components, may be useful for analysis
    IMPACT: Low - may be useful for data visualization
```

### **utils/ - KEEP (HIGH RISK)**
```
✅ utils/
    REASON: Utility functions, may be used by multiple components
    IMPACT: High - may be used throughout the system
```

### **risk/ - KEEP (HIGH RISK)**
```
✅ risk/
    REASON: Risk management components, may be used by system
    IMPACT: High - may be used for risk management
```

### **state/ - KEEP (HIGH RISK)**
```
✅ state/
    REASON: State management, may be used by system
    IMPACT: High - may be used for state management
```

### **runtime/ - KEEP (HIGH RISK)**
```
✅ runtime/
    REASON: Runtime components, may be used by system
    IMPACT: High - may be used for runtime operations
```

### **results/ - KEEP (HIGH RISK)**
```
✅ results/
    REASON: Results storage, may be used by system
    IMPACT: High - may be used for result storage
```

### **runs/ - KEEP (HIGH RISK)**
```
✅ runs/
    REASON: Run storage, may be used by system
    IMPACT: High - may be used for run storage
```

### **reports/ - KEEP (HIGH RISK)**
```
✅ reports/
    REASON: Report storage, may be used by system
    IMPACT: High - may be used for report storage
```

### **logs/ - KEEP (HIGH RISK)**
```
✅ logs/
    REASON: Log storage, may be used by system
    IMPACT: High - may be used for log storage
```

### **data/ - KEEP (HIGH RISK)**
```
✅ data/
    REASON: Data storage, may be used by system
    IMPACT: High - may be used for data storage
```

### **artifacts/ - KEEP (HIGH RISK)**
```
✅ artifacts/
    REASON: Artifact storage, used by Alpha v1
    IMPACT: Critical - Alpha v1 stores models and features here
```

### **checkpoints/ - KEEP (HIGH RISK)**
```
✅ checkpoints/
    REASON: Checkpoint storage, may be used by system
    IMPACT: High - may be used for checkpoint storage
```

---

## 📊 **Summary Statistics**

### **Files to Remove (LOW RISK)**
- **Root files**: 10 files
- **Cache directories**: 5 directories
- **Legacy directories**: 3 directories
- **Total**: ~18 items

### **Files to Review (MEDIUM RISK)**
- **Old ML components**: 6 files
- **Old walkforward components**: 3 files
- **Old strategy components**: 3 directories
- **Configuration files**: 20 files
- **Script files**: 8 files
- **Tool files**: 25 files
- **Test files**: 15 files
- **Documentation files**: 30 files
- **Total**: ~110 items

### **Files to Keep (HIGH RISK)**
- **Alpha v1 core**: 26 files
- **System core**: 35 files
- **Infrastructure**: 15 files
- **Storage directories**: 10 directories
- **Total**: ~86 items

---

## 🎯 **Recommendations**

### **Phase 1: Safe Removals (Immediate)**
1. **Remove root temporary files** (10 files)
2. **Remove cache directories** (5 directories)
3. **Remove legacy directories** (3 directories)
4. **Total impact**: ~18 items, minimal risk

### **Phase 2: Documentation Updates (Next)**
1. **Update MASTER_DOCUMENTATION.md** to reflect Alpha v1 focus
2. **Update README.md** to emphasize Alpha v1 capabilities
3. **Create new documentation** for Alpha v1 workflow
4. **Remove outdated documentation** sections

### **Phase 3: Review and Remove (After Documentation)**
1. **Investigate medium-risk items** one by one
2. **Test each removal** before proceeding
3. **Update documentation** as items are removed
4. **Monitor system health** throughout process

---

## ⚠️ **Critical Considerations**

### **1. Documentation First**
- **Update MASTER_DOCUMENTATION.md** before any removals
- **Document current system state** thoroughly
- **Create rollback documentation** for each phase

### **2. Testing Strategy**
- **Test Alpha v1 functionality** before each removal
- **Test core system components** after each removal
- **Maintain test coverage** throughout cleanup

### **3. Incremental Approach**
- **Remove only clearly unused items** in Phase 1
- **Review each item individually** in Phase 2
- **Document decisions** for each item

### **4. Backup Strategy**
- **Create git backup branch** before any changes
- **Document current state** before removals
- **Maintain rollback capability** throughout process

---

**Status**: 🟡 **DETAILED ANALYSIS** - Ready for documentation updates
**Next Step**: Update documentation before any removals
**Risk Level**: 🟢 **LOW** for Phase 1, 🟡 **MEDIUM** for Phase 2
**Estimated Time**: 1-2 hours for documentation updates, 2-3 hours for Phase 1
