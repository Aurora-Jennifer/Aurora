# Codebase Cleanup Proposal (DRY-RUN)

**Date**: 2025-08-18  
**Session**: Alpha v1 Implementation  
**Status**: DRY-RUN - No changes made yet

## Executive Summary

After implementing the Alpha v1 ML pipeline, we need to clean up the codebase to remove deprecated files, consolidate duplicate code, and ensure full compliance with the Aurora ruleset. This proposal outlines a systematic cleanup approach with minimal risk.

## Current State Analysis

### ✅ **What's Working Well**
- **Alpha v1 Pipeline**: Complete ML workflow implemented and tested
- **Core Engine**: Backtesting and paper trading engines functional
- **Testing**: Comprehensive test coverage for new ML components
- **Documentation**: Complete runbook and audit trail created
- **Configuration**: All parameters externalized to YAML files

### ⚠️ **Areas Needing Cleanup**
- **Deprecated Files**: Multiple unused files in `attic/` directory
- **Documentation Drift**: Some docs may not reflect current capabilities
- **Configuration Bloat**: Some config files may be unused
- **Code Quality**: Some debug statements in production code

## Cleanup Proposal (Phase-by-Phase)

### **Phase 1: Safe Removals** (Risk: Very Low)

#### 1.1 Remove Deprecated Test Files
**Location**: `attic/root_tests/`
**Files to Remove**:
- `test_walkforward_ml.py` (16KB, 492 lines)
- `test_data_integrity.py` (19KB, 575 lines)
- `test_ml_backtesting.py` (14KB, 464 lines)
- `test_paper_engine.py` (7.6KB, 212 lines)
- `final_ml_validation.py` (16KB, 507 lines)
- `test_core_functionality.py` (27KB, 766 lines)
- `test_component_validation.py` (14KB, 382 lines)
- `test_enhanced_trading_system.py` (8.2KB, 267 lines)
- `test_meaningful_validation.py` (16KB, 476 lines)

**Justification**: 
- Not referenced by pytest (which uses `tests/` directory)
- No active imports found in codebase
- Already archived in `attic/` directory

#### 1.2 Remove Empty Directories
**Location**: `attic/scripts/`
**Action**: Remove directory (contains only `__pycache__/`)

#### 1.3 Remove Legacy Documentation
**Location**: `attic/docs/`
**Files to Remove**:
- `HONEST_SYSTEM_ASSESSMENT.md`
- `FINAL_VALIDATION_REPORT.md`
- `COMPREHENSIVE_VALIDATION_REPORT.md`
- `ENHANCED_TRADING_SYSTEM_SUMMARY.md`
- `WALKFORWARD_AND_DATA_FIXES_SUMMARY.md`
- `UNIFIED_CONTEXT_SUMMARY.md`
- `UNIFIED_CONTEXT.md`
- `LEAN_CODEBASE_SUMMARY.md`
- `unification_plan.md`
- `END_TO_END_TESTING_GUIDE.md`

**Justification**: 
- Superseded by newer documentation
- No active references found
- Current docs are in `docs/` directory

#### 1.4 Remove Legacy Configs
**Location**: `attic/config/`, `attic/legacy_configs/`
**Action**: Remove old configuration files

**Justification**: 
- Superseded by current config structure
- No active references found

### **Phase 2: Documentation Updates** (Risk: Very Low)

#### 2.1 Update Core Documentation
**Files to Update**:
- `README.md` ✅ (Already updated with Alpha v1 info)
- `MASTER_DOCUMENTATION.md` - Add ML capabilities
- `docs/changelogs/CHANGELOG.md` - Document Alpha v1 implementation
- `docs/guides/` - Add Alpha v1 usage examples

#### 2.2 Create New Guides
**New Files to Create**:
- `docs/guides/ML_MODEL_DEVELOPMENT.md` - How to iterate on Alpha v1
- `docs/guides/FEATURE_ENGINEERING.md` - How to add new features
- `docs/guides/MODEL_EVALUATION.md` - How to assess model performance

### **Phase 3: Configuration Cleanup** (Risk: Medium)

#### 3.1 Audit Config Directory
**Location**: `config/`
**Action**: Review and remove unused config files

**Files to Review**:
- `test_*.json` files in root directory
- Old backtest configurations
- Unused profile configurations

#### 3.2 Standardize Environment Variables
**Action**: Ensure consistent naming and documentation

### **Phase 4: Code Quality** (Risk: Low)

#### 4.1 Remove Debug Statements
**Action**: Remove debug statements from production code

**Files to Review**:
- `core/engine/backtest.py` - Multiple debug statements
- `core/portfolio.py` - Debug logging
- `scripts/walkforward_framework.py` - Debug statements

#### 4.2 Run Linting Cleanup
**Action**: Run ruff to remove unused imports and fix formatting

#### 4.3 Add Missing Type Hints
**Action**: Add type hints to functions missing them

## Risk Assessment

### **Phase 1: Safe Removals**
- **Risk Level**: Very Low
- **Impact**: None (files already archived)
- **Benefit**: Cleaner codebase, reduced confusion
- **Validation**: No active imports found

### **Phase 2: Documentation Updates**
- **Risk Level**: Very Low
- **Impact**: None (documentation only)
- **Benefit**: Better user experience, clearer guidance
- **Validation**: No code changes

### **Phase 3: Configuration Cleanup**
- **Risk Level**: Medium
- **Impact**: Potential configuration changes
- **Benefit**: Cleaner configuration, reduced confusion
- **Validation**: Need to verify no active references

### **Phase 4: Code Quality**
- **Risk Level**: Low
- **Impact**: Cosmetic changes only
- **Benefit**: Better code quality, easier maintenance
- **Validation**: Automated tools can handle most issues

## Implementation Plan

### **Step 1: Validation** (Before Any Changes)
```bash
# Verify no active references to attic files
grep -r "attic/" . --exclude-dir=attic

# Check for unused config files
find config/ -name "*.json" -exec basename {} \; | xargs -I {} grep -r "{}" . --exclude-dir=config

# Run all tests to ensure current state
python -m pytest tests/ -v
```

### **Step 2: Phase 1 Implementation**
```bash
# Remove deprecated test files
rm -rf attic/root_tests/

# Remove empty directories
rm -rf attic/scripts/

# Remove legacy documentation
rm -rf attic/docs/

# Remove legacy configs
rm -rf attic/config/ attic/legacy_configs/
```

### **Step 3: Phase 2 Implementation**
```bash
# Update documentation (manual review required)
# Create new guides (manual creation required)
```

### **Step 4: Phase 3 Implementation**
```bash
# Audit config directory (manual review required)
# Move test configs to appropriate location
```

### **Step 5: Phase 4 Implementation**
```bash
# Run ruff cleanup
ruff check --fix .

# Remove debug statements (manual review required)
# Add type hints (manual review required)
```

## Success Criteria

### **Immediate (After Phase 1)**
- [ ] Reduced file count by ~50 files
- [ ] Cleaner directory structure
- [ ] No broken references
- [ ] All tests still pass

### **Short-term (After Phase 2)**
- [ ] Documentation reflects current capabilities
- [ ] Clear guides for Alpha v1 usage
- [ ] Updated changelog with implementation details

### **Medium-term (After Phase 3)**
- [ ] Cleaner configuration structure
- [ ] No unused config files
- [ ] Standardized environment variables

### **Long-term (After Phase 4)**
- [ ] No debug statements in production code
- [ ] Consistent code formatting
- [ ] Complete type hints coverage
- [ ] Full Aurora ruleset compliance

## Rollback Plan

### **Phase 1 Rollback**
- No rollback needed (removing unused files)
- Files already archived in git history

### **Phase 2 Rollback**
```bash
git checkout HEAD~1 -- README.md MASTER_DOCUMENTATION.md docs/
```

### **Phase 3 Rollback**
```bash
git checkout HEAD~1 -- config/
```

### **Phase 4 Rollback**
```bash
git checkout HEAD~1 -- core/ scripts/
```

## Compliance with Aurora Ruleset

### ✅ **Compliant Areas**
- **Protected Paths**: No core/ files will be modified
- **Audit Trail**: This cleanup proposal documented
- **Testing**: All tests will continue to pass
- **Documentation**: Will be updated to reflect changes

### ⚠️ **Areas to Monitor**
- **Configuration Changes**: Need to verify no breaking changes
- **Import Dependencies**: Need to ensure no broken imports
- **Documentation Sync**: Need to keep docs current

## Next Steps

1. **Review this proposal** and approve the approach
2. **Run validation commands** to verify current state
3. **Execute Phase 1** (safe removals)
4. **Execute Phase 2** (documentation updates)
5. **Execute Phase 3** (configuration cleanup)
6. **Execute Phase 4** (code quality improvements)
7. **Verify compliance** with Aurora ruleset

## Approval Required

**To proceed with this cleanup, please provide approval with the token:**
```
APPROVE: CLEANUP-001
```

This will authorize the systematic cleanup of deprecated files and code quality improvements while maintaining full functionality and Aurora ruleset compliance.
