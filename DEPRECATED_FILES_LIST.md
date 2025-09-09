# Deprecated Files List

## Overview

This document lists files that are no longer used in the current automated paper trading system and should be moved to the `attic/` folder for archival.

## Files to Move to Attic

### 1. **Root Directory Files**

#### **Context and Documentation Files**
- `CLEANUP_SESSION_CONTEXT.md` - Old cleanup session context
- `MASTER_CONTEXT.md` - Old master context file
- `NEXT_SESSION_QUICK_REFERENCE.md` - Old session reference
- `CONTEXT_INSTRUCTIONS.md` - Old context instructions
- `CONTEXT_ORGANIZATION_SUMMARY.md` - Old context organization
- `CURRENT_SESSION_SUMMARY.md` - Old session summary
- `HANDOFF_COMPLETE_CONTEXT.md` - Old handoff context
- `IMMEDIATE_NEXT_STEPS.md` - Old next steps
- `FOLDER_ORGANIZATION_PLAN.md` - Old organization plan

#### **Status and Summary Files**
- `DAY_ONE_LAUNCH_SUMMARY.md` - Old launch summary
- `DAY_ONE_OPERATOR_CHECKLIST.md` - Old operator checklist
- `FINAL_HANDOFF_STATUS.md` - Old handoff status
- `FINAL_STATUS_UPDATE.md` - Old status update
- `SYSTEM_STATUS_SNAPSHOT.md` - Old system status
- `SYSTEMD_AUTOMATION_COMPLETE.md` - Old automation status
- `PRODUCTION_PIPELINE_STATUS.md` - Old pipeline status
- `PRODUCTION_RESULTS_SUMMARY.md` - Old results summary

#### **Analysis and Reports**
- `CODEBASE_ANALYSIS_REPORT.md` - Old codebase analysis
- `PHASE3_ANALYSIS.md` - Old phase 3 analysis
- `CLEANUP_PLAN.md` - Old cleanup plan
- `audit_report.txt` - Old audit report
- `readiness_report.json` - Old readiness report

#### **Temporary and Test Files**
- `test_*.parquet` - All test parquet files
- `test_*.json` - All test JSON files
- `test_*.py` - All test Python files in root
- `temp_*` - All temporary files
- `walkforward_results_*.json` - Old walkforward results

#### **Scripts and Utilities**
- `run_paper_trading_now.sh` - Old paper trading script
- `run_trading_cron.sh` - Old cron script
- `start_paper_trading.sh` - Old start script
- `stop_paper_trading.sh` - Old stop script
- `status_paper_trading.sh` - Old status script
- `validate_system.py` - Old validation script

### 2. **Configuration Files**

#### **Old Configuration Files**
- `test_*_config.json` - All test configuration files
- `temp_*_config.json` - All temporary configuration files
- `conda-env.yaml` - Old conda environment file
- `constraints.txt` - Old constraints file

### 3. **Documentation Files**

#### **Old Documentation**
- `README_SIMPLE.md` - Old simple README
- `README_ULTRA_SIMPLE.md` - Old ultra simple README
- `QUICK_REFERENCE_COMMANDS.md` - Old quick reference
- `SECURITY_CHECKLIST.md` - Old security checklist
- `SECURITY.md` - Old security documentation
- `INVESTOR_PRESENTATION.md` - Old investor presentation
- `PUBLIC_PRESENTATION.md` - Old public presentation
- `MASTER_DOCUMENTATION.md` - Old master documentation

### 4. **Log and Data Files**

#### **Log Files**
- `*.log` - All log files in root
- `gpu_dmon.log` - Old GPU monitoring log
- `grid_runner.log` - Old grid runner log
- `trading.log` - Old trading log

#### **Data Files**
- `grid_results.csv` - Old grid results
- `grid_results.json` - Old grid results JSON
- `coverage.json` - Old coverage data
- `indicators_comparison.png` - Old comparison image

### 5. **Build and Deployment Files**

#### **Build Files**
- `build_secure.py` - Old build script
- `setup_github.sh` - Old GitHub setup script
- `setup_portfolio_demo.py` - Old portfolio demo setup

#### **Deployment Files**
- `universal_v1` - Old universal file
- `PROVENANCE.sha256` - Old provenance file

### 6. **Miscellaneous Files**

#### **System Files**
- `joblib.externals.loky.backend.popen_loky_posix` - System file
- `=4.21` - System file
- `B[DataSanity]` - System file
- `C[Features]` - System file
- `D[Walkforward]` - System file
- `E[Simulation]` - System file
- `F[Metrics]` - System file
- `nn+ins)open(p,w).write(s)print("patched")PY` - System file

#### **Cache and Temporary Files**
- `__pycache__/` - Python cache directory
- `catboost_info/` - CatBoost info directory
- `temp_enhanced_crontab.txt` - Old crontab file

## Files to Keep (Current System)

### **Core System Files**
- `README.md` - Current README
- `daily_paper_trading.sh` - Current daily trading script
- `monitor_paper_trading.sh` - Current monitoring script
- `requirements.txt` - Current requirements
- `requirements-lock.txt` - Current locked requirements
- `Makefile` - Current Makefile
- `pytest.ini` - Current pytest configuration
- `pyproject.toml` - Current project configuration

### **Active Directories**
- `ml/` - Machine learning components
- `ops/` - Operations and automation
- `tools/` - Utility tools
- `data/` - Data storage
- `config/` - Configuration files
- `docs/` - Current documentation
- `tests/` - Current test suite
- `scripts/` - Current scripts

### **Current Documentation**
- `docs/ARCHITECTURE_OVERVIEW.md` - Current architecture
- `docs/LAUNCH_READINESS_CHECKLIST.md` - Current checklist
- `docs/SYSTEMD_AUTOMATION_GUIDE.md` - Current automation guide
- `docs/DATA_PIPELINE_ARCHITECTURE.md` - Current data pipeline
- `docs/AUTOMATED_PAPER_TRADING_GUIDE.md` - Current trading guide

## Migration Commands

### **Create Attic Structure**
```bash
# Create attic directories
mkdir -p attic/root_files
mkdir -p attic/old_docs
mkdir -p attic/old_configs
mkdir -p attic/old_scripts
mkdir -p attic/old_tests
mkdir -p attic/old_logs
mkdir -p attic/old_data
mkdir -p attic/system_files
```

### **Move Files to Attic**
```bash
# Move root directory files
mv CLEANUP_SESSION_CONTEXT.md attic/root_files/
mv MASTER_CONTEXT.md attic/root_files/
mv NEXT_SESSION_QUICK_REFERENCE.md attic/root_files/
mv CONTEXT_INSTRUCTIONS.md attic/root_files/
mv CONTEXT_ORGANIZATION_SUMMARY.md attic/root_files/
mv CURRENT_SESSION_SUMMARY.md attic/root_files/
mv HANDOFF_COMPLETE_CONTEXT.md attic/root_files/
mv IMMEDIATE_NEXT_STEPS.md attic/root_files/
mv FOLDER_ORGANIZATION_PLAN.md attic/root_files/

# Move status and summary files
mv DAY_ONE_LAUNCH_SUMMARY.md attic/root_files/
mv DAY_ONE_OPERATOR_CHECKLIST.md attic/root_files/
mv FINAL_HANDOFF_STATUS.md attic/root_files/
mv FINAL_STATUS_UPDATE.md attic/root_files/
mv SYSTEM_STATUS_SNAPSHOT.md attic/root_files/
mv SYSTEMD_AUTOMATION_COMPLETE.md attic/root_files/
mv PRODUCTION_PIPELINE_STATUS.md attic/root_files/
mv PRODUCTION_RESULTS_SUMMARY.md attic/root_files/

# Move analysis and reports
mv CODEBASE_ANALYSIS_REPORT.md attic/root_files/
mv PHASE3_ANALYSIS.md attic/root_files/
mv CLEANUP_PLAN.md attic/root_files/
mv audit_report.txt attic/root_files/
mv readiness_report.json attic/root_files/

# Move temporary and test files
mv test_*.parquet attic/old_data/
mv test_*.json attic/old_data/
mv test_*.py attic/old_tests/
mv temp_* attic/old_data/
mv walkforward_results_*.json attic/old_data/

# Move old scripts
mv run_paper_trading_now.sh attic/old_scripts/
mv run_trading_cron.sh attic/old_scripts/
mv start_paper_trading.sh attic/old_scripts/
mv stop_paper_trading.sh attic/old_scripts/
mv status_paper_trading.sh attic/old_scripts/
mv validate_system.py attic/old_scripts/

# Move old configuration files
mv test_*_config.json attic/old_configs/
mv temp_*_config.json attic/old_configs/
mv conda-env.yaml attic/old_configs/
mv constraints.txt attic/old_configs/

# Move old documentation
mv README_SIMPLE.md attic/old_docs/
mv README_ULTRA_SIMPLE.md attic/old_docs/
mv QUICK_REFERENCE_COMMANDS.md attic/old_docs/
mv SECURITY_CHECKLIST.md attic/old_docs/
mv SECURITY.md attic/old_docs/
mv INVESTOR_PRESENTATION.md attic/old_docs/
mv PUBLIC_PRESENTATION.md attic/old_docs/
mv MASTER_DOCUMENTATION.md attic/old_docs/

# Move log files
mv *.log attic/old_logs/
mv gpu_dmon.log attic/old_logs/
mv grid_runner.log attic/old_logs/
mv trading.log attic/old_logs/

# Move data files
mv grid_results.csv attic/old_data/
mv grid_results.json attic/old_data/
mv coverage.json attic/old_data/
mv indicators_comparison.png attic/old_data/

# Move build and deployment files
mv build_secure.py attic/old_scripts/
mv setup_github.sh attic/old_scripts/
mv setup_portfolio_demo.py attic/old_scripts/
mv universal_v1 attic/old_data/
mv PROVENANCE.sha256 attic/old_data/

# Move system files
mv joblib.externals.loky.backend.popen_loky_posix attic/system_files/
mv "=4.21" attic/system_files/
mv "B[DataSanity]" attic/system_files/
mv "C[Features]" attic/system_files/
mv "D[Walkforward]" attic/system_files/
mv "E[Simulation]" attic/system_files/
mv "F[Metrics]" attic/system_files/
mv "nn+ins)open(p,w).write(s)print(\"patched\")PY" attic/system_files/

# Move cache and temporary files
mv __pycache__ attic/system_files/
mv catboost_info attic/system_files/
mv temp_enhanced_crontab.txt attic/old_data/
```

### **Clean Up Empty Directories**
```bash
# Remove empty directories if any
find . -type d -empty -delete
```

## Verification

### **Check Attic Structure**
```bash
# Verify attic structure
tree attic/

# Count files in attic
find attic/ -type f | wc -l

# Check remaining files in root
ls -la | grep -v "^d" | wc -l
```

### **Verify Current System**
```bash
# Check current system files
ls -la README.md daily_paper_trading.sh monitor_paper_trading.sh

# Check current directories
ls -la ml/ ops/ tools/ data/ config/ docs/ tests/ scripts/

# Verify systemd services
systemctl --user status paper-*
```

## Benefits of Cleanup

### **1. Reduced Clutter**
- **Before**: 200+ files in root directory
- **After**: ~20 essential files in root directory
- **Improvement**: 90% reduction in root directory clutter

### **2. Improved Navigation**
- **Clear Structure**: Only current system files visible
- **Easy Maintenance**: Deprecated files archived but accessible
- **Better Organization**: Logical file organization

### **3. Enhanced Performance**
- **Faster Searches**: Fewer files to search through
- **Reduced Confusion**: Clear distinction between current and deprecated
- **Better Git Performance**: Fewer files to track

### **4. Professional Appearance**
- **Clean Repository**: Professional, organized appearance
- **Clear Documentation**: Current documentation easily accessible
- **Maintainable Codebase**: Easy to maintain and extend

## Recovery Procedures

### **If Files Are Needed**
```bash
# Search attic for specific files
find attic/ -name "*filename*" -type f

# Move specific file back
mv attic/root_files/specific_file.md ./

# Restore entire category
mv attic/old_docs/* ./
```

### **If Attic Needs Reorganization**
```bash
# Reorganize attic structure
mkdir -p attic/by_date/2025-09-08
mv attic/root_files/* attic/by_date/2025-09-08/

# Create index
find attic/ -type f > attic/FILE_INDEX.txt
```

## Maintenance

### **Regular Cleanup**
- **Monthly**: Review attic for files that can be permanently deleted
- **Quarterly**: Reorganize attic structure if needed
- **Annually**: Archive old attic contents to external storage

### **Documentation Updates**
- **Update README**: Reflect current file structure
- **Update Documentation**: Ensure all docs reference current files
- **Update Scripts**: Ensure scripts reference current file paths

### **Version Control**
- **Commit Changes**: Commit attic organization
- **Tag Releases**: Tag releases after cleanup
- **Document Changes**: Document cleanup in commit messages
