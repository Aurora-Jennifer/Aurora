# Alpha v1 Dependencies

## 🎯 **Executive Summary**

This document outlines all dependencies for the Alpha v1 ML system, including required system components, optional components, external dependencies, and configuration requirements.

## 📋 **Dependency Categories**

### **1. Core Alpha v1 Components (REQUIRED)**
Components that are essential for Alpha v1 functionality.

### **2. System Dependencies (REQUIRED)**
Core system components that Alpha v1 depends on.

### **3. Optional Components (OPTIONAL)**
Components that enhance Alpha v1 but are not required.

### **4. External Dependencies (REQUIRED)**
Third-party libraries and services.

---

## 🔧 **Core Alpha v1 Components (REQUIRED)**

### **ML Training Components**
```
ml/trainers/train_linear.py          # Alpha v1 Ridge regression trainer
ml/features/build_daily.py           # Alpha v1 feature engineering
ml/eval/alpha_eval.py                # Alpha v1 evaluation logic
ml/__init__.py                       # ML package initialization
```

**Purpose**: Core ML training and evaluation logic
**Dependencies**: sklearn, pandas, numpy, yfinance
**Status**: ✅ **REQUIRED** - Cannot function without these

### **Alpha v1 Tools**
```
tools/train_alpha_v1.py              # Alpha v1 training script
tools/validate_alpha.py              # Alpha v1 validation script
```

**Purpose**: Command-line tools for Alpha v1 operations
**Dependencies**: argparse, json, pathlib
**Status**: ✅ **REQUIRED** - Primary entry points for Alpha v1

### **Alpha v1 Scripts**
```
scripts/walkforward_alpha_v1.py      # Alpha v1 walkforward testing
scripts/compare_walkforward.py       # Comparison script
```

**Purpose**: Walkforward testing and comparison
**Dependencies**: core/walk/, core/engine/, core/metrics/
**Status**: ✅ **REQUIRED** - Essential for Alpha v1 validation

### **Alpha v1 Configuration**
```
config/features.yaml                 # Alpha v1 feature definitions
config/models.yaml                   # Alpha v1 model configurations
```

**Purpose**: Alpha v1 configuration files
**Dependencies**: yaml, pathlib
**Status**: ✅ **REQUIRED** - Configuration required for operation

### **Alpha v1 Tests**
```
tests/ml/test_leakage_guards.py      # Leakage prevention tests
tests/ml/test_alpha_eval_contract.py # Evaluation contract tests
```

**Purpose**: Alpha v1 testing and validation
**Dependencies**: pytest, numpy, pandas
**Status**: ✅ **REQUIRED** - Essential for validation

### **Alpha v1 Documentation**
```
docs/runbooks/alpha.md               # Alpha v1 runbook
docs/ALPHA_V1_WALKFORWARD_GUIDE.md   # Walkforward guide
docs/ALPHA_V1_SYSTEM_OVERVIEW.md     # System overview
docs/ALPHA_V1_DEPENDENCIES.md        # This document
```

**Purpose**: Alpha v1 documentation
**Dependencies**: None
**Status**: ✅ **REQUIRED** - Essential for understanding and maintenance

---

## 🏗️ **System Dependencies (REQUIRED)**

### **Core Engine Components**
```
core/engine/backtest.py              # Backtesting engine
core/engine/composer_integration.py  # Composer system integration
core/sim/simulate.py                 # Trading simulation
core/metrics/stats.py                # Performance metrics
```

**Purpose**: Core trading and simulation functionality
**Dependencies**: pandas, numpy
**Status**: ✅ **REQUIRED** - Alpha v1 depends on these for simulation

### **Walkforward Framework**
```
core/walk/folds.py                   # Walkforward fold generation
core/walk/ml_pipeline.py             # Alpha v1 ML pipeline integration
core/walk/pipeline.py                # Walkforward pipeline
core/walk/run.py                     # Walkforward execution
```

**Purpose**: Walkforward testing framework
**Dependencies**: pandas, numpy
**Status**: ✅ **REQUIRED** - Alpha v1 walkforward depends on these

### **Configuration System**
```
core/config_loader.py                # Configuration loading
core/utils.py                        # Core utilities
```

**Purpose**: Configuration and utility functions
**Dependencies**: yaml, pathlib
**Status**: ✅ **REQUIRED** - Alpha v1 depends on configuration system

### **Data Validation**
```
core/data_sanity.py                  # Data validation
config/data_sanity.yaml              # Data validation config
```

**Purpose**: Data integrity validation
**Dependencies**: pandas, numpy
**Status**: ✅ **REQUIRED** - Alpha v1 depends on data validation

### **Risk Management**
```
core/risk/guardrails.py              # Risk management
config/guardrails.yaml               # Risk management config
```

**Purpose**: Risk management and guardrails
**Dependencies**: None
**Status**: ✅ **REQUIRED** - Alpha v1 depends on risk management

---

## 🔄 **Optional Components (OPTIONAL)**

### **Old ML Components**
```
core/ml/profit_learner.py            # Old ML profit learning
core/ml/visualizer.py                # Old ML visualization
core/ml/warm_start.py                # Old ML warm start
```

**Purpose**: Legacy ML components
**Dependencies**: Various ML libraries
**Status**: ⚠️ **OPTIONAL** - Not used by Alpha v1, can be removed

### **Old Walkforward Components**
```
scripts/walkforward_framework.py     # Old regime-based walkforward
```

**Purpose**: Legacy walkforward framework
**Dependencies**: core/walk/, core/engine/
**Status**: ⚠️ **OPTIONAL** - Used for comparison only

### **Old Strategy Components**
```
strategies/                          # Old strategy implementations
signals/                             # Old signal processing
features/                            # Old feature engineering
```

**Purpose**: Legacy strategy and signal components
**Dependencies**: Various
**Status**: ⚠️ **OPTIONAL** - Not used by Alpha v1, can be removed

### **Old Configuration Files**
```
config/ml_backtest_*.json            # Old ML backtest configs
config/paper_*.json                  # Old paper trading configs
config/enhanced_*.json               # Old enhanced configs
```

**Purpose**: Legacy configuration files
**Dependencies**: None
**Status**: ⚠️ **OPTIONAL** - Not used by Alpha v1, can be removed

### **Old Test Files**
```
tests/ml/test_model_golden.py        # Old golden dataset tests
tests/ml/test_feature_stats.py       # Old feature statistics tests
tests/walkforward/test_*.py          # Old walkforward tests
```

**Purpose**: Legacy test files
**Dependencies**: pytest, various
**Status**: ⚠️ **OPTIONAL** - Not used by Alpha v1, can be removed

---

## 📦 **External Dependencies (REQUIRED)**

### **Python Libraries**
```python
# Core ML libraries
scikit-learn>=1.3.0                  # Ridge regression, pipelines
pandas>=2.0.0                        # Data manipulation
numpy>=1.24.0                        # Numerical computing

# Data sources
yfinance>=0.2.0                      # Market data
requests>=2.31.0                     # HTTP requests

# Configuration
pyyaml>=6.0                          # YAML configuration
pathlib                              # Path handling (built-in)

# Testing
pytest>=7.0.0                        # Testing framework
pytest-cov>=4.0.0                    # Coverage testing

# Utilities
tqdm>=4.65.0                         # Progress bars
python-dateutil>=2.8.0               # Date utilities
```

### **System Requirements**
```
Python 3.11+                         # Python version
8GB RAM minimum                      # Memory requirements
10GB disk space                      # Storage requirements
Internet connection                  # Data access
```

### **Optional External Dependencies**
```python
# Visualization (optional)
matplotlib>=3.7.0                    # Plotting
seaborn>=0.12.0                      # Statistical plotting

# Advanced ML (optional)
lightgbm>=4.0.0                      # Gradient boosting
xgboost>=2.0.0                       # Gradient boosting

# Cloud deployment (optional)
boto3>=1.26.0                        # AWS integration
google-cloud-storage>=2.0.0          # GCP integration
```

---

## 🔗 **Dependency Graph**

### **Alpha v1 Core Dependencies**
```
Alpha v1 Training
├── ml/trainers/train_linear.py
├── ml/features/build_daily.py
├── ml/eval/alpha_eval.py
├── config/features.yaml
├── config/models.yaml
└── tools/train_alpha_v1.py

Alpha v1 Walkforward
├── core/walk/ml_pipeline.py
├── core/walk/folds.py
├── core/engine/backtest.py
├── core/sim/simulate.py
├── core/metrics/stats.py
└── scripts/walkforward_alpha_v1.py

Alpha v1 Validation
├── tools/validate_alpha.py
├── tests/ml/test_leakage_guards.py
├── tests/ml/test_alpha_eval_contract.py
└── reports/alpha.schema.json
```

### **System Integration Dependencies**
```
Alpha v1 → Core Engine
├── core/engine/backtest.py
├── core/engine/composer_integration.py
└── core/sim/simulate.py

Alpha v1 → Walkforward Framework
├── core/walk/folds.py
├── core/walk/pipeline.py
└── core/walk/run.py

Alpha v1 → Configuration System
├── core/config_loader.py
├── core/utils.py
└── config/*.yaml

Alpha v1 → Data Validation
├── core/data_sanity.py
└── config/data_sanity.yaml
```

---

## 📊 **Dependency Analysis**

### **Critical Dependencies (Cannot Remove)**
- **ml/trainers/train_linear.py**: Core training logic
- **ml/features/build_daily.py**: Feature engineering
- **ml/eval/alpha_eval.py**: Evaluation logic
- **core/walk/ml_pipeline.py**: Walkforward integration
- **core/engine/backtest.py**: Backtesting engine
- **core/sim/simulate.py**: Trading simulation
- **config/features.yaml**: Feature configuration
- **config/models.yaml**: Model configuration

### **Important Dependencies (Should Keep)**
- **tools/train_alpha_v1.py**: Training script
- **tools/validate_alpha.py**: Validation script
- **scripts/walkforward_alpha_v1.py**: Walkforward testing
- **tests/ml/test_leakage_guards.py**: Leakage tests
- **core/config_loader.py**: Configuration loading
- **core/data_sanity.py**: Data validation

### **Optional Dependencies (Can Remove)**
- **core/ml/profit_learner.py**: Old ML component
- **core/ml/visualizer.py**: Old ML component
- **core/ml/warm_start.py**: Old ML component
- **strategies/**: Old strategy implementations
- **signals/**: Old signal processing
- **features/**: Old feature engineering
- **config/ml_backtest_*.json**: Old configs
- **tests/ml/test_model_golden.py**: Old tests

---

## 🚀 **Deployment Dependencies**

### **Production Requirements**
```
# Core Alpha v1 components
artifacts/models/linear_v1.pkl       # Trained model
artifacts/feature_store/             # Feature data
config/features.yaml                 # Feature config
config/models.yaml                   # Model config

# System components
core/walk/ml_pipeline.py             # ML pipeline
core/engine/backtest.py              # Backtesting
core/sim/simulate.py                 # Simulation
core/metrics/stats.py                # Metrics

# Tools and scripts
tools/train_alpha_v1.py              # Training
tools/validate_alpha.py              # Validation
scripts/walkforward_alpha_v1.py      # Testing
```

### **Development Requirements**
```
# Testing
tests/ml/test_leakage_guards.py      # Leakage tests
tests/ml/test_alpha_eval_contract.py # Contract tests
pytest                               # Test framework

# Documentation
docs/runbooks/alpha.md               # Runbook
docs/ALPHA_V1_SYSTEM_OVERVIEW.md     # Overview
docs/ALPHA_V1_DEPENDENCIES.md        # Dependencies

# Configuration
config/data_sanity.yaml              # Data validation
config/guardrails.yaml               # Risk management
```

---

## ⚠️ **Dependency Risks**

### **High Risk (Critical)**
- **Removing core Alpha v1 components**: Will break Alpha v1 functionality
- **Removing system dependencies**: Will break Alpha v1 integration
- **Removing external dependencies**: Will break Alpha v1 operation

### **Medium Risk (Important)**
- **Removing tools and scripts**: Will break Alpha v1 workflow
- **Removing tests**: Will break Alpha v1 validation
- **Removing documentation**: Will break Alpha v1 understanding

### **Low Risk (Optional)**
- **Removing old ML components**: Safe, not used by Alpha v1
- **Removing old strategies**: Safe, not used by Alpha v1
- **Removing old configs**: Safe, not used by Alpha v1

---

## 🎯 **Recommendations**

### **Keep (Required)**
- All Alpha v1 core components
- All system dependencies
- All external dependencies
- All tools and scripts
- All tests and documentation

### **Review (Optional)**
- Old ML components (can be removed)
- Old strategies (can be removed)
- Old configuration files (can be removed)
- Old test files (can be removed)

### **Remove (Safe)**
- Temporary files
- Cache directories
- Backup files
- Legacy directories

---

**Status**: ✅ **COMPLETE** - All Alpha v1 dependencies documented
**Risk Level**: 🟢 **LOW** - Clear dependency mapping available
**Next Steps**: 🚀 **Use this for safe cleanup planning**
