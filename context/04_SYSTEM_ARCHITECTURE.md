# 🏗️ System Architecture & Codebase Structure

## **Project Layout**

```
trader/
├── core/                    # Core trading engine
│   ├── engine/             # Backtest and paper trading engines
│   │   ├── backtest.py     # Backtesting engine
│   │   └── composer_integration.py  # Composer system integration
│   ├── composer/           # Two-level composer system
│   │   ├── contracts.py    # Composer interfaces
│   │   ├── registry.py     # Strategy registry and filtering
│   │   └── simple_composer.py  # Basic composer implementation
│   ├── strategy_selector.py  # ML-based strategy selection
│   ├── regime_detector.py    # Market regime identification
│   ├── portfolio.py         # Portfolio management
│   ├── risk/               # Risk management and guardrails
│   ├── ml/                 # Machine learning components
│   │   ├── profit_learner.py  # ML profit learning
│   │   ├── visualizer.py   # ML visualization
│   │   └── warm_start.py   # ML warm start capabilities
│   ├── walk/               # Walkforward analysis framework
│   │   ├── folds.py        # Fold generation
│   │   └── pipeline.py     # Walkforward pipeline
│   ├── data_sanity.py      # Data validation and integrity
│   ├── config_loader.py    # Configuration management
│   └── utils.py            # Utility functions
├── strategies/             # Trading strategies
│   ├── base.py             # Base strategy class
│   ├── ensemble_strategy.py  # Ensemble strategy implementation
│   └── factory.py          # Strategy factory
├── features/               # Feature engineering
│   ├── regime_features.py  # Regime-aware feature engineering
│   ├── ensemble.py         # Feature ensemble
│   └── feature_engine.py   # Comprehensive feature generation
├── config/                 # Configuration files
│   ├── base.yaml           # Base configuration
│   ├── risk_low.yaml       # Low risk profile
│   ├── risk_balanced.yaml  # Balanced risk profile
│   ├── risk_strict.yaml    # Strict risk profile
│   └── data_sanity.yaml    # Data sanity configuration
├── scripts/                # Utility scripts
│   ├── walkforward_with_composer.py  # Walkforward with composer
│   ├── walkforward_framework.py      # Walkforward framework
│   └── auto_ml_analysis.py # ML analysis automation
├── tests/                  # Test suite
│   ├── test_composer_refactoring.py  # Composer refactoring tests
│   ├── test_composer_end_to_end.py   # End-to-end composer tests
│   └── walkforward/        # Walkforward tests
├── cli/                    # Command line interfaces
│   ├── backtest.py         # Backtesting CLI
│   └── paper.py            # Paper trading CLI
└── docs/                   # Documentation (organized)
    ├── sessions/           # Current development sessions
    ├── analysis/           # System analysis
    ├── reports/            # Audit reports
    ├── guides/             # User guides
    ├── changelogs/         # Version history
    └── roadmaps/           # Future planning
```

## **Core Components Architecture**

### **1. Engine Layer**
```
core/engine/
├── backtest.py              # Main backtesting engine
├── composer_integration.py  # Composer system integration
└── paper_trading.py         # Paper trading engine
```

**Responsibilities**:
- Execute trading strategies
- Manage portfolio positions
- Handle risk management
- Integrate with composer system
- Process market data

### **2. Composer System**
```
core/composer/
├── contracts.py             # Composer interfaces and contracts
├── registry.py              # Strategy registry and filtering
└── simple_composer.py       # Basic composer implementation
```

**Two-Level Architecture**:
- **Level 1**: Strategy Selection (asset class, regime, weighting)
- **Level 2**: Performance Optimization (composite scoring, walkforward tuning)

### **3. Strategy Layer**
```
strategies/
├── base.py                  # Base strategy class
├── ensemble_strategy.py     # Ensemble strategy implementation
└── factory.py               # Strategy factory
```

**Strategy Types**:
- **Base Strategy**: Abstract base class for all strategies
- **Ensemble Strategy**: Combines multiple strategies
- **Regime-Aware**: Adapts to market conditions

### **4. Feature Engineering**
```
features/
├── regime_features.py       # Regime-aware feature engineering
├── ensemble.py              # Feature ensemble
└── feature_engine.py        # Comprehensive feature generation
```

**Feature Types**:
- **Technical Indicators**: 28 enhanced indicators
- **Regime Features**: Market condition features
- **Ensemble Features**: Combined feature sets

### **5. Machine Learning System**
```
core/ml/
├── profit_learner.py        # ML profit learning
├── visualizer.py            # ML visualization
└── warm_start.py            # ML warm start capabilities
```

**ML Components**:
- **Contextual Bandit**: Strategy selection with Thompson sampling
- **Continual Learning**: 19,088+ trades processed
- **Feature Importance**: Tracking and analysis

### **6. Risk Management**
```
core/risk/
├── guardrails.py            # Risk guardrails
└── portfolio.py             # Portfolio risk management
```

**Risk Controls**:
- **Position Sizing**: Volatility targeting
- **Drawdown Limits**: Maximum drawdown protection
- **Daily Loss Limits**: Daily loss protection
- **Multi-Layer**: Multiple risk control levels

### **7. Data Validation**
```
core/data_sanity.py          # Data validation and integrity
```

**Validation Features**:
- **OHLC Consistency**: Price relationship validation
- **Lookahead Detection**: Data leakage prevention
- **Timezone Handling**: UTC timezone enforcement
- **Missing Data**: NaN detection and handling

## **Configuration System**

### **Hierarchical Configuration**
```
config/
├── base.yaml               # Base configuration (required)
├── risk_low.yaml           # Low risk profile overlay
├── risk_balanced.yaml      # Balanced risk profile overlay
├── risk_strict.yaml        # Strict risk profile overlay
└── data_sanity.yaml        # Data sanity configuration
```

**Configuration Loading**:
```python
from core.config import load_config
cfg = load_config([
    Path("config/base.yaml"),
    Path("config/risk_balanced.yaml")
])
```

### **Key Configuration Sections**
- **Engine**: Core engine settings (min_history_bars, rng_seed)
- **Walkforward**: Walkforward analysis settings
- **Data**: Data source and processing settings
- **Risk**: Risk management parameters
- **Composer**: Composer system configuration
- **Tickers**: Default ticker symbols

## **Data Flow Architecture**

### **1. Data Ingestion**
```
yfinance → DataSanity Validation → Feature Engineering → Strategy Input
```

### **2. Strategy Execution**
```
Market Data → Regime Detection → Strategy Selection → Position Sizing → Execution
```

### **3. Composer Integration**
```
Features → Composer → Strategy Weights → Ensemble Decision → Risk Check → Action
```

### **4. Walkforward Analysis**
```
Historical Data → Fold Generation → Training → Testing → Performance Metrics
```

## **Integration Points**

### **1. Composer Integration**
- **Location**: `core/engine/composer_integration.py`
- **Purpose**: Integrate composer system with trading engine
- **Key Functions**: `get_composer_decision()`, `compose()`

### **2. Strategy Registry**
- **Location**: `core/composer/registry.py`
- **Purpose**: Manage and filter available strategies
- **Key Functions**: `build_composer_system()`, `filter_strategies()`

### **3. Data Sanity Integration**
- **Location**: `core/data_sanity.py`
- **Purpose**: Validate data integrity throughout pipeline
- **Key Functions**: `validate_dataframe()`, `validate_and_repair()`

### **4. Configuration Integration**
- **Location**: `core/config_loader.py`
- **Purpose**: Load and merge configuration files
- **Key Functions**: `load_config()`, `deep_merge()`

## **Performance Architecture**

### **1. Caching Strategy**
- **Data Caching**: yfinance data caching
- **Feature Caching**: Computed features cached
- **Result Caching**: Walkforward results cached

### **2. Parallel Processing**
- **Multi-Symbol**: Parallel processing of multiple symbols
- **Fold Processing**: Parallel walkforward fold processing
- **Feature Computation**: Parallel feature engineering

### **3. Memory Management**
- **Lazy Loading**: Load data only when needed
- **Cleanup**: Explicit cleanup between folds
- **Monitoring**: Memory usage tracking

## **Testing Architecture**

### **1. Unit Tests**
- **Component Tests**: Individual component testing
- **Integration Tests**: Component integration testing
- **Mock Tests**: Mock external dependencies

### **2. System Tests**
- **End-to-End Tests**: Full system testing
- **Performance Tests**: Performance regression testing
- **Stress Tests**: Large dataset testing

### **3. Validation Tests**
- **Data Validation**: Data sanity testing
- **Configuration Tests**: Configuration loading testing
- **Error Handling**: Error condition testing

## **Deployment Architecture**

### **1. Development Environment**
- **Local Testing**: Local development and testing
- **CI/CD**: Automated testing and deployment
- **Documentation**: Comprehensive documentation

### **2. Production Environment**
- **Paper Trading**: Live paper trading deployment
- **Monitoring**: Performance and error monitoring
- **Alerting**: Automated alerting system

### **3. Configuration Management**
- **Environment Variables**: Runtime configuration
- **Configuration Files**: YAML-based configuration
- **Profile Management**: Risk profile management
