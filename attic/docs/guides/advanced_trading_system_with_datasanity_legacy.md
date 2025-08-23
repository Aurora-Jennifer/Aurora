# 🚀 Advanced Trading System with DataSanity

A sophisticated algorithmic trading system featuring regime detection, adaptive features, comprehensive data validation, and IBKR integration. The system is designed for institutional-grade reliability with extensive testing and validation frameworks.

## 🎯 Project Overview

This trading system implements a multi-strategy ensemble approach with regime-aware position sizing and comprehensive data validation. The core innovation is the **DataSanity layer** - a comprehensive data validation and repair system that ensures data integrity across all market data sources. The system supports paper trading, backtesting, and walk-forward analysis with professional-grade risk management and performance tracking.

The system targets 65%+ annual returns through regime detection (trend, chop, volatile markets), adaptive feature engineering, and multi-signal blending. It includes extensive testing frameworks, performance monitoring, and safety guards for production deployment.

## 🎯 Key Design Goals

1. **Data Integrity First** - Comprehensive DataSanity validation layer prevents data corruption and lookahead bias
2. **Regime-Aware Trading** - Adaptive strategies that respond to market conditions (trend, chop, volatile)
3. **Risk Management** - Multi-level risk controls including position sizing, drawdown protection, and kill switches
4. **Performance Validation** - Extensive testing with property-based tests, falsification scenarios, and performance benchmarks
5. **Production Ready** - Comprehensive logging, monitoring, and error handling for live trading
6. **Modular Architecture** - Clean separation between data, strategies, engines, and validation layers
7. **Extensible Design** - Easy addition of new strategies, data sources, and validation rules
8. **Audit Trail** - Complete trade logging and performance tracking for compliance

## 🏗️ Architecture

### Core Trading Logic
- **Paper Trading Engine** (`core/engine/paper.py`) - Main trading orchestrator
- **Backtest Engine** (`core/engine/backtest.py`) - Historical performance testing
- **Strategy Factory** (`strategies/factory.py`) - Strategy instantiation and management
- **Portfolio Manager** (`core/portfolio.py`) - Position and risk management

### Data Fetching and Validation
- **DataSanity Layer** (`core/data_sanity.py`) - Comprehensive data validation and repair
- **IBKR Integration** (`brokers/ibkr_broker.py`) - Professional data and execution
- **Data Provider** (`brokers/data_provider.py`) - Abstracted data access layer
- **Feature Engineering** (`features/feature_engine.py`) - Technical indicator generation

### Strategies and Feature Engineering
- **Regime-Aware Ensemble** (`strategies/regime_aware_ensemble.py`) - Main adaptive strategy
- **Feature Reweighter** (`core/feature_reweighter.py`) - Dynamic feature importance
- **Regime Detector** (`core/regime_detector.py`) - Market condition identification
- **Strategy Selector** (`core/strategy_selector.py`) - ML-based strategy selection

### Backtest/Walkforward/Paper Trading Engines
- **Walkforward Framework** (`scripts/walkforward_framework.py`) - Out-of-sample testing
- **Paper Trading Engine** (`core/engine/paper.py`) - Live simulation
- **Backtest Engine** (`core/engine/backtest.py`) - Historical analysis
- **MVB Runner** (`core/mvb_runner.py`) - Multi-strategy validation

### Guardrails and Risk Checks
- **DataSanity Guards** - Runtime data validation enforcement
- **Kill Switches** - Automatic trading suspension on risk breaches
- **Risk Manager** (`core/risk/guardrails.py`) - Position and exposure limits
- **Performance Monitoring** (`core/performance.py`) - Real-time metrics tracking

### Test Suites
- **DataSanity Tests** (`tests/test_data_sanity_enforcement.py`) - Comprehensive validation testing
- **Property-Based Tests** (`tests/test_properties.py`) - Mathematical invariant testing
- **Falsification Tests** (`scripts/falsify_data_sanity.py`) - Edge case discovery
- **Performance Tests** - Speed and memory usage validation

## 🚀 Setup & Usage

### Prerequisites
- Python 3.8 or higher
- IBKR Gateway (for live data)
- Git

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd trader

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Configuration
1. **Copy environment template**:
   ```bash
   cp config/env_example.txt .env
   ```

2. **Configure IBKR settings** in `config/enhanced_paper_trading_config.json`:
   ```json
   {
     "use_ibkr": true,
     "ibkr_config": {
       "paper_trading": true,
       "host": "127.0.0.1",
       "port": 7497,
       "client_id": 12399
     }
   }
   ```

3. **Set up IBKR Gateway** (see `IBKR_GATEWAY_SETUP.md`)

### Running the System

#### Paper Trading
```bash
# Run daily trading session
python cli/paper.py --daily

# Run with specific config
python cli/paper.py --config config/enhanced_paper_trading_config.json
```

#### Backtesting
```bash
# Run backtest
python cli/backtest.py --config config/backtest_config.json

# Run walkforward analysis
python scripts/walkforward_framework.py --symbol SPY --train-len 252 --test-len 63
```

#### Automated Trading
```bash
# Setup systemd service (recommended)
sudo systemctl enable trading-bot.service
sudo systemctl start trading-bot.service

# Or use cron
crontab -e
# Add: 30 9 * * 1-5 /path/to/run_trading_cron.sh
```

#### Monitoring
```bash
# Check logs
tail -f logs/trading_bot.log

# Run Go/No-Go gate
make go-nogo

# Check system status
make quick-sanity
```

## 🧪 Testing Instructions

### Run All Tests
```bash
# Complete test suite
make test

# DataSanity validation tests only
make sanity

# Performance benchmarks
make bench-sanity

# Falsification battery
make falsify

# All DataSanity checks
make all-sanity
```

### Heavy Data Sanity Tests
```bash
# Property-based testing
pytest tests/test_properties.py -v

# Data corruption detection
pytest tests/test_corruption_detection.py -v

# Edge case testing
pytest tests/test_edge_cases.py -v

# Performance validation
pytest tests/test_data_sanity_enforcement.py::TestDataSanityEnforcement::test_performance_safety -v -s
```

### Test Categories
- **Unit Tests**: Core functionality validation
- **Integration Tests**: Component interaction testing
- **Property Tests**: Mathematical invariant validation
- **Falsification Tests**: Edge case discovery
- **Performance Tests**: Speed and memory validation
- **DataSanity Tests**: Data integrity validation

## 📁 File/Folder Glossary

### Core System
- `core/` - Core trading logic and utilities
  - `engine/` - Trading engines (paper, backtest)
  - `data_sanity.py` - Comprehensive data validation layer
  - `regime_detector.py` - Market condition detection
  - `feature_reweighter.py` - Adaptive feature importance
  - `strategy_selector.py` - ML-based strategy selection
  - `portfolio.py` - Position and risk management
  - `performance.py` - Performance tracking and metrics

### Strategies
- `strategies/` - Trading strategy implementations
  - `regime_aware_ensemble.py` - Main adaptive strategy
  - `ensemble_strategy.py` - Basic ensemble strategy
  - `sma_crossover.py` - Simple moving average strategy
  - `momentum.py` - Momentum-based strategy
  - `mean_reversion.py` - Mean reversion strategy
  - `factory.py` - Strategy instantiation

### Data and Brokers
- `brokers/` - Broker integrations
  - `ibkr_broker.py` - IBKR professional integration
  - `data_provider.py` - Abstracted data access
- `data/` - Cached market data
- `features/` - Feature engineering modules

### Configuration and CLI
- `config/` - Configuration files
  - `enhanced_paper_trading_config.json` - Main system config
  - `ibkr_config.json` - IBKR connection settings
  - `data_sanity.yaml` - Data validation rules
- `cli/` - Command-line interfaces
  - `paper.py` - Paper trading CLI
  - `backtest.py` - Backtesting CLI
  - `mvb.py` - Multi-strategy validation

### Testing and Validation
- `tests/` - Comprehensive test suites
  - `test_data_sanity_enforcement.py` - Data validation tests
  - `test_properties.py` - Property-based tests
  - `test_corruption_detection.py` - Data corruption tests
  - `test_edge_cases.py` - Edge case handling
- `scripts/` - Utility scripts
  - `walkforward_framework.py` - Walkforward analysis
  - `falsify_data_sanity.py` - Falsification testing
  - `go_nogo.py` - Production readiness gate

### Results and Logging
- `results/` - Performance results and reports
- `logs/` - Comprehensive logging
  - `trades/` - Trade execution logs
  - `performance/` - Performance metrics
  - `errors/` - Error tracking
  - `system/` - System operations

### Archived Code
- `attic/` - Archived/legacy code
  - `docs/` - Historical documentation
  - `root_tests/` - Superseded test files
  - `config/` - Old configuration files

## 🔧 Maintenance Notes

### Experimental/Legacy Code
- **`attic/`** - Contains archived code that has been superseded:
  - `attic/docs/` - Historical documentation and assessment reports
  - `attic/root_tests/` - Old test files replaced by current test suite
  - `attic/config/` - Deprecated configuration files

### Cache and Temporary Files
- **`__pycache__/`** - Python bytecode cache (auto-generated)
- **`.pytest_cache/`** - Pytest cache (auto-generated)
- **`.mypy_cache/`** - Type checking cache (auto-generated)
- **`.ruff_cache/`** - Linting cache (auto-generated)
- **`.hypothesis/`** - Property testing cache (auto-generated)

### Data Files
- **`data/ibkr/`** - Cached IBKR market data
- **`results/`** - Generated performance reports and logs
- **`logs/`** - Runtime logging output
- **`artifacts/`** - Generated artifacts from testing

### Configuration Management
- **Active configs**: `config/enhanced_paper_trading_config.json`, `config/ibkr_config.json`
- **Test configs**: `test_*_config.json` files for testing
- **Legacy configs**: Moved to `attic/config/`

## 📈 Performance Tracking

### Key Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Regime Performance**: Returns by market regime

### Monitoring
- **Real-time logs**: `logs/trading_bot.log`
- **Trade history**: `logs/trades/trades_YYYY-MM.log`
- **Performance metrics**: `results/performance_report.json`
- **Daily summaries**: `logs/performance/performance_YYYY-MM.log`

## 🔧 Troubleshooting

### Common Issues
1. **IBKR Connection Failed**: Check host, p
