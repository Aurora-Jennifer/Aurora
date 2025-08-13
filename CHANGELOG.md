# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern Python packaging with `pyproject.toml`
- Pre-commit hooks for code quality
- Comprehensive configuration documentation
- Contributing guidelines with conventional commits
- Automated testing setup

### Changed
- Normalized Python packaging structure
- Consolidated configuration files
- Improved code organization

### Cleanup
- Moved duplicate documentation files to `attic/`
- Moved development tools (dashboards, test files) to `attic/`
- Removed Python cache files and build artifacts
- Consolidated configuration documentation

## [0.1.0] - 2025-08-13

### Added
- Enhanced trading system with regime detection
- IBKR integration for market data and order execution
- Regime-aware ensemble strategy
- Comprehensive logging and monitoring
- Paper trading system
- Risk management framework
- Feature re-weighting system
- Discord notification support

### Features
- **Regime Detection**: Identifies trend, chop, and volatile market conditions
- **Multi-Signal Blending**: Combines trend-following, mean-reversion, and breakout signals
- **Risk Management**: Built-in position sizing and drawdown protection
- **Performance Tracking**: Comprehensive metrics and logging
- **IBKR Integration**: Professional-grade data and execution
- **Paper Trading**: Risk-free testing environment

### Technical
- Python 3.8+ compatibility
- Pandas and NumPy for data processing
- Scikit-learn for machine learning components
- Flask for web dashboard
- Plotly for interactive charts
