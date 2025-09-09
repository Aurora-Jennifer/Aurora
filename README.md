# Aurora Trading System

A production-ready quantitative trading system with fully automated paper trading capabilities.

## 🚀 Quick Start

```bash
# Check system status
./monitor_paper_trading.sh

# Run manual dry-run
python ops/enhanced_dry_run.py

# Emergency stop
touch kill.flag
```

## ✨ Key Features

- **🤖 Fully Automated**: Systemd-based automation with daily trading execution
- **📊 Real-time Data**: Alpaca API integration with 130 symbol coverage
- **🧠 Machine Learning**: XGBoost models with leakage prevention and feature engineering
- **🛡️ Risk Management**: Market-neutral positions with capacity constraints
- **📈 Monitoring**: Comprehensive logging and alerting system
- **🔒 Production Ready**: Bulletproof data pipeline with 100% coverage validation

## 🏗️ Architecture

### Core Components
- **📡 Data Pipeline**: `tools/fetch_bars_alpaca.py` - Real-time market data with symbol normalization
- **⚙️ Feature Engineering**: `ml/panel_builder.py` - Cross-sectional features with dispersion guards
- **🎯 Model Training**: `scripts/run_universe.py` - XGBoost training with leakage audit
- **💼 Paper Trading**: `ml/paper_trading_runner.py` - Automated execution with risk controls
- **📊 Monitoring**: `monitor_paper_trading.sh` - System health checks and status reporting

### 🤖 Automation Schedule
- **07:30 CT**: Preflight validation (`paper-preflight.service`)
- **08:00 CT**: Trading execution (`paper-trading.service`)
- **09:00-15:00 CT**: Hourly monitoring (`paper-status.service`)
- **15:15 CT**: End-of-day reporting (`paper-eod.service`)
- **16:00 CT**: Next-day data fetch (`paper-data-fetch.service`)

## 📚 Documentation

- **[Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)** - Complete system architecture
- **[Launch Readiness Checklist](docs/LAUNCH_READINESS_CHECKLIST.md)** - Pre/post launch procedures
- **[Systemd Automation Guide](docs/SYSTEMD_AUTOMATION_GUIDE.md)** - Automation setup and troubleshooting
- **[Data Pipeline Architecture](docs/DATA_PIPELINE_ARCHITECTURE.md)** - Data flow and quality assurance
- **[Automated Paper Trading Guide](docs/AUTOMATED_PAPER_TRADING_GUIDE.md)** - Complete trading operations guide

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Alpaca API credentials
- Systemd user services enabled
- User lingering enabled (`sudo loginctl enable-linger $USER`)

### Installation
```bash
# Clone repository
git clone https://github.com/Aurora-Jennifer/Aurora.git
cd Aurora

# Install dependencies
pip install -r requirements-lock.txt

# Setup environment
cp ~/.config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with your Alpaca API credentials

# Setup automation
./ops/setup_paper_trading_automation.sh
```

### Configuration
- **Environment**: `~/.config/paper-trading.env` - API credentials and system config
- **Universe**: `data/universe/top300.txt` - 130 symbol trading universe
- **Features**: 45 cross-sectional features with whitelist enforcement
- **Risk**: Market-neutral with capacity constraints and sector limits

## 📊 Monitoring

### System Status
```bash
# Check all services
./monitor_paper_trading.sh

# View live logs
journalctl --user -u paper-* -f

# Check data coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/latest/prices.parquet')
print(f'Coverage: {df.symbol.nunique()}/130 symbols')
"
```

### Key Metrics
- **📊 Data Coverage**: 130/130 symbols (100% required)
- **⚙️ Feature Count**: 45 features with cross-sectional dispersion
- **🎯 Model Accuracy**: IC < 0.1 (no leakage detected)
- **🛡️ Risk Metrics**: Beta ≈ 0, sector neutral, capacity constrained

## 🔧 Troubleshooting

### Common Issues
- **Service Failures**: Check `journalctl --user -u paper-*`
- **Data Issues**: Run `./daily_paper_trading.sh fetch`
- **Feature Issues**: Run `python ops/enhanced_dry_run.py`
- **Model Issues**: Retrain with `python scripts/run_universe.py`

### Emergency Procedures
- **🚨 Kill Switch**: `touch kill.flag` (within 60 seconds)
- **🔄 Service Restart**: `systemctl --user restart paper-*`
- **💾 Data Recovery**: Restore from backup
- **🔄 System Recovery**: `git checkout paper-launch-d0`

## 📈 Performance

### Targets
- **📊 Sharpe Ratio**: > 1.0
- **📉 Maximum Drawdown**: < 10%
- **🎯 Win Rate**: > 55%
- **🔄 Turnover**: < 2.0 (monthly)

### Monitoring
- **📊 Daily Reports**: Automatic performance reporting
- **🛡️ Risk Monitoring**: Continuous risk exposure tracking
- **💻 System Health**: Automated health checks
- **🚨 Alerting**: Critical event notifications

## 🔒 Security

### API Security
- **🔄 Credential Rotation**: Regular API key updates
- **🔐 Environment Isolation**: Secure credential storage
- **👤 Access Control**: Proper file permissions (600)
- **🌐 Network Security**: HTTPS-only API calls

### Data Security
- **🔐 Data Encryption**: Sensitive data encrypted at rest
- **👤 Access Control**: Restricted data access
- **📝 Audit Logging**: Complete access logging
- **📋 Data Retention**: Policy-compliant retention

## 🤝 Contributing

### Development
- **✨ Code Quality**: Ruff linting, type hints, comprehensive testing
- **📚 Documentation**: Complete documentation with examples
- **🔒 Security**: Security-first development practices
- **🧪 Testing**: Comprehensive test coverage

### Guidelines
- **📋 Aurora Ruleset**: Follow Aurora engineering charter
- **📝 Audit Trail**: Document all changes with audit trail
- **🧪 Testing**: Test all changes thoroughly
- **📚 Documentation**: Update documentation for all changes

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🆘 Support

### Documentation
- **🏗️ Architecture**: `docs/ARCHITECTURE_OVERVIEW.md`
- **🤖 Operations**: `docs/SYSTEMD_AUTOMATION_GUIDE.md`
- **📡 Data Pipeline**: `docs/DATA_PIPELINE_ARCHITECTURE.md`
- **💼 Trading Guide**: `docs/AUTOMATED_PAPER_TRADING_GUIDE.md`

### Monitoring
- **📊 System Status**: `./monitor_paper_trading.sh`
- **📝 Logs**: `journalctl --user -u paper-*`
- **🚨 Emergency**: `touch kill.flag`

### Contact
- **🐛 Issues**: [GitHub Issues](https://github.com/Aurora-Jennifer/Aurora/issues)
- **📚 Documentation**: `docs/` directory
- **📝 Logs**: `logs/` directory

## 🎯 Current Status

**✅ PRODUCTION READY**: The system is fully operational with automated paper trading
**📊 DATA COVERAGE**: 130/130 symbols (100%)
**🤖 AUTOMATION**: All systemd services active and scheduled
**🛡️ RISK CONTROLS**: Market-neutral with capacity constraints
**📈 MONITORING**: Comprehensive logging and alerting

**🚀 Ready for automated launch tomorrow at 08:00 CT!**

## Reality Check

**Before you get too excited:** This is a retail-grade trading system. It won't make you rich.

- 📊 **What it does:** Automated paper trading, real-time data, risk management
- 🚨 **What it doesn't:** Live trading, institutional features, guaranteed profits
- 🗺️ **Improvement roadmap:** See documentation for enhancement plans

**Bottom line:** Good for learning and paper trading, not for consistent profits.