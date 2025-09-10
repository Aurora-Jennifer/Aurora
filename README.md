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

- **🤖 Fully Automated**: Systemd-based automation with 5-minute trading execution
- **📊 Real-time Data**: Alpaca API integration with 130 symbol coverage
- **🧠 Machine Learning**: XGBoost models with leakage prevention and feature engineering
- **🛡️ Risk Management**: Advanced execution engine with two-phase batching and capital scaling
- **💰 Capital Scaling**: 2x position scaling with 15k order caps for maximum capital utilization
- **📈 Monitoring**: Comprehensive logging and alerting system
- **🔒 Production Ready**: Bulletproof data pipeline with 100% coverage validation

## 🏗️ Architecture

### Core Components
- **📡 Data Pipeline**: `tools/fetch_bars_alpaca.py` - Real-time market data with symbol normalization
- **⚙️ Feature Engineering**: `ml/panel_builder.py` - Cross-sectional features with dispersion guards
- **🎯 Model Training**: `scripts/run_universe.py` - XGBoost training with leakage audit
- **💼 Execution Engine**: `core/execution/` - Advanced order management with two-phase batching
- **🛡️ Risk Management**: `core/execution/risk_manager.py` - Comprehensive risk controls and throttling
- **💰 Position Sizing**: `core/execution/position_sizing.py` - Capital scaling and position optimization
- **📊 Monitoring**: `monitor_paper_trading.sh` - System health checks and status reporting

### 🤖 Automation Schedule
- **08:30-15:00 CT**: Every 5 minutes - Real-time trading execution (`paper-trading-session.service`)
- **Continuous**: Order reconciliation and position management
- **Real-time**: Risk monitoring and position sizing with capital scaling

## 📚 Documentation

- **[Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)** - Complete system architecture
- **[Execution System Status](docs/execution_system_final_status.md)** - Current execution engine status
- **[Systemd Automation Guide](docs/SYSTEMD_AUTOMATION_GUIDE.md)** - Automation setup and troubleshooting
- **[Data Pipeline Architecture](docs/DATA_PIPELINE_ARCHITECTURE.md)** - Data flow and quality assurance
- **[Automated Paper Trading Guide](docs/AUTOMATED_PAPER_TRADING_GUIDE.md)** - Complete trading operations guide
- **[Capital Scaling Guide](docs/CAPITAL_SCALING_GUIDE.md)** - Position sizing and capital utilization

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
- **💰 Capital Utilization**: 2x scaling with 15k order caps (~$44k deployed)
- **⚡ Execution Speed**: 5-minute intervals with two-phase batching

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

**✅ PRODUCTION READY**: Advanced execution engine with capital scaling operational
**📊 DATA COVERAGE**: 130/130 symbols (100%)
**🤖 AUTOMATION**: 5-minute trading intervals with systemd automation
**🛡️ RISK CONTROLS**: Two-phase batching with comprehensive risk management
**💰 CAPITAL SCALING**: 2x position scaling with 15k order caps
**📈 MONITORING**: Real-time execution monitoring and alerting

**🚀 Live and trading every 5 minutes during market hours!**

## Reality Check

**Before you get too excited:** This is a retail-grade trading system. It won't make you rich.

- 📊 **What it does:** Automated paper trading, real-time data, risk management
- 🚨 **What it doesn't:** Live trading, institutional features, guaranteed profits
- 🗺️ **Improvement roadmap:** See documentation for enhancement plans

**Bottom line:** Good for learning and paper trading, not for consistent profits.