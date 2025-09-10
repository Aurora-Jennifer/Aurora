# Main Branch Context

## Branch: `main`
**Status**: Production Ready  
**Last Updated**: Pre-execution engine implementation  
**Use Case**: Production deployment, stable releases

## 🎯 Current State

### Core Features
- ✅ Basic paper trading system
- ✅ XGBoost model integration
- ✅ Alpaca API integration
- ✅ Basic risk management
- ✅ Systemd automation
- ✅ Data pipeline with 130 symbols

### Execution System
- **Type**: Basic execution
- **Order Caps**: 5,000 per order
- **Position Limits**: 10% per symbol
- **Trading Schedule**: Daily execution
- **Capital Scaling**: 1x (no scaling)

### Configuration
```yaml
# config/execution.yaml
position_sizing:
  order_notional_cap: 5000.0
  max_position_size: 0.10
  capital_utilization_factor: 1.0

risk_management:
  max_pos_pct: 0.10
  max_order_notional: 5000
```

## 🚀 Quick Start

### Switch to Main Branch
```bash
# Save current work
git stash

# Switch to main
git checkout main

# Install dependencies
pip install -r requirements-lock.txt

# Setup environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with your credentials

# Start services
systemctl --user start paper-trading.service
systemctl --user enable paper-trading.service
```

### Service Management
```bash
# Check status
systemctl --user status paper-trading.service

# View logs
journalctl --user -u paper-trading.service -f

# Restart service
systemctl --user restart paper-trading.service
```

## 📊 Expected Performance

### Capital Utilization
- **Position Value**: ~$14,000
- **Capital Used**: ~7% of available capital
- **Order Sizes**: 500-5,000 per order
- **Risk Level**: Conservative

### Trading Behavior
- **Frequency**: Daily execution
- **Position Changes**: Moderate
- **Risk Management**: Basic controls
- **Monitoring**: Standard logging

## 🛡️ Risk Profile

### Strengths
- ✅ Stable and tested
- ✅ Conservative risk management
- ✅ Production-ready
- ✅ Well-documented

### Limitations
- ⚠️ Limited capital utilization
- ⚠️ Basic execution engine
- ⚠️ No real-time trading
- ⚠️ Limited position sizing

## 📚 Documentation

### Key Files
- `README.md` - Main documentation
- `docs/ARCHITECTURE_OVERVIEW.md` - System architecture
- `docs/AUTOMATED_PAPER_TRADING_GUIDE.md` - Trading guide
- `docs/SYSTEMD_AUTOMATION_GUIDE.md` - Automation guide

### Configuration Files
- `config/execution.yaml` - Execution configuration
- `config/paper-trading.env` - Environment variables
- `data/universe/top300.txt` - Trading universe

## 🔧 Troubleshooting

### Common Issues
1. **Service Not Starting**: Check systemd user services
2. **API Errors**: Verify Alpaca credentials
3. **Data Issues**: Check data pipeline status
4. **Model Errors**: Verify model files exist

### Debug Commands
```bash
# Check service status
systemctl --user status paper-trading.service

# View recent logs
journalctl --user -u paper-trading.service -n 50

# Check data coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/latest/prices.parquet')
print(f'Coverage: {df.symbol.nunique()}/130 symbols')
"
```

## 🎯 Use Cases

### Recommended For
- ✅ Production trading
- ✅ Stable, conservative approach
- ✅ Learning the system
- ✅ Long-term deployment

### Not Recommended For
- ❌ Active development
- ❌ High capital utilization
- ❌ Real-time trading
- ❌ Advanced features

## 🔄 Migration Notes

### From Development Branches
- Configuration may need adjustment
- Service names may differ
- Some features may not be available
- Test thoroughly after migration

### To Development Branches
- Backup current configuration
- Note current performance metrics
- Prepare for feature differences
- Update documentation

## 📈 Performance Expectations

### Typical Metrics
- **Sharpe Ratio**: 0.5-1.0
- **Max Drawdown**: 5-10%
- **Win Rate**: 50-60%
- **Capital Utilization**: 5-10%

### Monitoring
- Daily performance reports
- Risk metric tracking
- System health monitoring
- Error logging and alerting
