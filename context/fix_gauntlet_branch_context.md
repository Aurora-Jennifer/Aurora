# Fix/Gauntlet Branch Context

## Branch: `fix/gauntlet`
**Status**: Active Development  
**Last Updated**: Current (September 2025)  
**Use Case**: Advanced features, active development, testing

## 🎯 Current State

### Core Features
- ✅ Advanced execution engine with two-phase batching
- ✅ Capital scaling (2x position scaling)
- ✅ 15k order caps with comprehensive risk management
- ✅ 5-minute trading intervals
- ✅ Real-time order reconciliation
- ✅ Advanced pre-trade gate
- ✅ Comprehensive risk management
- ✅ Portfolio management with position tracking

### Execution System
- **Type**: Advanced execution engine
- **Order Caps**: 15,000 per order
- **Position Limits**: 15% per symbol
- **Trading Schedule**: Every 5 minutes during market hours
- **Capital Scaling**: 2x scaling factor
- **Batching**: Two-phase (reducers first, then openers)

### Configuration
```yaml
# config/execution.yaml
position_sizing:
  order_notional_cap: 15000.0
  max_position_size: 0.15
  capital_utilization_factor: 2.0

risk_management:
  max_pos_pct: 0.15
  max_order_notional: 15000
  max_notional_opener: 15000
  max_notional_reducer: 15000
```

## 🚀 Quick Start

### Switch to Fix/Gauntlet Branch
```bash
# Save current work
git stash

# Switch to fix/gauntlet
git checkout fix/gauntlet

# Install dependencies
pip install -r requirements-lock.txt

# Setup environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with your credentials

# Start services
systemctl --user start paper-trading-session.service
systemctl --user enable paper-trading-session.service
```

### Service Management
```bash
# Check status
systemctl --user status paper-trading-session.service

# View logs
journalctl --user -u paper-trading-session.service -f

# Restart service
systemctl --user restart paper-trading-session.service
```

## 📊 Expected Performance

### Capital Utilization
- **Position Value**: ~$44,000
- **Capital Used**: ~22% of available capital
- **Order Sizes**: Up to 15,000 per order
- **Risk Level**: Moderate to aggressive

### Trading Behavior
- **Frequency**: Every 5 minutes during market hours
- **Position Changes**: Dynamic and responsive
- **Risk Management**: Advanced controls
- **Monitoring**: Real-time execution monitoring

## 🛡️ Risk Profile

### Strengths
- ✅ High capital utilization
- ✅ Advanced execution engine
- ✅ Real-time trading
- ✅ Comprehensive risk management
- ✅ Two-phase batching
- ✅ Capital scaling

### Considerations
- ⚠️ Under active development
- ⚠️ More complex system
- ⚠️ Higher risk profile
- ⚠️ Requires more monitoring

## 📚 Documentation

### Key Files
- `README.md` - Updated with execution engine
- `docs/CAPITAL_SCALING_GUIDE.md` - Capital scaling guide
- `docs/execution_system_final_status.md` - Execution status
- `core/execution/` - Execution engine source code

### Configuration Files
- `config/execution.yaml` - Advanced execution configuration
- `config/paper-trading.env` - Environment variables
- `data/universe/top300.txt` - Trading universe

## 🔧 Troubleshooting

### Common Issues
1. **Execution Engine Errors**: Check execution engine logs
2. **Capital Scaling Issues**: Verify capital_utilization_factor
3. **Order Rejections**: Check pre-trade gate logs
4. **Service Failures**: Check systemd service status

### Debug Commands
```bash
# Check service status
systemctl --user status paper-trading-session.service

# View execution logs
journalctl --user -u paper-trading-session.service -n 100 | grep -E "SizeDecision|Two-phase plan|Exec meta"

# Check position sizing
journalctl --user -u paper-trading-session.service -n 50 | grep "SizeDecision.*TARGET"

# Monitor capital utilization
journalctl --user -u paper-trading-session.service -n 50 | grep "net_position_value"
```

## 🎯 Use Cases

### Recommended For
- ✅ Active development
- ✅ High capital utilization
- ✅ Real-time trading
- ✅ Advanced features testing
- ✅ Performance optimization

### Not Recommended For
- ❌ Production without testing
- ❌ Conservative trading
- ❌ Set-and-forget deployment
- ❌ Limited monitoring

## 🔄 Migration Notes

### From Main Branch
- Significant configuration changes required
- Service names change (paper-trading → paper-trading-session)
- New execution engine features
- Higher capital utilization
- More frequent trading

### To Main Branch
- Backup current configuration
- Note performance differences
- Prepare for feature reduction
- Update monitoring approach

## 📈 Performance Expectations

### Typical Metrics
- **Sharpe Ratio**: 0.8-1.5
- **Max Drawdown**: 8-15%
- **Win Rate**: 55-65%
- **Capital Utilization**: 20-30%

### Advanced Features
- **Two-Phase Batching**: Reducers first, then openers
- **Capital Scaling**: 2x position scaling
- **Real-time Reconciliation**: Continuous order tracking
- **Advanced Risk Management**: Comprehensive controls

## 🚨 Important Notes

### Service Differences
- **Service Name**: `paper-trading-session.service` (not `paper-trading.service`)
- **Timer**: Runs every 5 minutes during market hours
- **Execution**: Advanced two-phase batching

### Configuration Differences
- **Capital Scaling**: 2x factor enabled
- **Order Caps**: 15k instead of 5k
- **Position Limits**: 15% instead of 10%
- **Risk Management**: More comprehensive

### Monitoring Requirements
- **Real-time Monitoring**: More frequent checks needed
- **Execution Logs**: Monitor execution engine performance
- **Capital Utilization**: Track capital scaling effectiveness
- **Risk Metrics**: Monitor advanced risk controls

## 🔧 Development Workflow

### Making Changes
1. **Test Locally**: Always test changes locally first
2. **Monitor Logs**: Watch execution logs closely
3. **Check Performance**: Monitor capital utilization
4. **Document Changes**: Update documentation

### Testing New Features
1. **Paper Trading**: Test in paper trading mode
2. **Small Positions**: Start with small position sizes
3. **Monitor Closely**: Watch for unexpected behavior
4. **Gradual Rollout**: Increase position sizes gradually
