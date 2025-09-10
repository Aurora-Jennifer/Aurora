# Branch Context Guide

## Overview

This guide provides context and switching instructions for different branches in the Aurora Trading System repository.

## 🌿 Branch Overview

### Main Branches

#### `main` - Production Ready
- **Status**: Stable, production-ready codebase
- **Features**: Core trading system with basic execution
- **Use Case**: Production deployment, stable releases
- **Last Updated**: Pre-execution engine implementation

#### `fix/gauntlet` - Current Development
- **Status**: Active development with advanced execution engine
- **Features**: 
  - Advanced execution engine with two-phase batching
  - Capital scaling (2x position scaling)
  - 15k order caps with comprehensive risk management
  - 5-minute trading intervals
  - Real-time order reconciliation
- **Use Case**: Active development, testing new features
- **Last Updated**: Current (September 2025)

#### `master` - Legacy
- **Status**: Legacy branch (deprecated)
- **Features**: Original system implementation
- **Use Case**: Historical reference only
- **Last Updated**: Pre-refactoring

### Feature Branches

#### `feat/harden-datasanity-guards`
- **Status**: Data sanity improvements
- **Features**: Enhanced data validation and guards
- **Use Case**: Data quality improvements

#### `chore/cleanup-20250819`
- **Status**: Code cleanup and organization
- **Features**: Repository cleanup and organization
- **Use Case**: Maintenance and cleanup

#### `docs/lint-triage`
- **Status**: Documentation improvements
- **Features**: Documentation updates and linting fixes
- **Use Case**: Documentation maintenance

## 🔄 Branch Switching Guide

### Quick Context Switch

#### Switch to Production (main)
```bash
# Save current work
git stash

# Switch to main
git checkout main

# Restore work if needed
git stash pop
```

#### Switch to Development (fix/gauntlet)
```bash
# Save current work
git stash

# Switch to development
git checkout fix/gauntlet

# Restore work if needed
git stash pop
```

### Full Context Restoration

#### Production Context (main)
```bash
# Switch to main
git checkout main

# Install dependencies
pip install -r requirements-lock.txt

# Setup production environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with production credentials

# Start production services
systemctl --user start paper-trading.service
```

#### Development Context (fix/gauntlet)
```bash
# Switch to development
git checkout fix/gauntlet

# Install dependencies
pip install -r requirements-lock.txt

# Setup development environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with development credentials

# Start development services
systemctl --user start paper-trading-session.service
```

## 📊 Branch Comparison

| Feature | main | fix/gauntlet | master |
|---------|------|--------------|--------|
| Execution Engine | Basic | Advanced | None |
| Capital Scaling | No | 2x scaling | No |
| Order Caps | 5k | 15k | N/A |
| Trading Intervals | Daily | 5-minute | N/A |
| Risk Management | Basic | Advanced | Basic |
| Two-Phase Batching | No | Yes | No |
| Real-time Reconciliation | No | Yes | No |

## 🎯 Use Case Recommendations

### For Production Trading
- **Branch**: `main`
- **Reason**: Stable, tested, production-ready
- **Trade-off**: Limited features, basic execution

### For Development/Testing
- **Branch**: `fix/gauntlet`
- **Reason**: Latest features, advanced execution
- **Trade-off**: More complex, under active development

### For Historical Reference
- **Branch**: `master`
- **Reason**: Original implementation
- **Trade-off**: Outdated, not maintained

## 🔧 Context-Specific Configuration

### Main Branch Configuration
```yaml
# config/execution.yaml (main)
position_sizing:
  order_notional_cap: 5000.0
  max_position_size: 0.10
  capital_utilization_factor: 1.0

risk_management:
  max_pos_pct: 0.10
  max_order_notional: 5000
```

### Fix/Gauntlet Configuration
```yaml
# config/execution.yaml (fix/gauntlet)
position_sizing:
  order_notional_cap: 15000.0
  max_position_size: 0.15
  capital_utilization_factor: 2.0

risk_management:
  max_pos_pct: 0.15
  max_order_notional: 15000
```

## 🚨 Important Notes

### Service Management
- **main**: Uses `paper-trading.service`
- **fix/gauntlet**: Uses `paper-trading-session.service`
- Always stop services before switching branches

### Configuration Files
- Each branch may have different configuration requirements
- Always check `config/execution.yaml` after switching
- Update environment variables as needed

### Dependencies
- Some branches may have different dependency requirements
- Always run `pip install -r requirements-lock.txt` after switching

## 📚 Related Documentation

- **[README.md](../README.md)** - Main project documentation
- **[Capital Scaling Guide](CAPITAL_SCALING_GUIDE.md)** - Capital scaling implementation
- **[Execution System Status](execution_system_final_status.md)** - Current execution status
- **[Systemd Automation Guide](SYSTEMD_AUTOMATION_GUIDE.md)** - Service management

## 🔄 Migration Guide

### From main to fix/gauntlet
1. **Backup**: Create backup of current configuration
2. **Switch**: `git checkout fix/gauntlet`
3. **Update Config**: Review and update `config/execution.yaml`
4. **Update Services**: Switch to `paper-trading-session.service`
5. **Test**: Run system tests to verify functionality

### From fix/gauntlet to main
1. **Backup**: Create backup of current configuration
2. **Switch**: `git checkout main`
3. **Update Config**: Review and update `config/execution.yaml`
4. **Update Services**: Switch to `paper-trading.service`
5. **Test**: Run system tests to verify functionality

## 🎯 Best Practices

1. **Always Stash**: Save work before switching branches
2. **Check Config**: Review configuration after switching
3. **Test Services**: Verify services work after switching
4. **Document Changes**: Keep track of branch-specific changes
5. **Backup First**: Always backup before major switches
