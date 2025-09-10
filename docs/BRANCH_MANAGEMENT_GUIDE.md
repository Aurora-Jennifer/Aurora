# Branch Management Guide

## Overview

This guide provides comprehensive instructions for managing branches in the Aurora Trading System repository, including context switching, development workflows, and deployment procedures.

## 🌿 Branch Structure

### Primary Branches

#### `main` - Production Ready
- **Purpose**: Stable, production-ready codebase
- **Features**: Core trading system with basic execution
- **Service**: `paper-trading.service`
- **Configuration**: Conservative settings (5k caps, 10% limits)

#### `fix/gauntlet` - Active Development
- **Purpose**: Advanced features and active development
- **Features**: Advanced execution engine with capital scaling
- **Service**: `paper-trading-session.service`
- **Configuration**: Aggressive settings (15k caps, 15% limits, 2x scaling)

#### `master` - Legacy
- **Purpose**: Historical reference (deprecated)
- **Features**: Original system implementation
- **Service**: N/A
- **Configuration**: Legacy settings

### Feature Branches

#### `feat/harden-datasanity-guards`
- **Purpose**: Data sanity improvements
- **Features**: Enhanced data validation
- **Status**: Stable

#### `chore/cleanup-20250819`
- **Purpose**: Code cleanup and organization
- **Features**: Repository maintenance
- **Status**: Complete

#### `docs/lint-triage`
- **Purpose**: Documentation improvements
- **Features**: Documentation updates
- **Status**: Complete

## 🔄 Context Switching

### Quick Switch Commands

#### Switch to Production (main)
```bash
# Save current work
git stash push -m "Work in progress before switching to main"

# Switch to main
git checkout main

# Update dependencies
pip install -r requirements-lock.txt

# Stop current services
systemctl --user stop paper-trading-session.service || true

# Start production services
systemctl --user start paper-trading.service
systemctl --user enable paper-trading.service

# Restore work if needed
git stash pop
```

#### Switch to Development (fix/gauntlet)
```bash
# Save current work
git stash push -m "Work in progress before switching to fix/gauntlet"

# Switch to development
git checkout fix/gauntlet

# Update dependencies
pip install -r requirements-lock.txt

# Stop current services
systemctl --user stop paper-trading.service || true

# Start development services
systemctl --user start paper-trading-session.service
systemctl --user enable paper-trading-session.service

# Restore work if needed
git stash pop
```

### Full Context Restoration

#### Production Context Setup
```bash
# Switch to main
git checkout main

# Install dependencies
pip install -r requirements-lock.txt

# Setup production environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with production credentials

# Configure production settings
# Edit config/execution.yaml for conservative settings

# Start production services
systemctl --user start paper-trading.service
systemctl --user enable paper-trading.service

# Verify setup
systemctl --user status paper-trading.service
journalctl --user -u paper-trading.service -n 10
```

#### Development Context Setup
```bash
# Switch to fix/gauntlet
git checkout fix/gauntlet

# Install dependencies
pip install -r requirements-lock.txt

# Setup development environment
cp config/paper-trading.env.example ~/.config/paper-trading.env
# Edit with development credentials

# Configure development settings
# Edit config/execution.yaml for aggressive settings

# Start development services
systemctl --user start paper-trading-session.service
systemctl --user enable paper-trading-session.service

# Verify setup
systemctl --user status paper-trading-session.service
journalctl --user -u paper-trading-session.service -n 10
```

## 📊 Branch Comparison Matrix

| Feature | main | fix/gauntlet | master |
|---------|------|--------------|--------|
| **Execution Engine** | Basic | Advanced | None |
| **Capital Scaling** | 1x | 2x | N/A |
| **Order Caps** | 5k | 15k | N/A |
| **Position Limits** | 10% | 15% | N/A |
| **Trading Intervals** | Daily | 5-minute | N/A |
| **Risk Management** | Basic | Advanced | Basic |
| **Two-Phase Batching** | No | Yes | No |
| **Real-time Reconciliation** | No | Yes | No |
| **Service Name** | paper-trading | paper-trading-session | N/A |
| **Capital Utilization** | ~7% | ~22% | N/A |

## 🎯 Use Case Recommendations

### Choose `main` for:
- ✅ Production trading
- ✅ Stable, conservative approach
- ✅ Learning the system
- ✅ Long-term deployment
- ✅ Set-and-forget operation

### Choose `fix/gauntlet` for:
- ✅ Active development
- ✅ High capital utilization
- ✅ Real-time trading
- ✅ Advanced features testing
- ✅ Performance optimization

### Choose `master` for:
- ✅ Historical reference
- ✅ Understanding original implementation
- ✅ Legacy code analysis

## 🔧 Development Workflow

### Feature Development
```bash
# Start from fix/gauntlet
git checkout fix/gauntlet

# Create feature branch
git checkout -b feat/new-feature

# Make changes
# ... develop feature ...

# Test changes
systemctl --user restart paper-trading-session.service
journalctl --user -u paper-trading-session.service -f

# Commit changes
git add .
git commit -m "feat: implement new feature"

# Push feature branch
git push origin feat/new-feature

# Create pull request to fix/gauntlet
```

### Bug Fixes
```bash
# Start from appropriate branch
git checkout main  # or fix/gauntlet

# Create bugfix branch
git checkout -b fix/bug-description

# Make fixes
# ... fix bug ...

# Test fixes
systemctl --user restart paper-trading.service  # or paper-trading-session.service
journalctl --user -u paper-trading.service -f

# Commit fixes
git add .
git commit -m "fix: resolve bug description"

# Push bugfix branch
git push origin fix/bug-description

# Create pull request
```

### Documentation Updates
```bash
# Start from main
git checkout main

# Create docs branch
git checkout -b docs/update-documentation

# Update documentation
# ... update docs ...

# Commit changes
git add .
git commit -m "docs: update documentation"

# Push docs branch
git push origin docs/update-documentation

# Create pull request
```

## 🚀 Deployment Procedures

### Deploy to Production
```bash
# Ensure on main branch
git checkout main

# Pull latest changes
git pull origin main

# Install dependencies
pip install -r requirements-lock.txt

# Stop development services
systemctl --user stop paper-trading-session.service || true

# Start production services
systemctl --user start paper-trading.service
systemctl --user enable paper-trading.service

# Verify deployment
systemctl --user status paper-trading.service
journalctl --user -u paper-trading.service -n 20
```

### Deploy Development Features
```bash
# Ensure on fix/gauntlet branch
git checkout fix/gauntlet

# Pull latest changes
git pull origin fix/gauntlet

# Install dependencies
pip install -r requirements-lock.txt

# Stop production services
systemctl --user stop paper-trading.service || true

# Start development services
systemctl --user start paper-trading-session.service
systemctl --user enable paper-trading-session.service

# Verify deployment
systemctl --user status paper-trading-session.service
journalctl --user -u paper-trading-session.service -n 20
```

## 🛡️ Safety Procedures

### Before Switching Branches
1. **Save Work**: Always stash or commit current work
2. **Stop Services**: Stop all running services
3. **Backup Config**: Backup current configuration
4. **Check Status**: Verify current system status

### After Switching Branches
1. **Update Dependencies**: Install required dependencies
2. **Update Configuration**: Review and update configuration
3. **Start Services**: Start appropriate services
4. **Verify Setup**: Check service status and logs
5. **Test Functionality**: Run basic tests

### Emergency Procedures
```bash
# Emergency stop all services
systemctl --user stop paper-trading.service
systemctl --user stop paper-trading-session.service

# Emergency switch to stable branch
git stash
git checkout main
systemctl --user start paper-trading.service

# Emergency restore from backup
git checkout backup-before-doc-updates
systemctl --user start paper-trading.service
```

## 📚 Context Files

### Available Context Files
- `context/BRANCH_CONTEXT.md` - Main branch context guide
- `context/main_branch_context.md` - Main branch specific context
- `context/fix_gauntlet_branch_context.md` - Development branch context
- `context/production_branch_context.md` - Future production branch context

### Using Context Files
```bash
# View branch context
cat context/BRANCH_CONTEXT.md

# View specific branch context
cat context/main_branch_context.md
cat context/fix_gauntlet_branch_context.md
```

## 🔍 Monitoring and Verification

### Service Status Checks
```bash
# Check main branch service
systemctl --user status paper-trading.service

# Check development branch service
systemctl --user status paper-trading-session.service

# Check all services
systemctl --user list-units --type=service | grep paper
```

### Log Monitoring
```bash
# Monitor main branch logs
journalctl --user -u paper-trading.service -f

# Monitor development branch logs
journalctl --user -u paper-trading-session.service -f

# Check recent logs
journalctl --user -u paper-trading*.service -n 50
```

### Configuration Verification
```bash
# Check current configuration
grep -A 10 "position_sizing:" config/execution.yaml

# Check service configuration
systemctl --user show paper-trading.service
systemctl --user show paper-trading-session.service
```

## 🎯 Best Practices

### Branch Management
1. **Always Stash**: Save work before switching
2. **Stop Services**: Stop services before switching
3. **Update Dependencies**: Install dependencies after switching
4. **Verify Configuration**: Check configuration after switching
5. **Test Services**: Verify services work after switching

### Development Workflow
1. **Feature Branches**: Use feature branches for new development
2. **Pull Requests**: Use pull requests for code review
3. **Testing**: Test thoroughly before merging
4. **Documentation**: Update documentation with changes
5. **Version Control**: Use semantic versioning

### Deployment
1. **Staging First**: Test in staging before production
2. **Rollback Plan**: Always have a rollback plan
3. **Monitoring**: Monitor after deployment
4. **Documentation**: Document deployment procedures
5. **Backup**: Backup before major deployments

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status
systemctl --user status paper-trading.service

# Check logs
journalctl --user -u paper-trading.service -n 50

# Restart service
systemctl --user restart paper-trading.service
```

#### Configuration Issues
```bash
# Check configuration syntax
python -c "import yaml; yaml.safe_load(open('config/execution.yaml'))"

# Compare configurations
diff config/execution.yaml config/execution.yaml.example
```

#### Dependency Issues
```bash
# Reinstall dependencies
pip install -r requirements-lock.txt

# Check Python version
python --version

# Check environment
which python
```

### Emergency Recovery
```bash
# Emergency stop
systemctl --user stop paper-trading*.service

# Emergency switch to stable
git stash
git checkout main
systemctl --user start paper-trading.service

# Emergency restore
git checkout backup-before-doc-updates
systemctl --user start paper-trading.service
```
