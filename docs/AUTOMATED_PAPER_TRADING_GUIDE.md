# Automated Paper Trading Guide

## Overview

The Aurora trading system now operates as a fully automated paper trading system using systemd user services. This guide covers the complete automated workflow, monitoring, and operational procedures.

## System Architecture

### Automated Workflow

```
07:30 CT → Preflight Validation → 08:00 CT → Trading Execution → 09:00 CT → Hourly Monitoring → 15:15 CT → EOD Reporting → 16:00 CT → Data Fetch
```

### Core Components

#### 1. **Preflight Validation** (`paper-preflight.service`)
- **Schedule**: 07:30 CT daily (weekdays only)
- **Purpose**: System health checks and data validation
- **Duration**: ~2-3 minutes
- **Critical**: Yes (blocks trading if fails)

**Validation Steps**:
- Data coverage check (130/130 symbols required)
- Feature pipeline validation
- Model loading verification
- Risk parameter validation
- Environment configuration check

#### 2. **Trading Execution** (`paper-trading.service`)
- **Schedule**: 08:00 CT daily (weekdays only)
- **Purpose**: Main trading execution and position management
- **Duration**: ~5-10 minutes
- **Dependencies**: Preflight validation must pass

**Execution Steps**:
- Load trained model and features
- Generate trading signals
- Apply risk controls and position sizing
- Execute paper trades
- Log all transactions

#### 3. **Hourly Monitoring** (`paper-status.service`)
- **Schedule**: Every hour during market hours (09:00-15:00 CT)
- **Purpose**: Intraday system health monitoring
- **Duration**: ~30 seconds
- **Critical**: No (informational only)

**Monitoring Checks**:
- System resource usage
- Data pipeline status
- Position reconciliation
- Performance metrics

#### 4. **End-of-Day Reporting** (`paper-eod.service`)
- **Schedule**: 15:15 CT daily (weekdays only)
- **Purpose**: Daily performance reporting and position reconciliation
- **Duration**: ~2-3 minutes
- **Dependencies**: Trading execution completed

**Reporting Steps**:
- Calculate daily P&L
- Reconcile all positions
- Generate performance metrics
- Create daily report
- Update performance tracking

#### 5. **Data Fetch** (`paper-data-fetch.service`)
- **Schedule**: 16:00 CT daily (weekdays only)
- **Purpose**: Fetch next-day market data
- **Duration**: ~3-5 minutes
- **Dependencies**: None

**Data Fetch Steps**:
- Connect to Alpaca API
- Fetch latest market data
- Validate data coverage
- Store in `data/latest/prices.parquet`
- Prepare for next trading day

## Data Pipeline

### Real-Time Data Flow

```
Alpaca API → fetch_bars_alpaca.py → prices.parquet → panel_builder.py → features.parquet → run_universe.py → predictions.json → paper_trading_runner.py → positions.json
```

### Data Quality Gates

#### 1. **Coverage Validation**
- **Requirement**: 130/130 symbols must have data
- **Validation**: `check_data_coverage()` function
- **Failure Action**: Preflight fails, trading blocked

#### 2. **Feature Validation**
- **Requirement**: 45 features must be present and valid
- **Validation**: Whitelist enforcement
- **Failure Action**: Preflight fails, trading blocked

#### 3. **Model Validation**
- **Requirement**: Trained model must load successfully
- **Validation**: Model loading and prediction test
- **Failure Action**: Preflight fails, trading blocked

## Risk Management

### Automated Risk Controls

#### 1. **Position Limits**
- **Market Neutrality**: Beta ≈ 0 across all positions
- **Sector Limits**: Sector exposure within defined bounds
- **Capacity Constraints**: ADV-based position limits
- **Gross Exposure**: Total exposure within risk limits

#### 2. **Data Quality Controls**
- **Coverage Gates**: 100% symbol coverage required
- **Feature Quality**: Cross-sectional dispersion validation
- **Temporal Validation**: No lookahead bias
- **Model Validation**: OOF IC < 0.1 required

#### 3. **Operational Controls**
- **Kill Switch**: `touch kill.flag` for emergency halt
- **Environment Validation**: All required variables present
- **Service Health**: All systemd services operational
- **Resource Monitoring**: System resource usage tracking

## Monitoring and Alerting

### System Monitoring

#### 1. **Service Status Monitoring**
```bash
# Check all service status
./monitor_paper_trading.sh

# Check specific service
systemctl --user status paper-preflight.service

# Check timer schedules
systemctl --user list-timers paper-*
```

#### 2. **Log Monitoring**
```bash
# View recent logs
journalctl --user -u paper-* --since "1 hour ago"

# Follow live logs
journalctl --user -u paper-* -f

# View specific service logs
journalctl --user -u paper-preflight.service -f
```

#### 3. **Performance Monitoring**
- **Data Coverage**: 130/130 symbols
- **Feature Count**: 45 features
- **Model Accuracy**: IC metrics
- **Risk Metrics**: Beta, sector exposure
- **Performance**: P&L, Sharpe ratio

### Alerting System

#### 1. **Critical Alerts**
- **Data Coverage < 100%**: Trading blocked
- **Feature Pipeline Failure**: Trading blocked
- **Model Loading Failure**: Trading blocked
- **Risk Limit Breach**: Position adjustment required

#### 2. **Warning Alerts**
- **High System Resource Usage**: Monitor closely
- **Unusual Performance Metrics**: Investigate
- **Data Quality Issues**: Review and fix
- **Service Failures**: Restart services

#### 3. **Emergency Procedures**
- **Kill Switch**: `touch kill.flag` (within 60 seconds)
- **Service Restart**: `systemctl --user restart paper-*`
- **Data Recovery**: Re-fetch data if needed
- **System Recovery**: Complete system restart

## Operational Procedures

### Daily Operations

#### 1. **Pre-Market (07:30 CT)**
- **Preflight Validation**: Automatic system health checks
- **Data Validation**: Ensure 100% data coverage
- **Model Validation**: Verify model loading
- **Risk Validation**: Check all risk parameters

#### 2. **Market Open (08:00 CT)**
- **Trading Execution**: Automatic position generation
- **Risk Controls**: Apply all risk limits
- **Position Sizing**: Calculate optimal position sizes
- **Order Execution**: Submit paper trades

#### 3. **Intraday (09:00-15:00 CT)**
- **Hourly Monitoring**: System health checks
- **Position Tracking**: Monitor all positions
- **Performance Tracking**: Track P&L and metrics
- **Risk Monitoring**: Ensure risk limits maintained

#### 4. **Market Close (15:15 CT)**
- **EOD Reporting**: Generate daily performance report
- **Position Reconciliation**: Verify all positions
- **P&L Calculation**: Calculate daily performance
- **Risk Reporting**: Final risk exposure report

#### 5. **After Hours (16:00 CT)**
- **Data Fetch**: Get next-day market data
- **Data Validation**: Ensure data quality
- **System Preparation**: Prepare for next trading day
- **Backup**: Backup important data

### Weekly Operations

#### 1. **Monday Morning**
- **System Health Check**: Verify all services operational
- **Data Validation**: Check weekend data updates
- **Model Validation**: Verify model performance
- **Risk Review**: Review risk parameters

#### 2. **Friday Afternoon**
- **Weekly Performance Review**: Analyze weekly performance
- **Risk Assessment**: Review risk exposure
- **System Maintenance**: Clean logs and temporary files
- **Backup**: Full system backup

### Monthly Operations

#### 1. **Month-End**
- **Performance Analysis**: Comprehensive performance review
- **Risk Assessment**: Full risk exposure analysis
- **Model Performance**: Review model accuracy
- **System Optimization**: Identify improvement opportunities

#### 2. **Model Updates**
- **Retraining**: Retrain models with latest data
- **Validation**: Validate new model performance
- **Deployment**: Deploy updated models
- **Testing**: Test new models in paper trading

## Troubleshooting

### Common Issues

#### 1. **Service Failures**
**Symptoms**: Service shows "failed" status
**Causes**: 
- Missing environment variables
- Incorrect file paths
- Permission issues
- Network connectivity

**Solutions**:
```bash
# Check service logs
journalctl --user -u paper-preflight.service -n 50

# Verify environment
source ~/.config/paper-trading.env
echo $APCA_API_KEY_ID

# Restart service
systemctl --user restart paper-preflight.service
```

#### 2. **Data Coverage Issues**
**Symptoms**: Preflight fails with coverage error
**Causes**:
- API connectivity issues
- Symbol normalization problems
- Data quality issues

**Solutions**:
```bash
# Check data coverage
python -c "
import pandas as pd
df = pd.read_parquet('data/latest/prices.parquet')
print(f'Coverage: {df.symbol.nunique()}/130 symbols')
"

# Re-fetch data
./daily_paper_trading.sh fetch
```

#### 3. **Feature Pipeline Issues**
**Symptoms**: Feature validation fails
**Causes**:
- Missing feature files
- Feature calculation errors
- Data quality issues

**Solutions**:
```bash
# Check feature files
ls -la data/latest/features.parquet

# Rebuild features
python ml/panel_builder.py

# Validate features
python ops/enhanced_dry_run.py
```

#### 4. **Model Loading Issues**
**Symptoms**: Model validation fails
**Causes**:
- Missing model files
- Model corruption
- Version compatibility

**Solutions**:
```bash
# Check model files
ls -la models/

# Retrain model
python scripts/run_universe.py

# Test model loading
python -c "import joblib; model = joblib.load('models/trained_model.pkl')"
```

### Emergency Procedures

#### 1. **Kill Switch Activation**
```bash
# Emergency halt (within 60 seconds)
touch kill.flag

# Verify halt
./monitor_paper_trading.sh
```

#### 2. **Service Recovery**
```bash
# Stop all services
systemctl --user stop paper-*

# Restart services
systemctl --user start paper-*

# Check status
systemctl --user status paper-*
```

#### 3. **Data Recovery**
```bash
# Restore from backup
cp data/backup/prices.parquet data/latest/

# Re-validate
python ops/enhanced_dry_run.py
```

#### 4. **Complete System Recovery**
```bash
# Revert to previous tag
git checkout paper-launch-d0

# Rebuild system
pip install -r requirements-lock.txt

# Restart services
systemctl --user restart paper-*
```

## Performance Metrics

### Key Performance Indicators

#### 1. **System Performance**
- **Uptime**: > 99% of trading hours
- **Data Coverage**: 100% (130/130 symbols)
- **Feature Count**: 45 features
- **Model Accuracy**: IC metrics

#### 2. **Trading Performance**
- **Information Coefficient**: Target > 0.05
- **Sharpe Ratio**: Target > 1.0
- **Maximum Drawdown**: Target < 10%
- **Win Rate**: Target > 55%

#### 3. **Risk Metrics**
- **Market Beta**: Target ≈ 0
- **Sector Exposure**: Within defined limits
- **Gross Exposure**: Within risk limits
- **Turnover**: Target < 2.0 (monthly)

### Reporting

#### 1. **Daily Reports**
- **Performance Summary**: P&L, Sharpe, drawdown
- **Risk Summary**: Beta, sector exposure, gross exposure
- **System Summary**: Uptime, data coverage, feature count
- **Alert Summary**: Critical alerts and warnings

#### 2. **Weekly Reports**
- **Performance Analysis**: Weekly performance trends
- **Risk Analysis**: Risk exposure trends
- **System Analysis**: System health trends
- **Improvement Opportunities**: Areas for optimization

#### 3. **Monthly Reports**
- **Comprehensive Performance**: Full performance analysis
- **Risk Assessment**: Complete risk exposure analysis
- **Model Performance**: Model accuracy and stability
- **System Optimization**: System improvement recommendations

## Security and Compliance

### Security Measures

#### 1. **API Security**
- **Credential Rotation**: Regular API key updates
- **Environment Isolation**: Keys stored in secure config files
- **Access Control**: Proper file permissions (600)
- **Network Security**: All API calls use HTTPS

#### 2. **Data Security**
- **Data Encryption**: Sensitive data encrypted at rest
- **Access Control**: Restrict data access
- **Audit Logging**: Log all data access
- **Data Retention**: Follow retention policies

#### 3. **System Security**
- **Service Isolation**: Services run as user (not root)
- **File Permissions**: Proper file permissions
- **Network Security**: Firewall rules as needed
- **Regular Updates**: Keep system updated

### Compliance

#### 1. **Data Governance**
- **Feature Whitelist**: 45 protected features with content hashing
- **Sector Snapshots**: Frozen sector mappings for consistency
- **Audit Trail**: Complete change tracking and rollback capability
- **Data Lineage**: Track data from source to prediction

#### 2. **Risk Management**
- **Position Limits**: Enforce all position limits
- **Risk Monitoring**: Continuous risk monitoring
- **Risk Reporting**: Regular risk reports
- **Risk Controls**: Automated risk controls

#### 3. **Operational Compliance**
- **Service Monitoring**: Continuous service monitoring
- **Performance Monitoring**: Continuous performance monitoring
- **Alert Management**: Proper alert handling
- **Incident Response**: Documented incident response procedures

## Future Enhancements

### Phase 2: Live Trading
- **Broker Integration**: Direct order routing to Alpaca
- **Real-time Data**: WebSocket feeds for live price updates
- **Risk Controls**: Real-time position monitoring and limits
- **Compliance**: Enhanced compliance and reporting

### Phase 3: Multi-Asset
- **Asset Classes**: Extend to crypto, forex, commodities
- **Cross-asset Correlation**: Portfolio optimization across asset classes
- **Regime Detection**: Adaptive strategies based on market conditions
- **Advanced Risk**: Multi-asset risk management

### Phase 4: Advanced Features
- **Machine Learning**: Advanced ML models and techniques
- **Alternative Data**: Incorporate alternative data sources
- **Real-time Analytics**: Real-time performance analytics
- **Advanced Reporting**: Enhanced reporting and visualization

## Getting Started

### Quick Start
```bash
# Check system status
./monitor_paper_trading.sh

# Run manual dry-run
python ops/enhanced_dry_run.py

# Emergency stop
touch kill.flag
```

### Key Files
- **Main runner**: `scripts/run_universe.py`
- **Data fetcher**: `tools/fetch_bars_alpaca.py`
- **Dry run**: `ops/enhanced_dry_run.py`
- **Monitoring**: `monitor_paper_trading.sh`
- **Config**: `~/.config/paper-trading.env`

### Support
- **Documentation**: `docs/` directory
- **Logs**: `logs/` directory and `journalctl --user -u paper-*`
- **Monitoring**: `./monitor_paper_trading.sh`
- **Emergency**: `touch kill.flag`
