# Deployment Readiness Guide

## Overview

This document explains the deployment readiness system that provides honest, capability-aware reporting for the bulletproof trading system.

## Component Status Definitions

The deployment system tracks the following component statuses:

### `env_ok`
- **Definition**: Environment validation passes
- **Check**: Python version, required packages, optional packages (LightGBM, XGBoost)
- **Required**: Yes
- **Failure Impact**: Deployment aborts

### `robustness_ok`
- **Definition**: Robustness tests pass
- **Check**: Cost stress testing, signal lag validation, activity gates
- **Required**: Yes
- **Failure Impact**: System not ready for production

### `oos_ok`
- **Definition**: Out-of-sample validation passes
- **Check**: Multi-slice OOS testing, performance consistency
- **Required**: Yes
- **Failure Impact**: Strategy performance not validated

### `lag_ok`
- **Definition**: Signal lag detection passes
- **Check**: Temporal integrity, no future information leakage
- **Required**: Yes
- **Failure Impact**: Strategy may have lookahead bias

### `portfolio_ok`
- **Definition**: Portfolio construction passes
- **Check**: Multi-asset aggregation, risk controls, position bounds
- **Required**: Yes
- **Failure Impact**: Portfolio not ready for deployment

### `ablation_ok`
- **Definition**: Feature ablation analysis passes
- **Check**: Feature importance analysis, ΔSharpe reporting
- **Required**: No (informational)
- **Failure Impact**: Reduced insight into strategy robustness

### `monitoring_ok`
- **Definition**: Monitoring system setup passes
- **Check**: Nightly monitoring, alerting, data freshness validation
- **Required**: Yes
- **Failure Impact**: No production monitoring

## Deployment Status

### `READY`
- **Condition**: All required components pass (`env_ok`, `robustness_ok`, `oos_ok`, `lag_ok`, `portfolio_ok`, `monitoring_ok`)
- **Meaning**: System is ready for production deployment
- **Action**: Proceed with live deployment

### `PARTIAL`
- **Condition**: Some components fail, but core functionality works
- **Meaning**: System has issues that should be addressed
- **Action**: Review failed components and fix before production

### `FAILED`
- **Condition**: Critical components fail (environment, robustness)
- **Meaning**: System is not ready for deployment
- **Action**: Fix critical issues before proceeding

## Capability Gating

The system automatically detects and handles missing ML libraries:

### LightGBM
- **Detection**: `import lightgbm` test
- **Behavior**: LGBM configs are skipped with warning log
- **Impact**: Reduced model variety, but system continues

### XGBoost
- **Detection**: `import xgboost` test
- **Behavior**: XGB configs are skipped with warning log
- **Impact**: Reduced model variety, but system continues

### Capability Reporting
- **Console**: Shows which libraries are missing
- **JSON Report**: Includes capability status in `deployment_report.json`
- **Logs**: Clear warnings when configs are skipped

## Deployment Report

### Location
- **File**: `deployment_report.json`
- **Path**: `{output_dir}/deployment_report.json`

### Structure
```json
{
  "summary": "READY|PARTIAL|FAILED",
  "component_status": {
    "env_ok": true,
    "robustness_ok": true,
    "oos_ok": true,
    "lag_ok": true,
    "portfolio_ok": true,
    "ablation_ok": true,
    "monitoring_ok": true
  },
  "capabilities": {
    "python": true,
    "lightgbm": false,
    "xgboost": true
  }
}
```

### Usage
- **CI/CD**: Check `summary` field for deployment gates
- **Monitoring**: Track component status over time
- **Debugging**: Identify specific component failures

## Policy: No Success Banners Unless READY

### Rule
**No unconditional "production ready" or "deployment successful" banners unless `summary == "READY"`**

### Implementation
- Console output shows ✅/❌ per component
- Overall status shows `READY|PARTIAL|FAILED`
- Capability warnings shown when libraries missing
- No misleading success messages

### Examples

#### READY Status
```
=== SYSTEM SUMMARY ===
 - env_ok: ✅
 - robustness_ok: ✅
 - oos_ok: ✅
 - lag_ok: ✅
 - portfolio_ok: ✅
 - ablation_ok: ✅
 - monitoring_ok: ✅

🚦 Deployment status: READY
```

#### PARTIAL Status
```
=== SYSTEM SUMMARY ===
 - env_ok: ✅
 - robustness_ok: ✅
 - oos_ok: ✅
 - lag_ok: ✅
 - portfolio_ok: ❌
 - ablation_ok: ✅
 - monitoring_ok: ❌

🚦 Deployment status: PARTIAL
ℹ️  LightGBM not available — LGBM configs were skipped.
```

## Baseline Context Stamping

### Purpose
Provide reproducible context for baseline comparisons to explain changing numbers.

### Location
- **File**: `baseline_context.json`
- **Path**: `{output_dir}/baseline_context.json`

### Structure
```json
{
  "symbol": "AAPL",
  "start": "2020-01-01",
  "end": "2024-01-01",
  "costs_bps": 3.0,
  "annualization": "252d",
  "data_source": "yfinance",
  "auto_adjust": true
}
```

### Usage
- **Reproducibility**: Understand why baseline results change
- **Debugging**: Verify data sources and parameters
- **Documentation**: Track analysis parameters over time

## Commands

### Run Deployment Check
```bash
python scripts/deploy_phase3.py --config config/deployment.yaml --output-dir deployment/final
```

### Run Unit Tests
```bash
pytest -q tests/test_portfolio_aggregator.py tests/test_capability_gating.py
```

### Run Baseline Comparison
```bash
python scripts/baseline_comparison.py --symbol AAPL \
  --strategy-results results/cost_stress_03bps/AAPL/grid_results.csv \
  --output-dir reports/baseline_comparison --costs-bps 3.0
```

## Troubleshooting

### Common Issues

#### Environment Check Fails
- **Cause**: Missing required packages
- **Fix**: Install missing packages or update environment

#### Robustness Tests Fail
- **Cause**: Cost stress or signal lag issues
- **Fix**: Review strategy parameters and data quality

#### Portfolio Construction Fails
- **Cause**: No strategies pass filtering criteria
- **Fix**: Relax filtering criteria or improve strategy performance

#### Monitoring Setup Fails
- **Cause**: Missing data files or configuration issues
- **Fix**: Check data pipeline and monitoring configuration

### Debugging Steps

1. **Check Component Status**: Review `deployment_report.json`
2. **Review Logs**: Look for specific error messages
3. **Verify Capabilities**: Check which ML libraries are available
4. **Test Individual Components**: Run specific tests in isolation
5. **Check Dependencies**: Ensure all required files exist

## Best Practices

### Development
- Run deployment check before committing
- Fix all component failures before merging
- Test with missing optional dependencies

### Production
- Only deploy when status is `READY`
- Monitor component status over time
- Set up alerts for component failures

### Maintenance
- Regularly update capability checks
- Review and update component requirements
- Keep deployment documentation current

---

*This document ensures honest, capability-aware deployment reporting for the bulletproof trading system.*
