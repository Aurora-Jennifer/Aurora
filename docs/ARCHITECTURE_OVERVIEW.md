# Aurora Trading System - Architecture Overview

## System Architecture

### Core Components

#### 1. Data Pipeline (`tools/fetch_bars_alpaca.py`)
- **Purpose**: Fetches real-time market data from Alpaca API
- **Features**:
  - Symbol normalization (BRK-B → BRK.B, FB → META, etc.)
  - Batching for large symbol sets (50 symbols per request)
  - Strict mode to prevent silent mock fallback
  - Data coverage validation (130/130 symbols required)
  - Timezone-aware date handling
- **Output**: `data/latest/prices.parquet`

#### 2. Feature Engineering (`ml/panel_builder.py`)
- **Purpose**: Builds cross-sectional features from raw market data
- **Key Features**:
  - Cross-sectional transformations using `groupby.transform()`
  - Safe residualization against market/sector factors
  - Feature dispersion guards to prevent flat predictions
  - Whitelist enforcement (45 protected features)
- **Output**: Feature matrix with proper cross-sectional dispersion

#### 3. Model Training (`scripts/run_universe.py`)
- **Purpose**: Trains XGBoost models with proper validation
- **Features**:
  - Leakage audit (OOF IC < 0.1 required)
  - Structural leakage detection
  - Deterministic training (random_state, n_jobs=1)
  - Early stopping and device consistency
- **Output**: Trained models with honest IC metrics

#### 4. Risk Management (`ml/risk_neutralization.py`, `ml/capacity_enforcement.py`)
- **Purpose**: Ensures market-neutral, capacity-aware positions
- **Features**:
  - Market beta neutralization
  - Sector exposure limits
  - ADV-based capacity constraints
  - Volume-dependent slippage model
- **Output**: Risk-adjusted position weights

#### 5. Paper Trading Engine (`ml/paper_trading_runner.py`)
- **Purpose**: Executes paper trades with real market data
- **Features**:
  - Kill-switch integration
  - Position reconciliation
  - Daily P&L reporting
  - Alert system integration
- **Output**: Trade logs and performance metrics

### Automation Layer

#### Systemd Services
- **`paper-preflight.service`**: Pre-market validation (07:30 CT)
- **`paper-trading.service`**: Main trading execution (08:00 CT)
- **`paper-status.service`**: Intraday monitoring (hourly)
- **`paper-eod.service`**: End-of-day reporting (15:15 CT)
- **`paper-data-fetch.service`**: Next-day data preparation (16:00 CT)

#### Environment Management
- **Persistent Config**: `~/.config/paper-trading.env`
- **Auto-loading**: Environment variables loaded by all services
- **Security**: API keys stored securely with proper permissions

### Data Flow

```
Alpaca API → fetch_bars_alpaca.py → prices.parquet
    ↓
panel_builder.py → feature_matrix.parquet
    ↓
run_universe.py → trained_model.pkl
    ↓
paper_trading_runner.py → positions.json
    ↓
EOD reporting → performance_metrics.json
```

### Key Architectural Decisions

#### 1. Cross-Sectional Feature Engineering
- **Problem**: `groupby.apply()` caused index misalignment and flat predictions
- **Solution**: Use `groupby.transform()` for all cross-sectional operations
- **Result**: Proper feature dispersion and non-flat predictions

#### 2. Data Coverage Validation
- **Problem**: Partial data could lead to incomplete portfolios
- **Solution**: Hard gate requiring 100% symbol coverage (130/130)
- **Result**: Bulletproof data pipeline with fail-fast validation

#### 3. Leakage Prevention
- **Problem**: High OOF IC (0.8+) indicated data leakage
- **Solution**: Structural leakage audit and feature allowlist
- **Result**: Honest IC metrics (< 0.1) for reliable backtesting

#### 4. Capacity Management
- **Problem**: Unrealistic position sizes and slippage assumptions
- **Solution**: ADV-based limits and volume-dependent cost model
- **Result**: Realistic trading costs and capacity constraints

### Monitoring and Observability

#### Logging
- **Structured JSON logs** with run_id, symbol, fold, latency_ms, pnl
- **Production logging** with proper encoding and buffering
- **Systemd journal integration** for service monitoring

#### Metrics
- **Performance**: Sharpe ratio, max drawdown, win rate, volatility
- **Risk**: Market beta, sector exposure, gross exposure
- **Operational**: Data coverage, feature count, model accuracy

#### Alerts
- **Kill-switch**: `touch kill.flag` for emergency halt
- **Data issues**: Coverage < 100% triggers preflight failure
- **Performance**: Anomaly detection for unusual metrics

### Security and Compliance

#### API Security
- **Credential rotation**: Regular API key updates
- **Environment isolation**: Keys stored in secure config files
- **Access control**: Proper file permissions (600)

#### Data Governance
- **Feature whitelist**: 45 protected features with content hashing
- **Sector snapshots**: Frozen sector mappings for consistency
- **Audit trail**: Complete change tracking and rollback capability

### Deployment Architecture

#### Production Readiness
- **Deterministic execution**: Fixed random seeds and device consistency
- **Error handling**: Graceful degradation and fail-safe defaults
- **Rollback capability**: Tagged releases with quick revert options

#### Scalability
- **Universe size**: Currently 130 symbols, expandable to 1000+
- **Feature count**: 45 features with room for growth
- **Model complexity**: XGBoost with early stopping for efficiency

### Future Enhancements

#### Phase 2: Live Trading
- **Broker integration**: Direct order routing to Alpaca
- **Real-time data**: WebSocket feeds for live price updates
- **Risk controls**: Real-time position monitoring and limits

#### Phase 3: Multi-Asset
- **Asset classes**: Extend to crypto, forex, commodities
- **Cross-asset correlation**: Portfolio optimization across asset classes
- **Regime detection**: Adaptive strategies based on market conditions

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

### Troubleshooting
- **Data issues**: Check `data/latest/prices.parquet` exists and has 130 symbols
- **Service issues**: Use `journalctl --user -u paper-*` for logs
- **Environment issues**: Verify `~/.config/paper-trading.env` is loaded
- **Performance issues**: Check kill-switch status and system resources
