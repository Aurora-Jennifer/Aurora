# Alpaca Paper Trading Service Dependencies

## Main Entry Point
- `scripts/live/enhanced_profit_trader_optimized.py` (if exists)
- `scripts/paper_runner.py` (alternative entry point)

## Core Trading Engine
- `core/engine/paper.py` - Main paper trading engine
- `brokers/paper.py` - Paper broker implementation
- `brokers/interface.py` - Broker interface

## Configuration Files
- `config/base.yaml` - Base configuration
- `config/paper_trading_config.json` - Paper trading config
- `config/enhanced_paper_trading_config.json` - Enhanced config
- `config/enhanced_paper_trading_config_unified.json` - Unified config
- `config/paper_config.json` - Paper config
- `config/paper-trading.env` - Environment variables
- `config/profiles/paper_strict.yaml` - Strict profile
- `config/models.yaml` - Model registry

## Data Providers
- `scripts/data/alpaca_batch_optimized.py` - Optimized Alpaca batch fetcher
- `scripts/data/alpaca_batch.py` - Alpaca batch fetcher
- `scripts/data/providers/alpaca_provider.py` - Alpaca data provider (if exists)
- `brokers/data_provider.py` - IBKR data provider interface
- `brokers/ibkr_broker.py` - IBKR broker config

## Core Components
- `core/enhanced_logging.py` - TradingLogger
- `core/feature_reweighter.py` - FeatureReweighter, AdaptiveFeatureEngine
- `core/notifications.py` - DiscordConfig, DiscordNotifier
- `core/performance.py` - GrowthTargetCalculator
- `core/regime_detector.py` - RegimeDetector
- `core/strategy_selector.py` - StrategySelector
- `core/utils.py` - ensure_directories
- `core/data_sanity.py` - get_data_sanity_wrapper
- `core/risk/guardrails.py` - RiskGuardrails (optional)
- `core/telemetry/snapshot.py` - TelemetrySnapshot (optional)

## Strategies
- `strategies/factory.py` - strategy_factory
- `strategies/regime_aware_ensemble.py` - RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams

## ML/Model Components
- `ml/model_interface.py` - ModelSpec
- `ml/registry.py` - load_model
- `ml/runtime.py` - build_features, infer_weights, compute_turnover, detect_weight_spikes, set_seeds

## Utilities
- `utils/ops_runtime.py` - kill_switch, notify_ntfy
- `tools/provenance.py` - write_provenance

## CLI
- `cli/paper.py` - CLI interface (if exists)

## Tests
- `tests/paper/test_paper_runner_smoke.py` - Smoke tests
- `test_paper_trading_config.json` - Test config

## Documentation
- `docs/runbooks/paper.md` - Paper trading runbook
- `docs/analysis/paper_daily_*.md` - Analysis docs

## External Dependencies (Python packages)
- `alpaca-trade-api` or `alpaca-py` - Alpaca API client
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `yfinance` - Yahoo Finance fallback data
- `yaml` - Configuration parsing
- `requests` - HTTP requests for Alpaca API

## Data Files (Runtime)
- `data/smoke_cache/*.parquet` - Cached market data
- `reports/paper_run.meta.json` - Run metadata
- `reports/paper_provenance.json` - Provenance tracking
- `reports/runner_state.json` - State persistence
- `logs/trades/*.jsonl` - Trade logs

## Environment Variables
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `ALPACA_BASE_URL` - Alpaca API base URL (optional)
- `GITHUB_REPOSITORY` - For issue creation (optional)
- `GITHUB_TOKEN` or `GH_TOKEN` - For issue creation (optional)

