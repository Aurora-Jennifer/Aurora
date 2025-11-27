<!--
ALPACA Paper Trading Service

Copyright (C) 2024  <name of author>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

# ALPACA Paper Trading Service - Complete File List

## Files Packaged (33 files)

### Scripts
- `scripts/paper_runner.py`
- `scripts/data/alpaca_batch_optimized.py`
- `scripts/data/alpaca_batch.py`

### Core Engine
- `core/engine/paper.py`

### Core Components
- `core/enhanced_logging.py`
- `core/feature_reweighter.py`
- `core/notifications.py`
- `core/performance.py`
- `core/regime_detector.py`
- `core/strategy_selector.py`
- `core/utils.py`
- `core/data_sanity.py`
- `core/risk/guardrails.py` (optional)
- `core/telemetry/snapshot.py` (optional)

### Brokers
- `brokers/paper.py`
- `brokers/interface.py`
- `brokers/data_provider.py`
- `brokers/ibkr_broker.py`

### Strategies
- `strategies/factory.py`
- `strategies/regime_aware_ensemble.py`

### ML/Model
- `ml/model_interface.py`
- `ml/registry.py`
- `ml/runtime.py`

### Utilities
- `utils/ops_runtime.py`
- `tools/provenance.py`

### CLI
- `cli/paper.py`

### Configuration
- `config/base.yaml`
- `config/models.yaml`
- `config/paper_trading_config.json`
- `config/paper_config.json`
- `config/enhanced_paper_trading_config.json`
- `config/enhanced_paper_trading_config_unified.json`
- `config/profiles/paper_strict.yaml`
- `config/paper-trading.env`

## Missing Files (Not Found)
- `scripts/live/enhanced_profit_trader_optimized.py` - Main entry point (not found in repo)
- `scripts/data/providers/alpaca_provider.py` - Alpaca data provider (if exists)

## Notes
- All files are copies; originals remain in their original locations
- Import paths may need adjustment if running from this folder
- `__init__.py` files added to make directories Python packages

