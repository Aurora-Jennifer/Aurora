# Changes

## Actions
- `config/base.yaml`: Added `datasanity` configuration section with engine selection and telemetry settings
- `core/data_sanity/config.py`: Created centralized configuration management with dot-separated path access
- `core/data_sanity/telemetry.py`: Implemented JSONL telemetry system for validation runs
- `core/data_sanity/metrics.py`: Added metrics collection with error budget tracking
- `core/data_sanity/api.py`: Added engine switch facade and telemetry imports
- `core/data_sanity/main.py`: Updated docstring to indicate v1 implementation
- `scripts/canary_datasanity.py`: Created canary testing script for v1/v2 comparison
- `tests/datasanity/test_engine_switch.py`: Added comprehensive tests for engine switch functionality
- `Makefile`: Added `canary` target for running canary tests
- `.pre-commit-config.yaml`: Added DataSanity smoke test hook

## Commands run
```bash
# Test engine switch functionality
python -c "from core.data_sanity import validate_and_repair_with_engine_switch; print('Engine switch import successful')"

# Test configuration loading
python -c "from core.data_sanity.config import get_cfg; print('Config loaded:', get_cfg('datasanity.engine', 'v1'))"

# Test engine switch tests
pytest tests/datasanity/test_engine_switch.py::test_engine_switch_basic -v

# Test canary script
python scripts/canary_datasanity.py --profiles walkforward_smoke --verbose
```

## Key implementations
1. **Engine Switch Facade**: `validate_and_repair_with_engine_switch()` routes validation requests based on `datasanity.engine` config
2. **Telemetry System**: Emits structured JSON logs to `artifacts/ds_runs/validation_telemetry.jsonl` with run metadata
3. **Metrics Collection**: Tracks validation counters, error codes, and severity levels for monitoring
4. **Canary Testing**: Compares v1 vs v2 outcomes and detects regressions with detailed reporting
5. **Configuration Management**: Centralized config access with fallback defaults and error handling
6. **Test Coverage**: Comprehensive tests for engine switch, telemetry, and error handling scenarios

## Configuration changes
- Added `datasanity.engine: v1` (default) to `config/base.yaml`
- Added `datasanity.telemetry.enabled: true` for telemetry control
- Added `datasanity.telemetry.output_dir: artifacts/ds_runs` for log location
