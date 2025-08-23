# DataSanity Module — Summary

Purpose: Validate and repair OHLCV time series; enforce data contracts before feature or engine use. Provides telemetry and metrics for observability and an engine switch facade for v1→v2 rollout.

Entrypoints:
- `core/data_sanity/api.py` — public API and engine switch facade
- `core/data_sanity/main.py` — v1 validator implementation
- `scripts/canary_datasanity.py` — v1 vs v2 comparison and regression detection

Do-not-touch:
- Public API symbols exported by `core/data_sanity/api.py`
- Validation contracts in `core/data_sanity/main.py`

```mermaid
graph TD
  A[Input DataFrame] --> B[validate_and_repair_with_engine_switch]
  B --> C{datasanity.engine}
  C -- v1 --> D[DataSanityValidator (v1)]
  C -- v2 --> E[DataSanityValidator (v2 placeholder)]
  D --> F[ValidationResult + Cleaned DF]
  E --> F[ValidationResult + Cleaned DF]
  F --> G[emit_validation_telemetry]
  F --> H[bump/export_metrics]
  I[scripts/canary_datasanity.py] --> B
```

### API (selected public)
- `DataSanityValidator`
- `ValidationResult`
- `DataSanityGuard`
- `DataSanityWrapper`
- `validate_market_data`
- `get_data_sanity_wrapper`
- `attach_guard`
- `get_guard`
- `assert_validated`
- `map_ohlcv`
- `enforce_groupwise_time_order`
- `repair_nonfinite_ohlc`
- `coerce_ohlcv_numeric`
- `canonicalize_datetime_index`
- `emit_validation_telemetry`
- `get_telemetry_stats`
- `bump`
- `get_metrics`
- `reset_metrics`
- `export_metrics`
- `validate_and_repair_with_engine_switch`
