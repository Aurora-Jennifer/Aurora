# DataSanity Operations Runbook

Purpose: Operate and monitor the DataSanity subsystem (validation, telemetry, canary, engine switch) in CI and runtime.

Entrypoints:
- Engine switch API: `core/data_sanity/api.py: validate_and_repair_with_engine_switch`
- Canary: `scripts/canary_datasanity.py`
- Telemetry output: `artifacts/ds_runs/validation_telemetry.jsonl`

Do-not-touch:
- Public API exports in `core/data_sanity/api.py`
- Validation contracts in `core/data_sanity/main.py`

## Procedures

### 1) Smoke and Walkforward (local)
```bash
make smoke
python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO --datasanity-profile walkforward_smoke
```

### 2) Canary (CI/local)
```bash
python scripts/canary_datasanity.py --profiles walkforward_smoke --verbose
# results saved to artifacts/canary_results.json
```
- Exit code non-zero if regressions (v2-only failures) are detected.

### 3) Telemetry and Metrics
- Telemetry: JSONL at `artifacts/ds_runs/validation_telemetry.jsonl`
- Metrics export via `core.data_sanity.api.export_metrics()`
- Monitor p95 `validation_time` and total repairs per symbol/profile.

### 4) Rotation & Retention
- Policy: size-based rotation 10 MiB x 5 backups (to be implemented).
- Manual cleanup if file exceeds 50 MiB:
```bash
: > artifacts/ds_runs/validation_telemetry.jsonl  # truncate
```

### 5) Regression Triage
1. Re-run canary with `--verbose`.
2. Diff flags/repairs between v1 and v2 in the JSON output.
3. If v2 regression confirmed, set `datasanity.engine: v1` and open a TODO.

### 6) Alerts (planned)
- Add WARN lines from `scripts/analyze_metrics.py` for slow validation or anomaly rates.

## References
- Config: `config/base.yaml` (datasanity.engine, telemetry settings)
- Schema: `config/data_schema.yaml`
- Metrics schema: `reports/metrics.schema.json`
