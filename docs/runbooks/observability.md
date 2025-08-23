# Observability Runbook

Purpose: Operate and validate run/fold metrics, detect anomalies, and enforce drift checks.

Artifacts:
- Run header: `reports/run.json`
- Fold metrics: `reports/folds/*.json`
- Metrics schema: `reports/metrics.schema.json`

Procedures:
- Validate metrics schema
```bash
python scripts/validate_metrics.py --run reports/run.json --folds reports/folds/
```
- Analyze metrics for warnings (non-blocking)
```bash
python scripts/analyze_metrics.py --config config/base.yaml --reports reports/
```
- Promotion checks (on release)
  - Ensure `run.json.config_hash` matches active config bundle hash
  - Ensure `run.json.data_hash` matches expected snapshot

Anomaly thresholds (from `config/base.yaml`): sharpe_min, drawdown_max, winrate_min, vol_max.
