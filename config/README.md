# config

## Structure
- `profiles/` — feature/model/risk profiles (e.g., `golden_linear`, `golden_xgb_v2`)
- `brokers/` — paper/live adapter configs (secrets managed externally)
- `components/` — `costs.yaml`, risk profiles, sizing
- `notifications/` — sinks (e.g., Discord/webhook)
- `data_schema.yaml` — canonical columns + `missing_data` policy

> Paths are relative to repo root. Keep runtime knobs in config, not code.
