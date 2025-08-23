# Comprehensive Metrics Collection — Roadmap (2025-08-22 10:30)
**Prompt:** "Add comprehensive metrics collection (IC, turnover, fill_rate, latency)"

## Context
- Paper trading readiness checklist shows comprehensive metrics as next priority
- Need observability into performance: IC, turnover, fill_rate, latency, memory_peak
- Current logging is basic; missing key performance indicators
- Export parity tests completed successfully

## Plan (now)
1) Implement IC calculation (Information Coefficient)
2) Add turnover tracking in paper broker
3) Wire latency and memory monitoring
4) Create metrics dashboard/visualization
5) Add metrics validation to CI
6) Ensure metrics logged in paper trading loop

## Success criteria
- All key metrics logged: IC, turnover, fill_rate, latency, memory_peak
- Metrics format: JSON with run_id, timestamp, values
- Metrics collection overhead ≤5ms
- Metrics logged in paper trading loop
- Metrics within expected ranges in golden smoke tests
