# CI Failures

Purpose: Steps to triage and fix CI.

Entrypoints: , .

Do-not-touch: protected paths unless approved.

- Check smoke summary and pytest.out artifact.
- Run python scripts/multi_walkforward_report.py --smoke --validate-data --log-level INFO
SMOKE OK | folds=1 | symbols=SPY,TSLA,BTC-USD | sharpe=-1.714 maxdd=-1.821
Report written: docs/analysis/walkforward_smoke_20250820_073551.md locally; inspect .
- Fix determinism/timeouts; re-run.
