# Paper Runner

- Start: `python scripts/paper_runner.py --symbols SPY,TSLA --poll-sec 5 --profile risk_balanced --ntfy`
- Kill-switch: create `kill.flag` to stop gracefully
- Logs:
  - Trades JSONL: `logs/trades/YYYY-MM-DD.jsonl`
  - Meta: `reports/paper_run.meta.json`
  - Provenance: `reports/paper_provenance.json`
- Guards: daily_loss_limit, max_drawdown, max_leverage, max_gross_exposure, max_position_pct
