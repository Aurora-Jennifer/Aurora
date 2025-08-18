import json, glob
from pathlib import Path
from datetime import datetime, timezone

def roll():
    files = sorted(glob.glob("logs/trades/*.jsonl"))
    trades = 0; syms=set()
    for fp in files[-3:]: # Process last 3 days of logs
        for line in open(fp):
            obj = json.loads(line)
            trades += 1
            syms.add(obj.get("symbol"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = {"trades": trades, "symbols": sorted(list(syms))}
    Path("reports").mkdir(exist_ok=True)
    Path("docs/analysis").mkdir(parents=True, exist_ok=True)
    Path(f"reports/live_daily_{ts}.json").write_text(json.dumps(out, indent=2))
    Path(f"docs/analysis/live_daily_{ts}.md").write_text(
        f"# Live Daily {ts}\n- trades: {trades}\n- symbols: {', '.join(out['symbols'])}\n"
    )

if __name__ == "__main__":
    roll()
