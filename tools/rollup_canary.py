import json
import glob
from pathlib import Path
from datetime import datetime, timezone


def roll() -> None:
    files = sorted(glob.glob("logs/canary/*.jsonl"))
    trades = 0
    symbols: set[str] = set()
    for fp in files[-3:]:
        with open(fp) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                trades += 1
                s = obj.get("symbol")
                if s:
                    symbols.add(s)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = {"trades": trades, "symbols": sorted(symbols)}
    Path("reports").mkdir(exist_ok=True)
    Path("docs/analysis").mkdir(parents=True, exist_ok=True)
    Path(f"reports/canary_daily_{ts}.json").write_text(json.dumps(out, indent=2))
    Path(f"docs/analysis/canary_daily_{ts}.md").write_text(
        f"# Canary Daily {ts}\n- trades: {trades}\n- symbols: {', '.join(out['symbols'])}\n"
    )


if __name__ == "__main__":
    roll()


