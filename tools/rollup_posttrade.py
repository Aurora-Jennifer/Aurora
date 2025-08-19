import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REQ_COLS = ["ts", "symbol", "px_theory", "px_fill", "notional"]


def info(msg):
    print(f"[rollup] {msg}", file=sys.stderr)


def load_trades(pattern: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))[-max_files:]
    info(f"found {len(files)} files: {files[-3:] if files else files}")
    rows = []
    for fp in files:
        with open(fp) as f:
            for i, line in enumerate(f, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    info(f"skip {fp}:{i}: json error {e}")
    df = pd.DataFrame(rows)
    info(f"parsed {len(df)} rows")
    return df


def rollup(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "reason": "no_rows"}
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        return {"trades": int(len(df)), "reason": f"missing_cols:{','.join(missing)}"}

    df["dt"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt", "px_theory", "px_fill", "notional"])
    if df.empty:
        return {"trades": 0, "reason": "all_rows_invalid"}

    df["slip_bps_real"] = (df["px_fill"] - df["px_theory"]) / df["px_theory"] * 1e4
    notional = df["notional"].abs().sum()
    out = {
        "trades": int(len(df)),
        "symbols": sorted(df["symbol"].astype(str).unique().tolist()),
        "slippage_bps_median": float(df["slip_bps_real"].median()),
        "slippage_bps_p95": float(df["slip_bps_real"].quantile(0.95)),
        "turnover_notional": float(notional),
        "start": df["dt"].min().isoformat(),
        "end": df["dt"].max().isoformat(),
    }
    return out


def write_reports(summary: dict, out_prefix: str):
    ts = datetime.now(UTC).strftime("%Y%m%d")
    Path("reports").mkdir(exist_ok=True)
    Path("docs/analysis").mkdir(parents=True, exist_ok=True)
    (Path("reports") / f"{out_prefix}_{ts}.json").write_text(json.dumps(summary, indent=2))
    md = [
        f"# {out_prefix.replace('_', ' ').title()} {ts}",
        f"- trades: {summary.get('trades', 0)}",
        f"- symbols: {', '.join(summary.get('symbols', []))}",
        f"- slippage (bps): median {summary.get('slippage_bps_median', 'n/a')}, p95 {summary.get('slippage_bps_p95', 'n/a')}",
        f"- turnover notional: {summary.get('turnover_notional', 0):,.2f}",
        f"- window: {summary.get('start', '?')} → {summary.get('end', '?')}",
        f"- reason: {summary.get('reason', '')}" if "reason" in summary else "",
        "",
    ]
    (Path("docs/analysis") / f"{out_prefix}_{ts}.md").write_text("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/trades/*.jsonl", help="glob for trade logs")
    ap.add_argument("--max-files", type=int, default=10, help="number of latest files to include")
    ap.add_argument("--out-prefix", default="paper_daily", help="report filename prefix")
    args = ap.parse_args()

    df = load_trades(args.glob, args.max_files)
    summary = rollup(df)
    write_reports(summary, args.out_prefix)
    print(json.dumps(summary, indent=2))
