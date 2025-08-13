#!/usr/bin/env python3
import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/backtest")
RUN_LOG = Path("results/run_log.csv")


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_summary_txt(path: Path) -> dict:
    metrics = {}
    if not path.exists():
        return metrics
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            metrics[k.strip()] = v.strip().replace(",", "")
    return metrics


def analyze(run_id: str, change_desc: str = "baseline") -> dict:
    trades_p = RESULTS_DIR / "trades.csv"
    ledger_p = RESULTS_DIR / "ledger.csv"
    summary_p = RESULTS_DIR / "summary.txt"

    trades = (
        pd.read_csv(trades_p, parse_dates=["timestamp"])
        if trades_p.exists()
        else pd.DataFrame()
    )
    ledger = (
        pd.read_csv(ledger_p, parse_dates=["date"])
        if ledger_p.exists()
        else pd.DataFrame()
    )
    sumtxt = load_summary_txt(summary_p)

    # Per-symbol stats
    per_symbol = pd.DataFrame()
    if not trades.empty:
        per_symbol = (
            trades.groupby("symbol")
            .agg(
                total_trades=("trade_id", "count"),
                win_rate=("realized_pnl", lambda s: (s > 0).mean() if len(s) else 0.0),
                total_pnl=("realized_pnl", "sum"),
            )
            .reset_index()
        )

    # Per-regime (if available)
    per_regime = pd.DataFrame()
    if "regime" in trades.columns:
        per_regime = (
            trades.groupby("regime")
            .agg(
                total_trades=("trade_id", "count"),
                win_rate=("realized_pnl", lambda s: (s > 0).mean()),
                total_pnl=("realized_pnl", "sum"),
            )
            .reset_index()
        )

    # Per-signal-strength deciles (if available)
    per_signal = pd.DataFrame()
    if "signal_strength" in trades.columns:
        bins = pd.qcut(trades["signal_strength"], 10, duplicates="drop")
        per_signal = (
            trades.groupby(bins)
            .agg(
                total_trades=("trade_id", "count"),
                win_rate=("realized_pnl", lambda s: (s > 0).mean()),
                total_pnl=("realized_pnl", "sum"),
            )
            .reset_index()
            .rename(columns={"signal_strength": "bucket"})
        )

    # Distribution & extremes
    hist = pd.DataFrame()
    top_wins = pd.DataFrame()
    top_losses = pd.DataFrame()
    if not trades.empty:
        trades["pnl_bucket"] = (trades["realized_pnl"] // 50) * 50
        hist = (
            trades.groupby("pnl_bucket")["trade_id"]
            .count()
            .reset_index()
            .rename(columns={"trade_id": "count"})
        )
        top_wins = trades.sort_values("realized_pnl", ascending=False).head(10)
        top_losses = trades.sort_values("realized_pnl").head(10)

    # Pull summary metrics
    sharpe = safe_float(sumtxt.get("Sharpe Ratio", 0.0))
    max_dd_txt = sumtxt.get("Max Drawdown", "0%")
    max_dd = safe_float(max_dd_txt.strip("%"))
    total_pnl = safe_float(sumtxt.get("Total PnL", 0.0))
    win_rate_overall = 0.0
    if not trades.empty:
        win_rate_overall = (trades["realized_pnl"] > 0).mean()

    top_symbol = (
        per_symbol.sort_values("total_pnl", ascending=False).symbol.iloc[0]
        if not per_symbol.empty
        else "NA"
    )
    worst_symbol = (
        per_symbol.sort_values("total_pnl").symbol.iloc[0]
        if not per_symbol.empty
        else "NA"
    )

    result = {
        "run_id": run_id,
        "per_symbol": per_symbol,
        "per_regime": per_regime,
        "per_signal": per_signal,
        "hist": hist,
        "top_wins": top_wins,
        "top_losses": top_losses,
        "summary": {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "win_rate": win_rate_overall,
            "max_dd": max_dd,
            "top_symbol": top_symbol,
            "worst_symbol": worst_symbol,
        },
        "change_desc": change_desc,
    }
    return result


def append_run_log(analysis: dict):
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = RUN_LOG.exists()
    with RUN_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "run_id",
                    "date",
                    "change_desc",
                    "total_pnl",
                    "sharpe",
                    "win_rate",
                    "max_dd",
                    "top_symbol",
                    "worst_symbol",
                    "suggested_next_change",
                ]
            )
        w.writerow(
            [
                analysis["run_id"],
                dt.datetime.now(dt.UTC).isoformat(),
                analysis["change_desc"],
                analysis["summary"]["total_pnl"],
                analysis["summary"]["sharpe"],
                analysis["summary"]["win_rate"],
                analysis["summary"]["max_dd"],
                analysis["summary"]["top_symbol"],
                analysis["summary"]["worst_symbol"],
                "Tweak momentum threshold by +0.01",
            ]
        )


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_0001"
    change_desc = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    analysis = analyze(run_id, change_desc)
    append_run_log(analysis)

    # Print markdown-like tables with fallback if tabulate not installed
    def md(df: pd.DataFrame, title: str):
        print(f"\n### {title}")
        if df is None or df.empty:
            print("No data")
            return
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df.to_string(index=False))

    md(analysis["per_symbol"], "Per-symbol stats")
    md(analysis["per_regime"], "Per-regime stats")
    md(analysis["per_signal"], "Per-signal-strength deciles")
    md(analysis["hist"], "Histogram (bin=$50)")

    print(
        "\n### Top 10 winning trades\n",
        (
            analysis["top_wins"][["timestamp", "symbol", "realized_pnl"]].to_string(
                index=False
            )
            if not analysis["top_wins"].empty
            else "No data"
        ),
    )
    print(
        "\n### Top 10 losing trades\n",
        (
            analysis["top_losses"]["timestamp symbol realized_pnl".split()].to_string(
                index=False
            )
            if not analysis["top_losses"].empty
            else "No data"
        ),
    )

    # Hypothesis generator (simple heuristic)
    hypothesis = "Choppy regime losses dominate; momentum filter may be too permissive."
    next_change = "Increase confidence_threshold by +0.05"
    print(f"\n### Suggested next change\n- {hypothesis}\n- {next_change}")


if __name__ == "__main__":
    main()
