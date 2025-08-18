#!/usr/bin/env python3
"""
# © 2025 Aurora Analytics — Proprietary
# No redistribution, public posting, or model-training use.
# Internal-only. Ref: AUR-NOTICE-2025-01

Multi-profile, multi-symbol walkforward runner that uses the existing
walkforward framework APIs and writes a markdown report.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from walkforward_framework import (
    build_feature_table,
    gen_walkforward,
    LeakageProofPipeline,
    walkforward_run,
)


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required for this script") from e

    df = yf.download(symbol, start=start, end=end, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"No data for {symbol} in range {start}..{end}")
    # Ensure UTC index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df["Open High Low Close Volume".split()]


def choose_fold_params(n: int, target_folds: int) -> Tuple[int, int, int]:
    # Simple heuristic: 70% train / 30% test over target_folds, rolling by test
    if n < target_folds * 100:
        # small data; keep windows small
        test_len = max(30, n // (target_folds * 3))
    else:
        test_len = max(63, n // (target_folds * 3))
    train_len = max(test_len * 2, 126)
    stride = test_len
    return train_len, test_len, stride


def annualize_cagr(total_return: float, bars: int) -> float:
    if bars <= 0:
        return 0.0
    try:
        return (1.0 + total_return) ** (252.0 / float(bars)) - 1.0
    except Exception:
        return 0.0


def run_for_symbol_profile(
    symbol: str,
    start: str,
    end: str,
    target_folds: int,
    validate_data: bool,
) -> Tuple[List[Dict], List[int]]:
    data = load_data(symbol, start, end)
    X, y, prices = build_feature_table(data, warmup_days=60)
    n = len(X)
    train_len, test_len, stride = choose_fold_params(n, target_folds)
    folds = list(
        gen_walkforward(
            n=n,
            train_len=train_len,
            test_len=test_len,
            stride=stride,
            warmup=0,
            anchored=False,
        )
    )
    if len(folds) > target_folds:
        folds = folds[:target_folds]

    pipeline = LeakageProofPipeline(X, y)
    results = walkforward_run(
        pipeline,
        folds,
        prices,
        performance_mode="RELAXED",
        validate_data=validate_data,
    )
    # Attach fold lengths for CAGR
    fold_test_lengths = [f.test_hi - f.test_lo + 1 for f in folds]
    return results, fold_test_lengths


def make_report(
    out_path: Path,
    profiles: List[str],
    symbols: List[str],
    start: str,
    end: str,
    validate_data: bool,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Walkforward Report — {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n")
    lines.append(f"Date range: {start} → {end}\n")
    lines.append(f"Symbols: {', '.join(symbols)}\n")
    lines.append(f"Profiles: {', '.join(profiles)}\n")
    lines.append(f"DataSanity: {'ON' if validate_data else 'OFF'}\n")

    overall_summary: Dict[str, Dict[str, float]] = {}

    for profile in profiles:
        lines.append(f"\n## Profile: {profile}\n")
        lines.append("| Symbol | Fold | Sharpe | MaxDD | HitRate | Return | MedianHold | CAGR est. | Risk OK |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n")

        sharpe_vals: List[float] = []
        maxdd_vals: List[float] = []

        for symbol in symbols:
            results, fold_lengths = run_for_symbol_profile(
                symbol=symbol,
                start=start,
                end=end,
                target_folds=5,
                validate_data=validate_data,
            )
            for (fold_id, metrics, _), bars in zip(results, fold_lengths):
                sharpe = float(metrics.get("sharpe_nw", 0.0))
                max_dd = float(metrics.get("max_dd", 0.0))
                hit_rate = float(metrics.get("hit_rate", 0.0))
                total_return = float(metrics.get("total_return", 0.0))
                median_hold = float(metrics.get("median_hold", 0.0))
                cagr_est = annualize_cagr(total_return, bars)
                risk_ok = max_dd >= -0.03

                sharpe_vals.append(sharpe)
                maxdd_vals.append(max_dd)

                lines.append(
                    f"| {symbol} | {fold_id} | {sharpe:.3f} | {max_dd:.3f} | {hit_rate:.2f} | {total_return:.3f} | {median_hold:.0f} | {cagr_est:.3f} | {'✅' if risk_ok else '❌'} |\n"
                )

        # Profile summary
        avg_sharpe = float(np.mean(sharpe_vals)) if sharpe_vals else 0.0
        avg_maxdd = float(np.mean(maxdd_vals)) if maxdd_vals else 0.0
        overall_summary[profile] = {"avg_sharpe": avg_sharpe, "avg_maxdd": avg_maxdd}
        lines.append("\n")
        lines.append(f"- Average Sharpe: {avg_sharpe:.3f}\n")
        lines.append(f"- Average MaxDD: {avg_maxdd:.3f}\n")

    # Recommendation
    best_profile = None
    best_score = -1e9
    for p, s in overall_summary.items():
        score = s["avg_sharpe"] - abs(min(0.0, s["avg_maxdd"])) * 2.0
        if score > best_score:
            best_score = score
            best_profile = p
    lines.append("\n### Recommendation\n")
    if best_profile is None:
        lines.append("No clear winner.\n")
    else:
        lines.append(
            f"Recommend profile: {best_profile} (balance of Sharpe and drawdown).\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Multi-profile walkforward report")
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "BTC-USD", "TSLA"],
        help="Symbols to analyze",
    )
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=["risk_low", "risk_balanced", "risk_strict"],
    )
    ap.add_argument("--separate-crypto", action="store_true", help="Run crypto in a separate section")
    ap.add_argument("--validate-data", action="store_true")
    args = ap.parse_args()

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = Path("docs/analysis") / f"walkforward_{ts}.md"

    # Separate crypto vs. non-crypto if requested
    crypto = [s for s in args.symbols if s.endswith("-USD")]
    non_crypto = [s for s in args.symbols if s not in crypto]

    if args.separate_crypto and crypto and non_crypto:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_non = out_file.with_name(out_file.stem + f"_noncrypto_{ts}.md")
        out_cry = out_file.with_name(out_file.stem + f"_crypto_{ts}.md")
        make_report(out_non, args.profiles, non_crypto, args.start, args.end, args.validate_data)
        make_report(out_cry, args.profiles, crypto, args.start, args.end, args.validate_data)
        print(f"Report written: {out_non}\nReport written: {out_cry}")
    else:
        ordered = non_crypto + crypto
        make_report(
            out_path=out_file,
            profiles=args.profiles,
            symbols=ordered,
            start=args.start,
            end=args.end,
            validate_data=args.validate_data,
        )

    print(f"Report written: {out_file}")


if __name__ == "__main__":
    main()


