import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.metrics.stats import max_drawdown, sharpe_newey_west
from core.walk.folds import gen_walkforward
from core.walk.pipeline import Pipeline
from core.walk.run import (
    calculate_weighted_metrics,
    contiguous_trusted_span,
    fold_weight,
    gate_fold,
    run_fold,
    stitch_equity,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--train", type=int, default=252)
    ap.add_argument("--test", type=int, default=63)
    ap.add_argument("--stride", type=int, default=63)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--min-live-months", type=int, default=6)
    ap.add_argument("--output-dir", default="artifacts")
    args = ap.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pl.read_parquet(args.parquet).sort("ts")
    close = df["close"].to_numpy()
    X = df.select(["ret1", "ma20", "vol20", "zscore20"]).to_numpy()
    y = df["ret1"].to_numpy()

    # Generate fold dates for contiguous trusted calculation
    dates = df["ts"].to_numpy()
    fold_dates = []

    n = df.height
    folds = list(
        gen_walkforward(
            n, args.train, args.test, args.stride, warmup=20, anchored=args.anchored
        )
    )
    p = Pipeline(X, y)

    # Run folds
    out = []
    pnls = []

    print(f"Running {len(folds)} folds...")
    print("=" * 80)

    for f in folds:
        # Calculate fold dates
        start_date = dates[f.test_lo] if f.test_lo < len(dates) else dates[0]
        end_date = dates[f.test_hi] if f.test_hi < len(dates) else dates[-1]
        fold_dates.append((start_date, end_date))

        # Run fold
        m, pnl, sig = run_fold(p, close, X, y, f)
        m = gate_fold(
            m, gates={"min_days": 30, "min_trades": 5, "psr_min": 0.7, "max_dd": 0.2}
        )  # Relaxed min_trades from 10 to 5
        out.append(m)
        pnls.append(pnl)

        # Print fold results
        print(
            f"Fold {m['fold_id']:02d} | SharpeNW={m['sharpe_nw']:.2f} | PSR={m['psr']:.2f} | WR={m['win_rate']:.2f} | DD={m['max_dd']:.3f} | Trusted={m['trusted']}"
        )

    # Calculate weighted metrics
    weighted_metrics = calculate_weighted_metrics(out, pnls)

    # Calculate contiguous trusted span
    has_sufficient_span, best_span = contiguous_trusted_span(
        out, fold_dates, args.min_live_months
    )

    # Stitch equity curve
    stitched_equity = stitch_equity(pnls)

    # Save artifacts
    np.save(output_dir / "stitched_equity.npy", stitched_equity)

    # Add weights to metrics
    weights = [fold_weight(m) for m in out]
    for m, w in zip(out, weights):
        m["weight"] = float(w)

    # Add weighted metrics to output
    out.append({"aggregate": weighted_metrics})

    with open(output_dir / "artifacts_walk.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(stitched_equity) > 1:
        stitched_returns = np.diff(stitched_equity, prepend=stitched_equity[0])
        print(f"Stitched SharpeNW: {sharpe_newey_west(stitched_returns):.3f}")
        print(f"Stitched MaxDD: {max_drawdown(stitched_equity):.3f}")

    print(f"Weighted Sharpe: {weighted_metrics.get('weighted_sharpe', 0):.3f}")
    print(f"Weighted Sortino: {weighted_metrics.get('weighted_sortino', 0):.3f}")
    print(f"Weighted MaxDD: {weighted_metrics.get('weighted_max_dd', 0):.3f}")
    print(f"Weighted Win Rate: {weighted_metrics.get('weighted_win_rate', 0):.3f}")
    print(
        f"Trusted Folds: {weighted_metrics.get('trusted_folds', 0)}/{weighted_metrics.get('total_folds', 0)}"
    )
    print(f"Contiguous Trusted Span: {best_span.days} days")
    print(f"Sufficient Live Months ({args.min_live_months}): {has_sufficient_span}")

    # Payoff anomaly analysis
    payoff_anomalies = weighted_metrics.get("payoff_anomalies", [])
    if payoff_anomalies:
        print(f"\n🚨 PAYOFF ANOMALIES DETECTED ({len(payoff_anomalies)} folds):")
        for anomaly in payoff_anomalies:
            print(
                f"  Fold {anomaly['fold_id']}: WR={anomaly['win_rate']:.3f}, Sharpe={anomaly['sharpe_nw']:.3f}, Trades={anomaly['n_trades']}, Turnover={anomaly['turnover']:.3f}"
            )
        print(
            "  → High win rate with negative Sharpe suggests: tiny wins, large losses, or cost domination"
        )

    # Regime performance breakdown
    regime_stats = {}
    for m in out[:-1]:  # Exclude aggregate
        regime = m.get("regime", "unknown")
        if regime not in regime_stats:
            regime_stats[regime] = {
                "count": 0,
                "sharpe": [],
                "win_rate": [],
                "max_dd": [],
            }
        regime_stats[regime]["count"] += 1
        regime_stats[regime]["sharpe"].append(m["sharpe_nw"])
        regime_stats[regime]["win_rate"].append(m["win_rate"])
        regime_stats[regime]["max_dd"].append(m["max_dd"])

    print("\nREGIME PERFORMANCE:")
    for regime, stats in regime_stats.items():
        if stats["count"] > 0:
            avg_sharpe = np.mean(stats["sharpe"])
            avg_win_rate = np.mean(stats["win_rate"])
            avg_max_dd = np.mean(stats["max_dd"])
            print(
                f"  {regime}: {stats['count']} folds | Sharpe={avg_sharpe:.3f} | WR={avg_win_rate:.3f} | DD={avg_max_dd:.3f}"
            )

    # Turnover analysis
    turnovers = [m.get("turnover", 0) for m in out[:-1]]
    if turnovers:
        print("\nTURNOVER ANALYSIS:")
        print(f"  Median: {np.median(turnovers):.2f}")
        print(f"  Mean: {np.mean(turnovers):.2f}")
        print(f"  Max: {np.max(turnovers):.2f}")
        print(f"  High turnover folds (>100): {sum(1 for t in turnovers if t > 100)}")

    # Print dependency versions
    print("\nDependencies:")
    try:
        import numba

        print(f"  numba: {numba.__version__}")
    except:
        print("  numba: not available")

    try:
        print(f"  polars: {pl.__version__}")
    except:
        print("  polars: version unknown")

    print(f"  numpy: {np.__version__}")

    # Save config
    config = {
        "parquet_file": args.parquet,
        "train_days": args.train,
        "test_days": args.test,
        "stride_days": args.stride,
        "anchored": args.anchored,
        "min_live_months": args.min_live_months,
        "run_timestamp": datetime.now().isoformat(),
        "total_folds": len(folds),
        "data_shape": X.shape,
        "date_range": [str(dates[0]), str(dates[-1])],
    }

    with open(output_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"\nArtifacts saved to: {output_dir}")
    print("  - artifacts_walk.json: Per-fold metrics with weights")
    print("  - stitched_equity.npy: Continuous equity curve")
    print("  - config.json: Run configuration")


if __name__ == "__main__":
    main()
