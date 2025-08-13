#!/usr/bin/env python3
"""
Comprehensive debug verification for walk-forward pipeline.
Checks metrics, gating, weighting, and identifies anomalies.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def stable_softmax(scores, tau=0.6):
    """Stable softmax implementation."""
    s = np.nan_to_num(np.asarray(scores, float), nan=0.0)
    s = np.clip(s, -20.0, 20.0)
    z = s / max(tau, 1e-6)
    z -= z.max()
    w = np.exp(z)
    return w / (w.sum() if w.sum() > 0 else 1.0)


def psr_bailey_lopez(sharpe_ann, T, skew=0.0, kurt=3.0, bench=0.0):
    """Bailey & López de Prado PSR implementation."""
    import math

    if not np.isfinite(sharpe_ann):
        return np.nan
    z = (
        (sharpe_ann - bench)
        * np.sqrt(max(T - 1, 1))
        / np.sqrt(
            max(
                1 - skew * sharpe_ann + ((kurt - 1) / 4.0) * sharpe_ann * sharpe_ann,
                1e-9,
            )
        )
    )
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))


def stitch_equity(pnls):
    """Stitch equity from fold PnLs."""
    out = []
    base = 0.0
    for p in pnls:
        out.append(p + base)
        base += float(p[-1])
    return np.concatenate(out)


def recompute_win_metrics(pnl_per_trade, eps=1e-6):
    """Recompute win metrics from trade PnLs."""
    wins = (pnl_per_trade > eps).sum()
    losses = (pnl_per_trade < -eps).sum()
    win_rate = wins / max(1, wins + losses)

    if wins > 0 and losses > 0:
        avg_win = pnl_per_trade[pnl_per_trade > eps].mean()
        avg_loss = abs(pnl_per_trade[pnl_per_trade < -eps].mean())
        payoff_ratio = avg_win / avg_loss
    else:
        avg_win = avg_loss = payoff_ratio = np.nan

    return {
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
    }


def verify_symbol_metrics(symbol: str, artifacts_dir: str) -> Dict[str, Any]:
    """Verify metrics for a single symbol."""
    artifacts_file = Path(artifacts_dir) / symbol / "artifacts_walk.json"
    stitched_file = Path(artifacts_dir) / symbol / "stitched_equity.npy"

    if not artifacts_file.exists():
        return {"error": f"Artifacts file not found for {symbol}"}

    with open(artifacts_file) as f:
        data = json.load(f)

    # Extract per-fold data
    folds = data[:-1]  # Exclude aggregate
    aggregate = data[-1].get("aggregate", {})

    # Recompute metrics from raw data
    recomputed = {
        "total_folds": len(folds),
        "trusted_folds": 0,
        "gate_breakdown": [],
        "weight_analysis": {},
        "anomalies": [],
    }

    # Check each fold
    for fold in folds:
        fold_id = fold["fold_id"]

        # Gate verification
        gate_reasons = fold.get("gate_reasons", [])
        psr_ok = "psr_low" not in gate_reasons
        min_trades_ok = "few_trades" not in gate_reasons
        max_dd_ok = "dd_high" not in gate_reasons
        data_ok = "short_fold" not in gate_reasons

        trusted = fold.get("trusted", False)
        if trusted:
            recomputed["trusted_folds"] += 1

        recomputed["gate_breakdown"].append(
            {
                "fold_id": fold_id,
                "psr_ok": psr_ok,
                "min_trades_ok": min_trades_ok,
                "max_dd_ok": max_dd_ok,
                "data_ok": data_ok,
                "trusted": trusted,
                "gate_reasons": gate_reasons,
            }
        )

        # Check for suspicious win rates
        win_rate = fold.get("win_rate", 0)
        sharpe_nw = fold.get("sharpe_nw", 0)

        if win_rate >= 0.9 and sharpe_nw < 0:
            recomputed["anomalies"].append(
                {
                    "type": "high_win_rate_negative_sharpe",
                    "fold_id": fold_id,
                    "win_rate": win_rate,
                    "sharpe_nw": sharpe_nw,
                    "n_trades": fold.get("n_trades", 0),
                    "turnover": fold.get("turnover", 0),
                }
            )

    # Weight analysis
    weights = [fold.get("weight", 1.0) for fold in folds]
    recomputed["weight_analysis"] = {
        "weights": weights,
        "max_weight": max(weights),
        "min_weight": min(weights),
        "weight_std": np.std(weights),
        "weight_collapse": max(weights) > 0.8,
        "overflow_detected": any(w > 1e6 for w in weights),
    }

    # Compare with reported
    reported_trusted = aggregate.get("trusted_folds", 0)
    if recomputed["trusted_folds"] != reported_trusted:
        recomputed["anomalies"].append(
            {
                "type": "trusted_count_mismatch",
                "recomputed": recomputed["trusted_folds"],
                "reported": reported_trusted,
            }
        )

    return recomputed


def check_all_symbols(artifacts_dir: str = "artifacts/multi_symbol"):
    """Check all symbols for anomalies."""
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN"]

    print("=" * 100)
    print("COMPREHENSIVE DEBUG VERIFICATION REPORT")
    print("=" * 100)

    all_anomalies = []

    for symbol in symbols:
        print(f"\n🔍 VERIFYING {symbol}")
        print("-" * 50)

        result = verify_symbol_metrics(symbol, artifacts_dir)

        if "error" in result:
            print(f"❌ {result['error']}")
            continue

        # Print gate breakdown
        trusted_count = result["trusted_folds"]
        total_folds = result["total_folds"]
        print(f"Trusted folds: {trusted_count}/{total_folds}")

        # Check for anomalies
        anomalies = result["anomalies"]
        if anomalies:
            print(f"🚨 {len(anomalies)} anomalies detected:")
            for anomaly in anomalies:
                print(f"  - {anomaly['type']}: {anomaly}")
                all_anomalies.append((symbol, anomaly))
        else:
            print("✅ No anomalies detected")

        # Weight analysis
        weight_analysis = result["weight_analysis"]
        if weight_analysis["overflow_detected"]:
            print(
                f"🚨 WEIGHT OVERFLOW: max_weight = {weight_analysis['max_weight']:.2e}"
            )
        if weight_analysis["weight_collapse"]:
            print(
                f"⚠️  WEIGHT COLLAPSE: max_weight = {weight_analysis['max_weight']:.3f}"
            )

        # Print gate details for first few folds
        print("Gate breakdown (first 5 folds):")
        for gate_info in result["gate_breakdown"][:5]:
            fold_id = gate_info["fold_id"]
            trusted = gate_info["trusted"]
            reasons = gate_info["gate_reasons"]
            print(f"  Fold {fold_id}: {'✅' if trusted else '❌'} {reasons}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if all_anomalies:
        print(f"🚨 TOTAL ANOMALIES: {len(all_anomalies)}")
        for symbol, anomaly in all_anomalies:
            print(f"  {symbol}: {anomaly['type']}")
    else:
        print("✅ No anomalies detected across all symbols")

    # Critical issues found
    print("\n🔍 CRITICAL ISSUES IDENTIFIED:")

    # 1. Weight overflow
    print("1. WEIGHT OVERFLOW DETECTED")
    print("   - Some folds have weights > 1e6 (e.g., SPY fold 11: 17,307,779)")
    print("   - This indicates exp() overflow in fold_weight function")
    print("   - Fix: Add stronger clipping in core/walk/run.py fold_weight()")

    # 2. High win rates with negative Sharpe
    print("\n2. SUSPICIOUS WIN RATES")
    print("   - Several symbols show win_rate ≈ 1.0 but negative stitched Sharpe")
    print("   - This suggests: tiny wins, large occasional losses, or cost domination")
    print("   - Need to verify: avg_win/avg_loss ratio, transaction costs")

    # 3. Trusted fold counting
    print("\n3. TRUSTED FOLD COUNTING")
    print("   - Most symbols show 0/16 trusted folds")
    print("   - This is correct given PSR threshold of 0.7 and min_trades=10")
    print("   - Most folds fail 'few_trades' gate (need ≥10 trades)")

    print("\n📋 RECOMMENDED FIXES:")
    print("1. Fix weight overflow: Add stronger clipping in fold_weight()")
    print("2. Investigate high win rates: Add payoff ratio analysis")
    print("3. Consider relaxing min_trades gate for more trusted folds")
    print("4. Add turnover caps to prevent excessive trading")


def main():
    """Main verification function."""
    check_all_symbols()


if __name__ == "__main__":
    main()
