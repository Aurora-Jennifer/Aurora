#!/usr/bin/env python3
"""
Allocator-grade gate checks for walk-forward results.
Tests PSR, MaxDD, and trusted fold requirements.
"""

import json
import sys
from pathlib import Path

import numpy as np


def check_gates(results_file: str = "results/walkforward/results.json"):
    """Check allocator gates on walk-forward results."""

    if not Path(results_file).exists():
        print(f"FAIL: Results file {results_file} not found")
        return False

    with open(results_file) as f:
        results = json.load(f)

    fold_results = results["fold_results"]
    if not fold_results:
        print("FAIL: No fold results found")
        return False

    # Extract metrics
    sharpe_scores = [f["metrics"]["sharpe_nw"] for f in fold_results]
    max_dds = [f["metrics"]["max_dd"] for f in fold_results]
    [f["metrics"]["hit_rate"] for f in fold_results]
    n_trades = [f["n_trades"] for f in fold_results]

    # Gate 1: PSR (Probability of Sharpe Ratio) >= 0.95
    # For a strategy to be trusted, we need high confidence in positive Sharpe
    positive_sharpe_folds = sum(1 for s in sharpe_scores if s > 0)
    total_folds = len(sharpe_scores)
    psr = positive_sharpe_folds / total_folds if total_folds > 0 else 0.0

    print(f"PSR: {psr:.3f} ({positive_sharpe_folds}/{total_folds} folds with positive Sharpe)")

    # Gate 2: MaxDD <= 20%
    max_dd_threshold = 0.20
    max_dd_violations = sum(1 for dd in max_dds if dd < -max_dd_threshold)
    max_dd_gate = max_dd_violations == 0

    print(
        f"MaxDD Gate: {'PASS' if max_dd_gate else 'FAIL'} ({max_dd_violations} folds exceed {max_dd_threshold * 100}% drawdown)"
    )

    # Gate 3: Minimum trades per fold
    min_trades_threshold = 10
    low_trade_folds = sum(1 for n in n_trades if n < min_trades_threshold)
    trades_gate = low_trade_folds == 0

    print(
        f"Trades Gate: {'PASS' if trades_gate else 'FAIL'} ({low_trade_folds} folds have < {min_trades_threshold} trades)"
    )

    # Gate 4: Trusted fold requirements
    # Need at least 6 months of contiguous trusted folds
    trusted_folds = []
    for _i, (sharpe, max_dd, trades) in enumerate(
        zip(sharpe_scores, max_dds, n_trades, strict=False)
    ):
        is_trusted = (
            sharpe > 0  # positive Sharpe
            and max_dd > -max_dd_threshold  # acceptable drawdown
            and trades >= min_trades_threshold  # sufficient trades
        )
        trusted_folds.append(is_trusted)

    # Find longest contiguous trusted sequence
    max_contiguous = 0
    current_contiguous = 0
    for trusted in trusted_folds:
        if trusted:
            current_contiguous += 1
            max_contiguous = max(max_contiguous, current_contiguous)
        else:
            current_contiguous = 0

    # Assuming 3-month test periods, need at least 2 folds for 6 months
    min_contiguous_folds = 2
    contiguous_gate = max_contiguous >= min_contiguous_folds

    print(f"Contiguous Trusted: {max_contiguous} folds (need >= {min_contiguous_folds})")
    print(f"Contiguous Gate: {'PASS' if contiguous_gate else 'FAIL'}")

    # Overall gate status
    all_gates_passed = psr >= 0.95 and max_dd_gate and trades_gate and contiguous_gate

    print(f"\nOverall Gate Status: {'PASS' if all_gates_passed else 'FAIL'}")

    # Production weight calculation
    if all_gates_passed:
        # Calculate production weight based on recent performance
        recent_folds = fold_results[-3:]  # last 3 folds
        recent_sharpe = np.mean([f["metrics"]["sharpe_nw"] for f in recent_folds])
        recent_max_dd = np.mean([f["metrics"]["max_dd"] for f in recent_folds])

        # Simple weight calculation (can be made more sophisticated)
        weight = min(1.0, max(0.0, recent_sharpe * 0.5 + (1 + recent_max_dd) * 0.5))
        print(f"Production Weight: {weight:.3f}")
    else:
        print("Production Weight: 0.0 (gates not passed)")

    return all_gates_passed


if __name__ == "__main__":
    success = check_gates()
    sys.exit(0 if success else 1)
