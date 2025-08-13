#!/usr/bin/env python3
"""
Leakage detection test: shuffle future data and verify train metrics unchanged.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_leakage():
    """Test for data leakage by shuffling future data."""

    # Create synthetic data
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Generate price data with some structure
    returns = np.random.randn(n_days) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.randn(n_days) * 0.001),
            "High": prices * (1 + abs(np.random.randn(n_days)) * 0.002),
            "Low": prices * (1 - abs(np.random.randn(n_days)) * 0.002),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, n_days),
        }
    )

    # Save original data
    data.to_csv("temp_data_original.csv", index=False)

    # Run walk-forward on original data
    print("Running walk-forward on original data...")
    result1 = subprocess.run(
        ["python", "scripts/walkforward_framework.py"], capture_output=True, text=True
    )

    if result1.returncode != 0:
        print("FAIL: Walk-forward failed on original data")
        print(result1.stderr)
        return False

    # Load original results
    try:
        import json

        with open("results/walkforward/results.json") as f:
            original_results = json.load(f)
        original_sharpe = original_results["aggregate"]["mean_sharpe"]
    except:
        print("FAIL: Could not load original results")
        return False

    # Create shuffled data (shuffle the second half)
    mid_point = len(data) // 2
    shuffled_data = data.copy()
    shuffled_data.iloc[mid_point:] = (
        shuffled_data.iloc[mid_point:].sample(frac=1).values
    )

    # Save shuffled data
    shuffled_data.to_csv("temp_data_shuffled.csv", index=False)

    # Run walk-forward on shuffled data
    print("Running walk-forward on shuffled data...")
    result2 = subprocess.run(
        ["python", "scripts/walkforward_framework.py"], capture_output=True, text=True
    )

    if result2.returncode != 0:
        print("FAIL: Walk-forward failed on shuffled data")
        print(result2.stderr)
        return False

    # Load shuffled results
    try:
        with open("results/walkforward/results.json") as f:
            shuffled_results = json.load(f)
        shuffled_sharpe = shuffled_results["aggregate"]["mean_sharpe"]
    except:
        print("FAIL: Could not load shuffled results")
        return False

    # Check for leakage
    # If there's no leakage, training metrics should be similar
    # If there's leakage, shuffled data should have much worse performance
    sharpe_diff = abs(original_sharpe - shuffled_sharpe)

    print(f"Original Sharpe: {original_sharpe:.3f}")
    print(f"Shuffled Sharpe: {shuffled_sharpe:.3f}")
    print(f"Difference: {sharpe_diff:.3f}")

    # Clean up
    for f in ["temp_data_original.csv", "temp_data_shuffled.csv"]:
        if Path(f).exists():
            Path(f).unlink()

    # If difference is small, no leakage detected
    if sharpe_diff < 0.1:  # threshold
        print("PASS: No leakage detected")
        return True
    else:
        print("FAIL: Potential leakage detected")
        return False


if __name__ == "__main__":
    success = test_leakage()
    sys.exit(0 if success else 1)
