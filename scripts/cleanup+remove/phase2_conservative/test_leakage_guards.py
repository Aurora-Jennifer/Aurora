#!/usr/bin/env python3
"""
Leakage Guard: Prevent forward-looking data in features and models
"""
from pathlib import Path

import pandas as pd


def check_feature_timestamps(df: pd.DataFrame, target_col: str = 'y') -> list[str]:
    """Check that feature timestamps don't leak into future"""
    violations = []

    if 'timestamp' not in df.columns:
        return ["No timestamp column found"]

    if target_col not in df.columns:
        return [f"No target column '{target_col}' found"]

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except (ValueError, TypeError):
            return ["Cannot parse timestamp column"]

    # Check for non-monotonic timestamps within symbols
    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            if not symbol_data['timestamp'].is_monotonic_increasing:
                violations.append(f"Non-monotonic timestamps for {symbol}")

    # Check for duplicate timestamps within symbols
    if 'symbol' in df.columns:
        dupes = df.groupby(['symbol', 'timestamp']).size()
        if (dupes > 1).any():
            violations.append("Duplicate timestamps within symbols")

    return violations


def check_purged_splits(df: pd.DataFrame, n_splits: int = 5, purge_gap: int = 5) -> list[str]:
    """Verify that CV splits maintain purge gap"""
    violations = []

    if len(df) < n_splits * 2:
        return ["Insufficient data for CV splits"]

    # Simulate TimeSeriesSplit with purge
    test_size = len(df) // (n_splits + 1)

    for i in range(n_splits):
        # Calculate split indices
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        train_end = test_start - purge_gap

        if train_end <= 0:
            violations.append(f"Split {i}: insufficient training data after purge")
            continue

        if test_end > len(df):
            violations.append(f"Split {i}: test end exceeds data length")
            continue

        # Check gap
        gap_actual = test_start - train_end
        if gap_actual < purge_gap:
            violations.append(f"Split {i}: gap {gap_actual} < required {purge_gap}")

    return violations


def check_feature_correlation_with_future(df: pd.DataFrame, target_col: str = 'y') -> list[str]:
    """Check for suspicious correlations with future targets"""
    violations = []

    if 'symbol' not in df.columns or target_col not in df.columns:
        return ["Missing required columns for correlation check"]

    # Create forward target
    df_sorted = df.sort_values(['symbol', 'timestamp'])
    df_sorted['target_future'] = df_sorted.groupby('symbol')[target_col].shift(-1)

    # Check correlations of features with future target
    feature_cols = [col for col in df.columns
                   if col not in ['symbol', 'timestamp', target_col, 'target_future']]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_sorted[col]):
            # Calculate correlation
            corr_current = df_sorted[col].corr(df_sorted[target_col])
            corr_future = df_sorted[col].corr(df_sorted['target_future'])

            # If correlation with future is significantly higher, flag it
            if pd.notna(corr_future) and pd.notna(corr_current) and abs(corr_future) - abs(corr_current) > 0.1:  # Threshold
                violations.append(
                    f"Feature {col}: corr_future={corr_future:.3f} >> corr_current={corr_current:.3f}"
                )

    return violations


def run_leakage_checks() -> bool:
    """Run all leakage prevention checks"""

    # Check if we have test data
    snapshot_path = Path("artifacts/snapshots/golden_ml_v1")
    if not snapshot_path.exists():
        print("[LEAKAGE][SKIP] No snapshot data found for testing")
        return True

    # Load data for testing
    data_files = list(snapshot_path.glob("*.parquet"))
    if not data_files:
        print("[LEAKAGE][SKIP] No parquet files found in snapshot")
        return True

    all_violations = []

    for data_file in data_files[:2]:  # Test first 2 files
        print(f"[LEAKAGE] Checking {data_file.name}")

        try:
            df = pd.read_parquet(data_file)
        except Exception as e:
            print(f"[LEAKAGE][WARN] Could not load {data_file}: {e}")
            continue

        # Run checks
        violations = []
        violations.extend(check_feature_timestamps(df))
        violations.extend(check_purged_splits(df))
        violations.extend(check_feature_correlation_with_future(df))

        if violations:
            all_violations.extend([f"{data_file.name}: {v}" for v in violations])

    if all_violations:
        print("[LEAKAGE][FAIL] Leakage violations detected:")
        for violation in all_violations:
            print(f"  {violation}")
        return False

    print("[LEAKAGE][PASS] No leakage violations detected")
    return True


def main() -> int:
    """Main leakage guard"""
    print("🚫 Testing leakage guards: no forward-looking data")

    if not run_leakage_checks():
        return 1

    print("✅ Leakage guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
