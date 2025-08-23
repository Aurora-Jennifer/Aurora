#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from core.ml.build_features import build_matrix


def load_feature_config():
    """Load feature configuration"""
    import yaml
    with open("config/features.yaml") as f:
        return yaml.safe_load(f)


def compute_ic_by_period(df):
    """Compute IC for each asset separately (time-series correlation)"""
    correlations = []
    for asset in df['asset'].unique():
        asset_data = df[df['asset'] == asset]
        if len(asset_data) >= 20:  # Need more points for time-series correlation
            corr, _ = spearmanr(asset_data['y_true'], asset_data['y_pred'])
            if not np.isnan(corr):
                correlations.append(corr)
    return np.array(correlations)


def paired_t_test(ic_baseline, ic_ablated):
    """Perform paired t-test on IC differences"""
    if len(ic_baseline) != len(ic_ablated):
        # If different lengths, use the shorter one
        min_len = min(len(ic_baseline), len(ic_ablated))
        ic_baseline = ic_baseline[:min_len]
        ic_ablated = ic_ablated[:min_len]

    if len(ic_baseline) < 2:
        return 1.0  # Cannot compute t-test with < 2 observations

    # Paired t-test on IC differences
    t_stat, p_value = stats.ttest_rel(ic_baseline, ic_ablated)
    return p_value


def _resolve_cv(profile_cfg):
    """Prefer folds from search_history.json; fallback to profile config."""
    from sklearn.model_selection import TimeSeriesSplit
    n_splits = profile_cfg.get("cv_n_splits", 3)
    gap = profile_cfg.get("purge_gap_bars", 10)
    test_size = profile_cfg.get("cv_test_size", 256)
    hist = "reports/experiments/search_history.json"
    if os.path.exists(hist):
        try:
            with open(hist) as f:
                h = json.load(f)
            cv = h.get("meta", {}).get("cv", {})
            n_splits = cv.get("n_splits", n_splits)
            gap = cv.get("gap", gap)
            test_size = cv.get("test_size", test_size)
        except Exception:
            pass
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)


def run_ablation(profile="golden_linear"):
    """Run feature group ablation study"""
    # Load configurations
    import yaml
    with open("config/train_profiles.yaml") as f:
        train_config = yaml.safe_load(f)

    feature_config = load_feature_config()
    prof_cfg = train_config["train"]["profiles"][profile]

    # Load data
    frames, snapmeta = _load_snapshot(prof_cfg["data_snapshot"], prof_cfg["symbols"])

    # Build baseline features
    print("Building baseline features...")
    mats = [build_matrix(frames[s], prof_cfg["horizon_bars"]) for s in prof_cfg["symbols"]]
    X_baseline = pd.concat([m[0] for m in mats]).to_numpy(dtype=np.float64)
    y_baseline = pd.concat([m[1] for m in mats]).to_numpy(dtype=np.float64)

    # Train baseline model
    print("Training baseline model...")
    baseline_ic = train_and_evaluate(X_baseline, y_baseline, prof_cfg, tscv=_resolve_cv(prof_cfg))

    # Run ablation for each group
    results = []
    groups = feature_config["groups"]

    for group in groups:
        print(f"\nAblating group: {group}")

        # Build features excluding this group
        mats_ablated = [build_matrix(frames[s], prof_cfg["horizon_bars"], exclude_tags=[group])
                       for s in prof_cfg["symbols"]]
        X_ablated = pd.concat([m[0] for m in mats_ablated]).to_numpy(dtype=np.float64)
        y_ablated = pd.concat([m[1] for m in mats_ablated]).to_numpy(dtype=np.float64)

        # Train ablated model
        ablated_ic = train_and_evaluate(X_ablated, y_ablated, prof_cfg, tscv=_resolve_cv(prof_cfg))

        # Compute IC difference (ablated - baseline)
        ic_delta = ablated_ic - baseline_ic

        # Statistical significance (simplified - would need per-period ICs for proper test)
        p_value = 0.5  # Placeholder - would need proper paired test

        # Harmful means: removing the group helped (ablated > baseline) above threshold
        eps = float(feature_config["ablation"].get("min_ic_improvement", 0.005))
        results.append({
            "group": group,
            "baseline_ic": baseline_ic,
            "ablated_ic": ablated_ic,
            "ic_delta": ic_delta,
            "p_value": p_value,
            "is_harmful": ic_delta > eps
        })

        print(f"  Baseline IC: {baseline_ic:.4f}")
        print(f"  Ablated IC: {ablated_ic:.4f}")
        print(f"  Delta: {ic_delta:.4f}")
        print(f"  Harmful: {results[-1]['is_harmful']}")

    # Save results
    output_file = Path("reports/ablation_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all values are JSON serializable
    serializable_results = []
    for result in results:
        serializable_results.append({
            "group": str(result["group"]),
            "baseline_ic": float(result["baseline_ic"]),
            "ablated_ic": float(result["ablated_ic"]),
            "ic_delta": float(result["ic_delta"]),
            "p_value": float(result["p_value"]),
            "is_harmful": bool(result["is_harmful"])
        })

    with open(output_file, 'w') as f:
        json.dump({
            "profile": str(profile),
            "baseline_ic": float(baseline_ic),
            "results": serializable_results
        }, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Group':<12} {'Baseline IC':<12} {'Ablated IC':<12} {'Delta':<8} {'Harmful':<8}")
    print("-"*60)

    harmful_groups = []
    for result in results:
        harmful_marker = "YES" if result["is_harmful"] else "NO"
        print(f"{result['group']:<12} {result['baseline_ic']:<12.4f} {result['ablated_ic']:<12.4f} "
              f"{result['ic_delta']:<8.4f} {harmful_marker:<8}")

        if result["is_harmful"]:
            harmful_groups.append(result["group"])

    print("-"*60)
    print(f"Results saved to: {output_file}")

    if harmful_groups:
        print(f"⚠️  HARMFUL GROUPS: {', '.join(harmful_groups)}")
        return False
    print("✅ No harmful groups detected")
    return True


def train_and_evaluate(X, y, config, tscv=None):
    """Train model and return IC (simplified evaluation)"""
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit

    if tscv is None:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3, test_size=256, gap=config["purge_gap_bars"])
    preds, truths = [], []

    for tr, te in tscv.split(X):
        if config["model"]["kind"] == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=config["model"]["alpha"], random_state=42)
        elif config["model"]["kind"] == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=config["model"]["n_estimators"],
                max_depth=config["model"]["max_depth"],
                learning_rate=config["model"]["learning_rate"],
                random_state=config["model"]["random_state"],
                n_jobs=1
            )

        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        preds.append(p)
        truths.append(y[te])

    # Compute IC robustly (suppress constant warnings)
    y_all = np.concatenate(truths)
    p_all = np.concatenate(preds)
    if np.std(y_all) == 0 or np.std(p_all) == 0:
        return 0.0
    ic, _ = spearmanr(y_all, p_all)
    return 0.0 if np.isnan(ic) else float(ic)


def _load_snapshot(name, symbols):
    """Load data snapshot (copied from train_linear.py)"""
    import json

    import pandas as pd

    with open(f"artifacts/snapshots/{name}/manifest.json") as f:
        meta = json.load(f)
    frames = {}
    for s in symbols:
        df = pd.read_parquet(f"artifacts/snapshots/{name}/{s}.parquet")
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]  # Flatten to simple names
        frames[s] = df
    return frames, meta


def main():
    parser = argparse.ArgumentParser(description="Run feature group ablation study")
    parser.add_argument("--profile", default="golden_linear", help="Training profile to use")
    args = parser.parse_args()

    success = run_ablation(args.profile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
