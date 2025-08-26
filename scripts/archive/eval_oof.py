#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


def load_costs(profile: str | None = None):
    """Load cost model configuration; optional profile from costs.yaml."""
    with open("config/components/costs.yaml") as f:
        cfg = yaml.safe_load(f)
    if profile:
        prof = (cfg.get("profiles") or {}).get(profile)
        if not prof:
            raise SystemExit(f"Unknown cost profile: {profile}")
        return prof
    return cfg


def spearman_by_date(df):
    """Compute Spearman correlation by date"""
    # Check if we have cross-sectional data (multiple assets per date)
    # or time-series data (single asset over time)
    avg_assets_per_date = df.groupby('date')['asset'].nunique().mean()

    if avg_assets_per_date >= 5:
        # Cross-sectional: compute correlation across assets for each date
        correlations = []
        for date in df['date'].unique():
            date_data = df[df['date'] == date]
            if len(date_data) >= 5:  # Need at least 5 points for correlation
                corr, _ = spearmanr(date_data['y_true'], date_data['y_pred'])
                if not np.isnan(corr):
                    correlations.append(corr)
        return np.array(correlations)
    # Time-series: compute correlation across time for each asset
    correlations = []
    for asset in df['asset'].unique():
        asset_data = df[df['asset'] == asset]
        if len(asset_data) >= 20:  # Need more points for time-series correlation
            corr, _ = spearmanr(asset_data['y_true'], asset_data['y_pred'])
            if not np.isnan(corr):
                correlations.append(corr)
    return np.array(correlations)


def estimate_turnover(df):
    """Estimate portfolio turnover from predictions"""
    # Simple turnover estimate: how much the predictions change day-to-day
    df_sorted = df.sort_values(['asset', 'date'])
    df_sorted['pred_prev'] = df_sorted.groupby('asset')['y_pred'].shift(1)

    # Calculate absolute change in predictions
    turnover = np.abs(df_sorted['y_pred'] - df_sorted['pred_prev']).mean()
    return float(turnover)


def evaluate_oof(run_dir, cost_profile: str | None = None, out_path: Path | None = None):
    """Evaluate out-of-fold performance"""
    run_path = Path(run_dir)

    # Load predictions
    preds_file = run_path / "preds_oof.parquet"
    if not preds_file.exists():
        print(f"Error: {preds_file} not found")
        return False

    df = pd.read_parquet(preds_file)

    # Basic validation
    required_cols = ['date', 'asset', 'y_true', 'y_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return False

    # Compute metrics
    correlations = spearman_by_date(df)

    if len(correlations) == 0:
        print("Error: No valid correlations computed")
        return False

    # IC metrics
    ic = float(np.mean(correlations))
    ic_std = float(np.std(correlations))
    ir = ic / ic_std if ic_std > 0 else 0.0

    # Turnover
    turnover = estimate_turnover(df)

    # Coverage (percentage of dates with predictions)
    total_dates = df['date'].nunique()
    coverage = float(len(correlations) / total_dates) if total_dates > 0 else 0.0

    # Costs
    costs = load_costs(cost_profile)
    commission_bps = costs['commission_bps']
    half_spread_bps = costs['half_spread_bps']
    slippage_bps = costs['slippage_bps_per_turnover']

    # Total costs
    total_cost_bps = commission_bps + half_spread_bps + (turnover * slippage_bps)
    ic_after_costs = ic - (total_cost_bps / 10000.0)

    # Results
    results = {
        "ic": ic,
        "ic_std": ic_std,
        "ir": ir,
        "turnover": turnover,
        "coverage": coverage,
        "costs": {
            "commission_bps": commission_bps,
            "half_spread_bps": half_spread_bps,
            "slippage_bps": slippage_bps,
            "total_cost_bps": total_cost_bps
        },
        "ic_after_costs": ic_after_costs,
        "n_dates": len(correlations),
        "total_dates": total_dates
    }

    # Save results
    output_file = out_path or Path("reports/eval_oof.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("=== OOF Evaluation Results ===")
    print(f"IC: {ic:.4f} ± {ic_std:.4f}")
    print(f"IR: {ir:.4f}")
    print(f"Turnover: {turnover:.4f}")
    print(f"Coverage: {coverage:.1%}")
    print(f"Total Costs: {total_cost_bps:.2f} bps")
    print(f"IC after costs: {ic_after_costs:.4f}")
    print(f"Results saved to: {output_file}")

    # Gate check
    ci_blocking = os.getenv("CI_BLOCKING", "false").lower() == "true"
    min_ic_after_costs = float(os.getenv("MIN_IC_AFTER_COSTS", "0.05"))

    if ci_blocking and ic_after_costs < min_ic_after_costs:
        print(f"❌ FAIL: IC after costs {ic_after_costs:.4f} < {min_ic_after_costs}")
        return False

    if ci_blocking and turnover > 2.5:
        print(f"❌ FAIL: Turnover {turnover:.4f} > 2.5")
        return False

    print("✅ PASS: All gates passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate out-of-fold performance")
    parser.add_argument("run_dir", help="Directory containing preds_oof.parquet")
    parser.add_argument("--cost-profile", default=None, help="Cost profile key from costs.yaml")
    parser.add_argument("--out", default="reports/eval_oof.json", help="Output JSON path")
    args = parser.parse_args()

    success = evaluate_oof(args.run_dir, cost_profile=args.cost_profile, out_path=Path(args.out))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
