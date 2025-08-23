#!/usr/bin/env python3
"""
DataSanity Checker - Validates data against configurable rules
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_df(p):
    """Load DataFrame from parquet or CSV."""
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def fail(rule_id, msg, n=None):
    """Create a failure record."""
    return {"id": rule_id, "msg": msg, "n": int(n) if n is not None else None}


def main():
    ap = argparse.ArgumentParser(description="DataSanity checker")
    ap.add_argument("--features", required=True, help="Features file path")
    ap.add_argument("--labels", required=False, help="Labels file path")
    ap.add_argument("--rules", required=True, help="Rules configuration file")
    ap.add_argument("--out", required=True, help="Output report path")
    args = ap.parse_args()

    # Load configuration
    with open(args.rules) as f:
        cfg = yaml.safe_load(f)

    # Load data
    X = load_df(args.features)
    y = None
    if args.labels and Path(args.labels).exists():
        y = load_df(args.labels)
        if "y" in y.columns and "timestamp" in y.columns and "symbol" in y.columns:
            X = X.merge(y[["timestamp", "symbol", "y"]], on=["timestamp", "symbol"], how="left")

    fails, counts = [], {}

    # 1) Schema checks
    req = cfg["schema"]["required_cols"]
    missing = [c for c in req if c not in X.columns]
    if missing:
        fails.append(fail("schema_missing_cols", f"missing={missing}"))

    # dtype checks (best-effort)
    for col, dt in cfg["schema"]["dtypes"].items():
        if col in X.columns:
            ok = True
            if dt.startswith("datetime64") and not np.issubdtype(X[col].dtype, np.datetime64):
                try:
                    X[col] = pd.to_datetime(X[col], errors="raise")
                except (ValueError, TypeError):
                    ok = False
            elif dt.startswith("float") and not np.issubdtype(X[col].dtype, np.floating):
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                except (ValueError, TypeError):
                    ok = False
            elif dt == "string" and X[col].dtype.name != "string":
                X[col] = X[col].astype("string")
            if not ok:
                fails.append(fail("schema_dtype_mismatch", f"{col} expected {dt}, got {X[col].dtype}"))

    # 2) Row rules
    def count_bad(mask, rule_id):
        """Count bad rows and store in counts."""
        n = int(mask.sum())
        counts[rule_id] = n
        return n

    for r in cfg.get("row_rules", []):
        if r["kind"] == "non_null":
            mask = X[r["cols"]].isna().any(axis=1)
            if count_bad(mask, r["id"]):
                fails.append(fail(r["id"], "nulls present", n=mask.sum()))
        elif r["kind"] == "finite":
            mask = ~np.isfinite(X[r["cols"]]).all(axis=1)
            if count_bad(mask, r["id"]):
                fails.append(fail(r["id"], "non-finite present", n=mask.sum()))
        elif r["kind"] == "min":
            col = r["col"]
            mask = X[col] < r["min"]
            if count_bad(mask, r["id"]):
                fails.append(fail(r["id"], f"{col} < {r['min']}", n=mask.sum()))
        elif r["kind"] == "range":
            col = r["col"]
            mask = (X[col] < r["min"]) | (X[col] > r["max"])
            if count_bad(mask, r["id"]):
                fails.append(fail(r["id"], f"{col} outside [{r['min']},{r['max']}]", n=mask.sum()))
        elif r["kind"] == "range_soft":
            col = r["col"]
            mask = (X[col] < r["min"]) | (X[col] > r["max"])
            # soft rule: allow small fraction
            frac = mask.mean()
            counts[r["id"]] = float(frac)
            if frac > cfg["thresholds"]["soft_fail_fraction"]:
                fails.append(fail(r["id"], f"{col} outside soft range > {cfg['thresholds']['soft_fail_fraction']*100:.2f}% rows", n=int(mask.sum())))

    # 3) Dataset rules
    for r in cfg.get("dataset_rules", []):
        if r["kind"] == "no_dupes":
            n = int(X.duplicated(r["subset"]).sum())
            counts[r["id"]] = n
            if n:
                fails.append(fail(r["id"], f"duplicate keys in {r['subset']}", n=n))
        elif r["kind"] == "monotonic":
            g = X.sort_values([r["by"], r["col"]]).groupby(r["by"])[r["col"]]
            mono = g.apply(lambda s: s.is_monotonic_increasing)
            bad = (~mono).sum()
            counts[r["id"]] = int(bad)
            if bad:
                fails.append(fail(r["id"], f"non-monotonic {r['col']} within {r['by']} groups", n=int(bad)))
        elif r["kind"] == "no_future_leak" and "y" in X.columns:
            # crude leak tripwire: corr(feature_t, y_t+1) should drop if we shift features correctly
            try:
                from scipy.stats import spearmanr
                leaks = []
                Xs = X.sort_values(["symbol", "timestamp"]).copy()
                Xs["y_fwd1"] = Xs.groupby("symbol")["y"].shift(-1)
                for c in r["features"]:
                    if c in Xs.columns:
                        rho_now = abs(pd.Series(spearmanr(Xs[c], Xs["y"], nan_policy="omit")[0]).fillna(0))
                        rho_fwd = abs(pd.Series(spearmanr(Xs[c], Xs["y_fwd1"], nan_policy="omit")[0]).fillna(0))
                        if float(rho_fwd) - float(rho_now) > 0.05:  # heuristic bump → suspicious
                            leaks.append({"feature": c, "delta_rho": float(rho_fwd - rho_now)})
                counts[r["id"]] = len(leaks)
                if leaks:
                    fails.append({"id": r["id"], "msg": "suspected future leak (>0.05 Spearman bump vs y_fwd1)", "n": len(leaks), "details": leaks})
            except ImportError:
                print("[WARN] scipy not available, skipping leakage check")

    # Generate output
    ok = len(fails) == 0
    out = {"ok": ok, "failing_rules": fails, "counts": counts}

    # Save report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Print summary
    print(json.dumps({"datasanity": out}, indent=2))

    # Exit with appropriate code
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
