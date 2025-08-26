#!/usr/bin/env python3
import argparse
import json

import numpy as np


def paired_bootstrap_p(delta, B=10000, seed=42, side="greater"):
    rng = np.random.default_rng(seed)
    delta = np.array(delta, dtype=float)
    n = delta.size
    if n == 0:
        return 1.0
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = delta[idx].mean()
    if side == "greater":
        return float(np.mean(boots <= 0.0))
    return float(np.mean(boots >= 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="reports/experiments/search_history.json")
    ap.add_argument("--min_delta", type=float, default=0.003)
    ap.add_argument("--use-after-costs", action="store_true", default=False, help="Use ic_after_costs_per_fold arrays")
    args = ap.parse_args()

    with open(args.history) as f:
        h = json.load(f)
    arr_key = "ic_after_costs_per_fold" if args.use_after_costs else "ic_per_fold"
    base = np.array(h.get("baseline", {}).get(arr_key, []), float)
    best = np.array(h.get("best", {}).get(arr_key, []), float)
    if base.size == 0 or best.size == 0 or base.size != best.size:
        print("[SIG] no comparable per-fold IC arrays; skipping")
        return 0

    delta = best - base
    mean_delta = float(delta.mean())
    p_against = paired_bootstrap_p(delta, side="greater")

    es = float(mean_delta / (np.std(delta) + 1e-12))
    print(
        f"[SIG] baselineIC={base.mean():.4f} bestIC={best.mean():.4f} "
        f"\u0394IC={mean_delta:.4f} es={es:.3f} p_against={p_against:.4f}"
    )

    warn = (mean_delta < -args.min_delta) or (mean_delta <= 0 and p_against < 0.05)
    if warn:
        print(
            "[SIG][WARN] Best not significantly better than baseline "
            f"(\u0394={mean_delta:.4f}, p_against={p_against:.4f})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


