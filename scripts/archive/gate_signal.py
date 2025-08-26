import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="reports/experiments/search_history.json")
    ap.add_argument("--p_min", type=float, default=0.05)
    args = ap.parse_args()

    p = Path(args.history)
    if not p.exists():
        print("[GATE:SIGNAL][FAIL] search_history.json missing")
        return 1
    h = json.loads(p.read_text())
    key = "ic_after_costs_per_fold" if "ic_after_costs_per_fold" in h.get("best", {}) else "ic_per_fold"
    base = np.array(h.get("baseline", {}).get(key, []), float)
    best = np.array(h.get("best", {}).get(key, []), float)
    if base.size == 0 or best.size == 0 or base.size != best.size:
        print("[GATE:SIGNAL][FAIL] invalid fold arrays")
        return 1

    delta = best - base
    mean_delta = float(delta.mean())
    # Simple bootstrap p-value against improvement
    rng = np.random.default_rng(42)
    B = 2000
    boots = np.empty(B, float)
    n = delta.size
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = delta[idx].mean()
    p_against = float(np.mean(boots <= 0.0))

    print(json.dumps({"mean_delta": mean_delta, "p_against": p_against}, indent=2))
    if not (mean_delta > 0 and p_against < args.p_min):
        print("[GATE:SIGNAL][FAIL] insufficient OOS ΔIC or significance")
        return 1
    print("GATE_SIGNAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


