#!/usr/bin/env python3
import json
from pathlib import Path


def main():
    p = Path("reports/experiments/ablation_summary.json")
    if not p.exists():
        print("[ABLATE] no ablation summary present; skipping")
        return 0
    js = json.loads(p.read_text())
    warned = False
    for row in js.get("table", []):
        if row.get("group") == "baseline":
            continue
        d = float(row.get("delta_mean", 0.0))
        pval = float(row.get("p_against", 1.0))
        if d > 0.005 and pval <= 0.10:
            print(f"[ABLATE][WARN] Removing '{row['group']}' improves IC by {d:.4f} (p={pval:.3f}).")
            warned = True
    print("[ABLATE] OK" if not warned else "[ABLATE] Advisory warnings emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


