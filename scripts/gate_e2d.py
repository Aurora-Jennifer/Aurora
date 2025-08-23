#!/usr/bin/env python3
"""
E2D Gate - Advisory checks for end-to-decision pipeline
Checks: latency ≤ SLO, summary present, at least one decision, DataSanity status
"""
import json
import os

SLO = float(os.getenv("E2D_SLO_MS", "150"))
STRICT_DS = os.getenv("E2D_DS_STRICT", "0") == "1"
DECISION_REQUIRED = os.getenv("E2D_DECISION_REQUIRED", "1") == "1"
E2D_KILL = os.getenv("E2D_KILL", "0") == "1"


def read(p):
    """Read JSON file safely."""
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    # Circuit breaker check
    if E2D_KILL:
        print("[E2D][KILL] Circuit breaker activated - E2D_KILL=1")
        return 1  # hard fail

    base = "artifacts/e2d/last"
    summ_p = os.path.join(base, "summary.json")
    dec_p = os.path.join(base, "decision.json")

    if not os.path.exists(summ_p):
        print("[E2D][WARN] missing summary.json")
        return 0  # advisory

    s = read(summ_p)
    dur = float(s.get("total_latency_ms", 1e9))

    # Check latency SLO
    if dur > SLO:
        print(f"[E2D][WARN] latency {dur:.1f}ms exceeds SLO {SLO:.1f}ms")

    # Check DataSanity status
    ds = s.get("datasanity", {})
    if STRICT_DS and not ds.get("ok", False):
        failing_rules = ds.get("failing_rules", [])
        print(f"[E2D][WARN] DataSanity failed; rules={failing_rules}")

    # Check decisions
    n_decisions = 0
    if DECISION_REQUIRED:
        if not os.path.exists(dec_p):
            print("[E2D][WARN] missing decision.json")
        else:
            dec = read(dec_p)
            if not isinstance(dec, list):
                print("[E2D][WARN] decision.json is not a list")
            else:
                n_decisions = len(dec)
                if n_decisions == 0:
                    print("[E2D][WARN] no decisions emitted")

    # Summary line
    print(f"[E2D] summary: latency={dur:.1f}ms, ds_ok={ds.get('ok', None)}, decisions={n_decisions}")

    return 0  # advisory


if __name__ == "__main__":
    raise SystemExit(main())
