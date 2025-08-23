import argparse
import json
import os
from pathlib import Path


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/experiments/pnl.json")
    ap.add_argument("--min_sharpe", type=float, default=None)
    ap.add_argument("--min_psr", type=float, default=None)
    ap.add_argument("--max_dd", type=float, default=None)
    ap.add_argument("--min_calmar", type=float, default=None)
    args = ap.parse_args()

    p = Path(args.report)
    if not p.exists():
        print(f"[GATE:PNL][FAIL] missing report: {p}")
        return 1
    r = json.loads(p.read_text())

    sharpe = float(r.get("sharpe_after_costs", r.get("sharpe", 0.0)))
    psr = float(r.get("psr", 0.0))
    max_dd = float(r.get("max_drawdown", 1.0))
    calmar = float(r.get("calmar", 0.0))

    min_sharpe = args.min_sharpe if args.min_sharpe is not None else env_float("PNL_MIN_SHARPE", 1.0)
    min_psr = args.min_psr if args.min_psr is not None else env_float("PNL_MIN_PSR", 0.95)
    max_dd_cap = args.max_dd if args.max_dd is not None else env_float("PNL_MAX_DD", 0.30)
    min_calmar = args.min_calmar if args.min_calmar is not None else env_float("PNL_MIN_CALMAR", 0.5)

    ok = True
    if sharpe < min_sharpe:
        print(f"[GATE:PNL][FAIL] sharpe {sharpe:.2f} < {min_sharpe:.2f}")
        ok = False
    if psr < min_psr:
        print(f"[GATE:PNL][FAIL] psr {psr:.2f} < {min_psr:.2f}")
        ok = False
    if max_dd > max_dd_cap:
        print(f"[GATE:PNL][FAIL] max_dd {max_dd:.2f} > {max_dd_cap:.2f}")
        ok = False
    if calmar < min_calmar:
        print(f"[GATE:PNL][FAIL] calmar {calmar:.2f} < {min_calmar:.2f}")
        ok = False

    print(json.dumps({
        "sharpe_after_costs": sharpe,
        "psr": psr,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }, indent=2))

    if not ok:
        return 1
    print("GATE_PNL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


