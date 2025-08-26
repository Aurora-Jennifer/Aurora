import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf_report", default="reports/experiments/wf.json")
    ap.add_argument("--min_pos_ratio", type=float, default=0.6)
    args = ap.parse_args()

    p = Path(args.wf_report)
    if not p.exists():
        print(f"[GATE:WF][FAIL] missing report: {p}")
        return 1
    d = json.loads(p.read_text())
    segments = d.get("segments", [])
    if not segments:
        print("[GATE:WF][FAIL] no segments")
        return 1
    pos = sum(1 for s in segments if s.get("pnl_after_costs", 0.0) > 0.0 and s.get("ic", 0.0) > 0.0)
    ratio = pos / len(segments)
    print(json.dumps({"segments": len(segments), "positive_ratio": ratio}, indent=2))
    if ratio < args.min_pos_ratio:
        print(f"[GATE:WF][FAIL] pos_ratio {ratio:.2f} < {args.min_pos_ratio:.2f}")
        return 1
    print("GATE_WF_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


