from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from brokers.paper import PaperBroker
from utils.ops_runtime import kill_switch, notify_ntfy

try:
    from tools.provenance import write_provenance
except Exception:
    import os, sys
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        from provenance import write_provenance  # type: ignore
    except Exception:
        def write_provenance(*_a, **_k):
            return {}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,TSLA")
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--profile", default="risk_balanced")
    ap.add_argument("--ntfy", action="store_true")
    args = ap.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    Path("reports").mkdir(parents=True, exist_ok=True)
    broker = PaperBroker()
    meta = {
        "run_id": run_id,
        "profile": args.profile,
        "symbols": args.symbols.split(","),
        "start": run_id,
    }
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    write_provenance("reports/paper_provenance.json", ["config/base.yaml"])

    for stopped in kill_switch(interval_s=args.poll_sec):
        if stopped:
            break
        # integrate signal + routing here in next iteration

    meta["stop"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    if args.ntfy:
        notify_ntfy("Aurora: paper done", {"run_id": run_id})


if __name__ == "__main__":
    main()


