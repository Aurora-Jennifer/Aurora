import argparse
import json
import time
import uuid
from pathlib import Path

from oms.broker import Order
from oms.paper_adapter import PaperAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="artifacts/signals/out.jsonl")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    if not args.paper:
        print("[OMS] Only --paper supported in this scaffold")
        return 1
    broker = PaperAdapter()
    sigp = Path(args.signals)
    if not sigp.exists():
        print(f"[OMS] Missing signals: {sigp}")
        return 1
    with open(sigp) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            oid = uuid.uuid4().hex[:12]
            order = Order(id=oid, symbol=s.get("symbol", "SPY"), side="BUY", qty=1)
            ack = broker.place(order)
            print(json.dumps({"order_id": oid, "status": ack.get("status"), "ts": time.time()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


