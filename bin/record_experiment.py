#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: record_experiment.py <exp_name> <metrics_json>")
        return 2
    exp = sys.argv[1]
    metrics_path = Path(sys.argv[2])
    if not metrics_path.exists():
        print("metrics json not found:", metrics_path)
        return 1
    idx = Path("artifacts/experiments/index.jsonl")
    idx.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path) as f:
        metrics = json.load(f)
    rec = {
        "ts": time.time(),
        "sha": os.getenv("GIT_SHA", ""),
        "exp": exp,
        "profile": os.getenv("TRAIN_PROFILE", "ci"),
        "snapshot": "golden_ml_v1",
        "metrics": metrics,
    }
    with open(idx, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("Recorded:", exp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


