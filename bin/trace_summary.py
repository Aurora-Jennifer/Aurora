#!/usr/bin/env python3
import collections
import glob
import json

agg = collections.defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0})
for p in glob.glob("artifacts/traces/*.json"):
    with open(p) as f:
        d = json.load(f)
    for code, counts in d.get("summary", {}).items():
        for k, v in counts.items():
            agg[code][k] = agg[code].get(k, 0) + int(v)

print("RULE\tPASS\tWARN\tFAIL")
for rule, c in sorted(agg.items()):
    print(f"{rule}\t{c['pass']}\t{c['warn']}\t{c['fail']}")

# Non-blocking: CI enforces policy via step continue-on-error.

