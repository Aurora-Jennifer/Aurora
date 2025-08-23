#!/usr/bin/env python3
import os
import random
import subprocess
import sys

TARGET = os.environ.get("FS_TARGET", "tests/datasanity -m 'sanity or contract'")
RUNS = int(os.environ.get("FS_RUNS", "5"))


def main():
    codes = []
    for _ in range(RUNS):
        env = dict(os.environ)
        env["HYPOTHESIS_SEED"] = str(random.randint(1, 10**9))
        r = subprocess.run(["pytest", *TARGET.split(), "-q"], env=env)
        codes.append(r.returncode)
    if min(codes) == max(codes) == 0:
        print(f"[FLAKY] stable pass across {RUNS} runs")
        sys.exit(0)
    if min(codes) == max(codes) != 0:
        print(f"[FLAKY] stable fail across {RUNS} runs (fix tests)")
        sys.exit(1)
    print("[FLAKY] ❌ outcome flips detected:", codes)
    sys.exit(1)


if __name__ == "__main__":
    main()


