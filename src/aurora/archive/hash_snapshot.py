#!/usr/bin/env python
"""
Compute sha256 over all files in a snapshot dir (path arguments allowed).
Writes HASH.txt by default.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def dir_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "HASH.txt":
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/snapshots/golden_ml_v1")
    digest = dir_sha256(root)
    (root/"HASH.txt").write_text(digest+"\n")
    print(digest)

if __name__ == "__main__":
    main()
