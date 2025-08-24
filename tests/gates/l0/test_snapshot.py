"""
L0 Snapshot Immutability Contract
- Golden snapshot directory content-hash matches recorded hash
- Files are read-only (attempted writes must fail)
Assumes a recorded hash file exists at:
  artifacts/snapshots/golden_ml_v1/HASH.txt  (contains a single sha256 hex)
Create/update via your snapshot pipeline (outside of tests).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

HASH_FILE_NAME = "HASH.txt"

def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != HASH_FILE_NAME:
            yield p

def _dir_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in _iter_files(root):
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()

def test_snapshot_hash_matches_manifest(snapshot_dir: Path):
    hash_path = snapshot_dir / HASH_FILE_NAME
    assert hash_path.exists(), (
        f"Missing {hash_path}. Generate it via your snapshot job and commit it.\n"
        "Example: python scripts/hash_snapshot.py artifacts/snapshots/golden_ml_v1 > HASH.txt"
    )
    recorded = hash_path.read_text().strip()
    current = _dir_sha256(snapshot_dir)
    assert current == recorded, f"Snapshot hash mismatch:\n current={current}\n recorded={recorded}"

def test_snapshot_files_are_read_only(snapshot_dir: Path, tmp_path: Path):
    violators = []
    for p in _iter_files(snapshot_dir):
        if os.access(p, os.W_OK):
            violators.append(str(p))
        # Attempt to write should fail (copy file to tmp then try to overwrite original)
        try:
            with open(p, "ab") as f:
                f.write(b"X")  # should raise PermissionError on RO files
            violators.append(str(p))
        except PermissionError:
            pass  # expected
        finally:
            # Revert any accidental write on permissive FS
            try:
                os.truncate(p, os.path.getsize(p) - 1)  # noop if write failed
            except Exception:
                pass
    assert not violators, f"Snapshot files writable (violates immutability): {violators}"
