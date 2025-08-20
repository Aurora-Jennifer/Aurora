"""
File integrity helpers. Not enforced yet; import-ready only.
"""
from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

