"""Structured logging utilities (JSONL)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def log_event(topic: str, payload: dict[str, Any], base_dir: str = "logs") -> None:
    ensure_dir(base_dir)
    fname = os.path.join(base_dir, f"{topic}.jsonl")
    record = {"ts": datetime.utcnow().isoformat() + "Z", "topic": topic, **payload}
    with open(fname, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
