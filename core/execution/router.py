from __future__ import annotations
import time
from typing import Dict, Any, Iterable, Tuple


def rate_limit(last_ts: float, min_interval_s: float) -> float:
    now = time.monotonic()
    if now - last_ts < min_interval_s:
        time.sleep(min_interval_s - (now - last_ts))
    return time.monotonic()


def slice_order(symbol: str, side: str, qty: float, slices: int) -> Iterable[Tuple[str, str, float]]:
    n = max(1, int(slices))
    chunk = qty / n
    for _ in range(n):
        yield symbol, side, chunk


