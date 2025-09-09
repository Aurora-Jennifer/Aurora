from __future__ import annotations

from collections import defaultdict
from typing import Any


class Trace:
    """
    Lightweight decision tracing for DataSanity validators.

    Collects rule execution outcomes to assert test adequacy and publish coverage.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def hit(self, code: str, outcome: str, meta: dict[str, Any] | None = None) -> None:
        self.events.append({"code": code, "outcome": outcome, "meta": meta or {}})

    def summary(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0})
        for e in self.events:
            counts[e["code"]][e["outcome"]] = counts[e["code"]].get(e["outcome"], 0) + 1
        return dict(counts)


