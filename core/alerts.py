"""Alerting utilities with pluggable sinks and simple rules."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AlertRule:
    name: str
    when: str  # Python expression evaluated against context dict


class AlertSinks:
    @staticmethod
    def console(message: str) -> None:
        print(message)

    @staticmethod
    def file(message: str, path: str = "logs/alerts.log") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(message + "\n")

    @staticmethod
    def webhook(message: str, url: str) -> None:
        # Minimal dependency-free webhook (curl)
        os.system(
            f"curl -s -X POST -H 'Content-Type: application/json' -d '{json.dumps({'text': message})}' {url} >/dev/null 2>&1"
        )


class AlertEngine:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("alerts", {}).get("enabled", False))
        self.sinks = config.get("alerts", {}).get(
            "sinks", ["console"]
        )  # console, file, webhook
        self.rules = [AlertRule(**r) for r in config.get("alerts", {}).get("rules", [])]
        self.webhook_url = config.get("alerts", {}).get("webhook_url", "")

    def evaluate(self, context: dict[str, Any]) -> list[str]:
        if not self.enabled:
            return []
        triggered: list[str] = []
        for rule in self.rules:
            try:
                if eval(
                    rule.when,
                    {},
                    {
                        "risk": context.get("risk", {}),
                        "config": context.get("config", {}),
                        "selector": context.get("selector", {}),
                        "exec": context.get("exec", {}),
                    },
                ):
                    triggered.append(rule.name)
            except Exception:
                continue
        return triggered

    def notify(self, messages: list[str]) -> None:
        if not self.enabled or not messages:
            return
        for msg in messages:
            text = f"ALERT: {msg}"
            for sink in self.sinks:
                if sink == "console":
                    AlertSinks.console(text)
                elif sink == "file":
                    AlertSinks.file(text)
                elif sink == "webhook" and self.webhook_url:
                    AlertSinks.webhook(text, self.webhook_url)
