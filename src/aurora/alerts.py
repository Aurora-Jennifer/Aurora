"""Alerting utilities with pluggable sinks and simple rules."""

from __future__ import annotations

import ast
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def _safe_eval(expr: str, context: dict[str, Any]) -> bool:
    """Evaluate a simple boolean expression safely.

    Supports names from {risk, config, selector, exec}, literals, comparisons,
    boolean ops, subscripts, and attributes. No function calls.
    """
    allowed_names = {"risk", "config", "selector", "exec"}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            raise ValueError("unsupported boolean op")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _eval(node.operand)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = _eval(comparator)
                if isinstance(op, ast.Eq) and left != right:
                    return False
                if isinstance(op, ast.NotEq) and left == right or isinstance(op, ast.Gt) and not (left > right) or isinstance(op, ast.GtE) and not (left >= right) or isinstance(op, ast.Lt) and not (left < right) or isinstance(op, ast.LtE) and not (left <= right) or isinstance(op, ast.In) and left not in right or isinstance(op, ast.NotIn) and not (left not in right):
                    return False
                # continue to next comparison
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError("unknown name")
            return context.get(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Subscript):
            base = _eval(node.value)
            key = _eval(node.slice)
            return base[key]
        if isinstance(node, ast.Attribute):
            base = _eval(node.value)
            return getattr(base, node.attr)
        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]
        if isinstance(node, ast.Dict):
            return { _eval(k): _eval(v) for k, v in zip(node.keys, node.values, strict=False) }
        raise ValueError("unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return bool(_eval(tree))


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
        # Minimal dependency-free webhook using urllib (no shell) to avoid injection
        data = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 - controlled URL from config
                _ = resp.read()
        except Exception:
            pass


class AlertEngine:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("alerts", {}).get("enabled", False))
        self.sinks = config.get("alerts", {}).get("sinks", ["console"])  # console, file, webhook
        self.rules = [AlertRule(**r) for r in config.get("alerts", {}).get("rules", [])]
        self.webhook_url = config.get("alerts", {}).get("webhook_url", "")

    def evaluate(self, context: dict[str, Any]) -> list[str]:
        if not self.enabled:
            return []
        triggered: list[str] = []
        for rule in self.rules:
            try:
                if _safe_eval(
                    rule.when,
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
