#!/usr/bin/env python3
import sys
from pathlib import Path

import yaml
from jsonschema import ValidationError, validate

SCHEMA = {
    "type": "object",
    "properties": {
        "engine": {
            "type": "object",
            "properties": {"min_history_bars": {"type": "integer", "minimum": 1}},
        },
        "walkforward": {
            "type": "object",
            "properties": {
                "fold_length": {"type": "integer", "minimum": 1},
                "step_size": {"type": "integer", "minimum": 1},
                "allow_truncated_final_fold": {"type": "boolean"},
            },
        },
        "data": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "auto_adjust": {"type": "boolean"},
            },
        },
        "risk": {
            "type": "object",
            "properties": {
                "max_drawdown": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "daily_loss_limit": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    },
    "required": ["engine", "walkforward", "data", "risk"],
}


def main(path: str = "config/base.yaml") -> int:
    cfg = yaml.safe_load(Path(path).read_text())
    try:
        validate(cfg, SCHEMA)
    except ValidationError as e:
        print(f"[CONFIG FAIL] {e.message}")
        return 1
    print("[CONFIG OK]")
    return 0


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "config/base.yaml"
    raise SystemExit(main(p))
