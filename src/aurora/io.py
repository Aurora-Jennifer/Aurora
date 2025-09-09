"""
Safe I/O operations with security hardening
"""
from __future__ import annotations
import json
import pathlib
import yaml


def load_yaml(path: str | pathlib.Path) -> dict:
    """Load YAML file safely"""
    with open(path, "rt") as f:
        return yaml.safe_load(f)  # safe loader


def save_json(obj, path: str | pathlib.Path) -> None:
    """Save object as JSON"""
    with open(path, "wt") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_pickle_forbidden(*_, **__):
    """Forbid pickle loading by default for security"""
    raise RuntimeError("pickle loading is disabled by default (security).")
