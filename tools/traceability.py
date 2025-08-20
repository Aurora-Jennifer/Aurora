#!/usr/bin/env python3
"""
Generate a simple per-test traceability table from coverage.json produced by:

  pytest --cov=. --cov-context=test --cov-report=json:coverage.json

This script is best-effort: if coverage contexts are missing, it emits a
placeholder with guidance.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-test traceability table from coverage.json")
    parser.add_argument("--coverage", default="coverage.json", help="Path to coverage JSON file")
    parser.add_argument("--output", default="tests/_traceability.md", help="Output markdown path")
    return parser.parse_args()


def is_repo_source(path: str) -> bool:
    # Heuristic: include files under repo, excluding tests and virtualenv/site-packages
    lowered = path.replace("\\", "/").lower()
    if "/tests/" in lowered or lowered.endswith("/tests"):
        return False
    if "site-packages" in lowered or ".venv" in lowered or "/usr/lib/" in lowered:
        return False
    return True


def load_coverage(coverage_path: Path) -> Dict:
    with coverage_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_traceability(cov: Dict) -> Tuple[Dict[str, Set[str]], bool]:
    """Return mapping test_context -> set(source_files), and whether contexts were present."""
    files = cov.get("files") or {}
    has_contexts = False
    mapping: Dict[str, Set[str]] = {}

    for file_path, payload in files.items():
        if not is_repo_source(file_path):
            continue

        # coverage.py JSON may include contexts under different keys across versions.
        # Try common shapes:
        contexts = payload.get("contexts")
        handled = False
        if isinstance(contexts, dict):
            # coverage json with show_contexts=True often stores line-> [contexts]
            any_val = next(iter(contexts.values()), None)
            if isinstance(any_val, list):
                has_contexts = True
                handled = True
                seen_ctxs: Set[str] = set()
                for _, ctx_list in contexts.items():
                    for ctx in ctx_list or []:
                        if isinstance(ctx, str) and ctx:
                            if "test" in ctx or "/tests/" in ctx or "::test" in ctx:
                                seen_ctxs.add(ctx)
                for ctx in seen_ctxs:
                    mapping.setdefault(ctx, set()).add(file_path)
            else:
                # Rare shape: context -> payload (not observed in our runs). Best-effort parse.
                has_contexts = True
                handled = True
                for ctx in contexts.keys():
                    if not isinstance(ctx, str):
                        continue
                    if "test" in ctx or "/tests/" in ctx or "::test" in ctx:
                        mapping.setdefault(ctx, set()).add(file_path)

        if not handled:
            # Some versions may use a list of contexts per line; try contexts_by_lineno
            contexts_by_lineno = payload.get("contexts_by_lineno")
            if isinstance(contexts_by_lineno, dict):
                has_contexts = True
                # Aggregate all contexts observed for this file
                seen_ctxs: Set[str] = set()
                for _, ctx_list in contexts_by_lineno.items():
                    for ctx in ctx_list or []:
                        if isinstance(ctx, str):
                            if "test" in ctx or "/tests/" in ctx or "::test" in ctx:
                                seen_ctxs.add(ctx)
                for ctx in seen_ctxs:
                    mapping.setdefault(ctx, set()).add(file_path)
                continue

        # No contexts for this file; skip
        continue

    return mapping, has_contexts


def write_markdown(output_path: Path, mapping: Dict[str, Set[str]], contexts_present: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# Test Traceability\n\n")
        if not contexts_present:
            out.write(
                "Note: Coverage contexts not found in coverage.json. Ensure you run with coverage>=7 and pytest-cov using `--cov-context=test`, or generate JSON with contexts enabled.\n\n"
            )
        if not mapping:
            out.write("No traceability data available.\n")
            return

        out.write("test_context | target_files (count)\n")
        out.write("--- | ---\n")
        for ctx in sorted(mapping.keys()):
            files_list = sorted(mapping[ctx])
            # Compact display: show up to 5, then count
            preview = ", ".join(files_list[:5])
            extra = len(files_list) - 5
            if extra > 0:
                preview += f" (+{extra} more)"
            out.write(f"{ctx} | {preview} ({len(files_list)})\n")


def main() -> None:
    args = parse_args()
    coverage_path = Path(args.coverage)
    output_path = Path(args.output)

    if not coverage_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# Test Traceability\n\ncoverage.json not found. Ensure CI ran pytest with coverage and contexts.\n",
            encoding="utf-8",
        )
        return

    cov = load_coverage(coverage_path)
    mapping, contexts_present = build_traceability(cov)
    write_markdown(output_path, mapping, contexts_present)


if __name__ == "__main__":
    main()
