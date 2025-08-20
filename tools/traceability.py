#!/usr/bin/env python3
"""
Generate a per-test traceability table from coverage JSON with test contexts.

Requires running pytest with:
  pytest -q --cov=. --cov-context=test --cov-report=json:coverage.json

Outputs a Markdown table at tests/_traceability.md by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import ast


def parse_functions(file_path: Path) -> List[Tuple[str, int, int]]:
    """Return list of (qualified_name, start_line, end_line) for functions in file."""
    try:
        src = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    functions: List[Tuple[str, int, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def generic_visit(self, node):
            super().generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            qual = ".".join(self.stack + [node.name]) if self.stack else node.name
            # end_lineno requires Python 3.8+; fallback to start if missing
            end = getattr(node, "end_lineno", node.lineno)
            functions.append((qual, node.lineno, end))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    Visitor().visit(tree)
    return functions


def find_functions_hit(file_path: Path, lines: List[int]) -> List[str]:
    fn_ranges = parse_functions(file_path)
    if not fn_ranges:
        return []
    hit: set[str] = set()
    for name, lo, hi in fn_ranges:
        for ln in lines:
            if lo <= ln <= hi:
                hit.add(name)
                break
    return sorted(hit)


def build_traceability(coverage_json: Dict, project_root: Path) -> Dict[str, List[str]]:
    """Return mapping: test_context -> list of qualified function names hit (in project)."""
    files = coverage_json.get("files") or {}
    mapping: Dict[str, set[str]] = {}
    for filename, data in files.items():
        contexts_by_line: Dict[str, List[str]] = (data.get("contexts") or {})
        if not contexts_by_line:
            # No contexts recorded for this file; skip
            continue
        file_path = Path(filename)
        # Only map to repo files
        if not file_path.is_absolute():
            file_path = (project_root / filename).resolve()
        if not str(file_path).startswith(str(project_root.resolve())):
            continue
        # Derive function hits per line
        line_nums = [int(k) for k in contexts_by_line.keys() if k.isdigit()]
        fn_names = find_functions_hit(file_path, line_nums)
        if not fn_names:
            continue
        # Attribute each line's contexts to the hit function set for this file
        for ln_str, ctxs in contexts_by_line.items():
            for ctx in ctxs:
                # ctx is the test nodeid when using --cov-context=test
                if ctx not in mapping:
                    mapping[ctx] = set()
                for fn in fn_names:
                    mapping[ctx].add(f"{file_path.relative_to(project_root)}::{fn}")
    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in mapping.items()}


def write_markdown(mapping: Dict[str, List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("test_name | target_fn(s) | invariant | inputs | severity | oracle_type\n")
    lines.append(":- | :- | :- | :- | :- | :-\n")
    for test_name, fns in sorted(mapping.items()):
        targets = "<br/>".join(fns) if fns else ""
        lines.append(f"{test_name} | {targets} |  |  |  |  \n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="coverage.json")
    ap.add_argument("--output", default="tests/_traceability.md")
    args = ap.parse_args()

    cov_path = Path(args.coverage)
    if not cov_path.exists():
        print(f"ERROR: {cov_path} not found")
        return 2
    try:
        data = json.loads(cov_path.read_text())
    except Exception as e:
        print(f"ERROR: failed to parse coverage JSON: {e}")
        return 2
    root = Path(os.getcwd()).resolve()
    mapping = build_traceability(data, root)
    write_markdown(mapping, Path(args.output))
    print(f"Traceability written to {args.output} ({len(mapping)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


