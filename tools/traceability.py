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
        if isinstance(contexts, dict):
            has_contexts = True
            for ctx, ctx_payload in contexts.items():
                # Only consider test contexts
                if "test" not in ctx and "/tests/" not in ctx and "::test" not in ctx:
                    continue
                mapping.setdefault(ctx, set()).add(file_path)
            continue

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


