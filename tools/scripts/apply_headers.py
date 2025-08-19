#!/usr/bin/env python3
from pathlib import Path

HEADER_LINES = [
    "# © 2025 Aurora Analytics — Proprietary",
    "# No redistribution, public posting, or model-training use.",
    "# Internal-only. Ref: AUR-NOTICE-2025-01",
]


TARGET_DIRS = [
    "core",
    "brokers",
    "strategies",
    "features",
    "signals",
    "risk",
    "utils",
    "viz",
    "ml",
    "experiments",
    "cli",
    "apps",
    "api",
]


def should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in {"tests", "attic", ".venv", "venv", "__pycache__"} for p in parts):
        return True
    return path.suffix != ".py"


def has_header(text: str) -> bool:
    return "AUR-NOTICE-2025-01" in text or "© 2025 Aurora Analytics — Proprietary" in text


def insert_header(text: str) -> str:
    lines = text.splitlines()
    out = []
    idx = 0
    # Preserve shebang
    if lines and lines[0].startswith("#!/"):
        out.append(lines[0])
        idx = 1
        # Preserve encoding comment if present
        if idx < len(lines) and lines[idx].startswith("#") and "coding" in lines[idx]:
            out.append(lines[idx])
            idx += 1
    # Insert header
    out.extend(HEADER_LINES)
    # Ensure a blank line after header
    out.append("")
    # Append the rest
    out.extend(lines[idx:])
    return "\n".join(out) + ("\n" if not text.endswith("\n") else "")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    edited = 0
    for rel in TARGET_DIRS:
        base = repo_root / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if should_skip(path):
                continue
            text = path.read_text(encoding="utf-8")
            if has_header(text):
                continue
            new_text = insert_header(text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                edited += 1
    print(f"Applied headers to {edited} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
