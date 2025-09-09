#!/usr/bin/env python3
"""
Checklist Progress Calculator - Machine-enforced completion tracking.

Reads checklists/paper_ready.yaml and computes completion statistics.
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml


def calculate_progress(checklist_path: Path) -> dict[str, Any]:
    """Calculate progress from checklist YAML."""
    with open(checklist_path) as f:
        checklist = yaml.safe_load(f)

    total_items = 0
    done_count = 0
    todo_count = 0
    blocked_count = 0

    # Count items in each section
    for section_name, items in checklist.items():
        if section_name == "summary":
            continue

        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "status" in item:
                    total_items += 1
                    status = item["status"]
                    if status == "done":
                        done_count += 1
                    elif status == "todo":
                        todo_count += 1
                    elif status == "blocked":
                        blocked_count += 1

    completion_percentage = round((done_count / total_items) * 100) if total_items > 0 else 0

    return {
        "total_items": total_items,
        "done_count": done_count,
        "todo_count": todo_count,
        "blocked_count": blocked_count,
        "completion_percentage": completion_percentage
    }


def main():
    """Main progress calculation."""
    if len(sys.argv) < 2:
        checklist_path = Path("checklists/paper_ready.yaml")
    else:
        checklist_path = Path(sys.argv[1])

    if not checklist_path.exists():
        print(f"Error: Checklist file not found: {checklist_path}")
        return 1

    try:
        progress = calculate_progress(checklist_path)

        # Print human-readable summary
        print(f"Paper Trading Readiness: {progress['done_count']}/{progress['total_items']} ({progress['completion_percentage']}%)")
        print(f"  ✅ Done: {progress['done_count']}")
        print(f"  📋 Todo: {progress['todo_count']}")
        print(f"  🚫 Blocked: {progress['blocked_count']}")

        # Save machine-readable progress
        output_dir = Path("artifacts/status")
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_file = output_dir / "progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        print(f"Progress saved: {progress_file}")

        return 0

    except Exception as e:
        print(f"Error calculating progress: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
