import collections
import re
from pathlib import Path

report_path = Path("artifacts/ruff_full.txt")
if not report_path.exists():
    raise SystemExit("artifacts/ruff_full.txt not found. Run tools/scripts/lint_report.sh first.")

data = report_path.read_text().splitlines()
pat = re.compile(r"^(?P<path>.+?):\d+:\d+: (?P<code>[A-Z]\d{3})")
dirs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

for line in data:
    m = pat.match(line)
    if not m:
        continue
    path = m.group("path")
    code = m.group("code")
    top = path.split("/")[0]
    dirs[top][code] += 1

out = Path("docs/tech_debt/lint_checklist.md")
out.parent.mkdir(parents=True, exist_ok=True)
lines = ["# Lint Cleanup Checklist\n"]
for d, counts in sorted(dirs.items()):
    total = sum(counts.values())
    lines.append(f"\n## {d} — {total} issues")
    for code, cnt in counts.most_common():
        lines.append(f"- [ ] Fix `{code}` x{cnt}")

out.write_text("\n".join(lines))
print(f"Wrote {out}")


