import collections
import pathlib
import re
import sys

report = pathlib.Path("artifacts/ruff_full.txt")
if not report.exists():
    sys.exit("artifacts/ruff_full.txt not found. Run tools/scripts/lint_report.sh first.")

by_rule = collections.Counter()
by_dir = collections.Counter()
samples = {}

pat = re.compile(r"^(?P<path>.+?):\d+:\d+: (?P<code>[A-Z]\d{3})")
for line in report.read_text().splitlines():
    m = pat.match(line)
    if not m:
        continue
    path = m.group("path")
    code = m.group("code")
    top = path.split("/")[0]
    by_rule[code] += 1
    by_dir[top] += 1
    samples.setdefault(code, path)

out = pathlib.Path("docs/tech_debt/lint_summary.md")
out.parent.mkdir(parents=True, exist_ok=True)


def table(counter: collections.Counter) -> str:
    lines = ["| Item | Count |", "|---|---:|"]
    for k, v in counter.most_common():
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


md = []
md.append("# Ruff Lint Summary\n")
md.append("## By Rule Code\n")
md.append(table(by_rule))
md.append("\n\n## By Top-Level Directory\n")
md.append(table(by_dir))
md.append("\n\n## Sample Offenders (one per rule)\n")
for code, path in sorted(samples.items()):
    md.append(f"- `{code}` → `{path}`")
out.write_text("\n".join(md))
print(f"Wrote {out}")


