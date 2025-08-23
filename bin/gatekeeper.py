#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode(errors="ignore")


def parse_coverage(path="coverage.xml"):
    if not os.path.exists(path):
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        # coverage.py format
        lines_valid = int(root.attrib.get("lines-valid") or 0)
        lines_covered = int(root.attrib.get("lines-covered") or 0)
        branches_valid = int(root.attrib.get("branches-valid") or 0)
        branches_covered = int(root.attrib.get("branches-covered") or 0)
        line_cov = (lines_covered / lines_valid * 100.0) if lines_valid else None
        branch_cov = (branches_covered / branches_valid * 100.0) if branches_valid else None
        return {"line": line_cov, "branch": branch_cov}
    except Exception:
        return None


def parse_mutmut_junit(path="artifacts/mutmut-report.xml"):
    if not os.path.exists(path):
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tests = int(root.attrib.get("tests") or 0)
        failures = int(root.attrib.get("failures") or 0)
        errors = int(root.attrib.get("errors") or 0)
        passed = tests - failures - errors
        score = (passed / tests * 100.0) if tests else None
        return {"tests": tests, "passed": passed, "score": score}
    except Exception:
        return None


def heatmap_ok(path="artifacts/trace_heatmap.tsv"):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) <= 1:
            return False
        ok = True
        for row in lines[1:]:
            parts = row.split("\t")
            if len(parts) < 4:
                continue
            _, p, _, fail = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            if not (p >= 1 and fail >= 1):
                ok = False
        return ok
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["train", "paper"], default="train")
    ap.add_argument("--sentry", action="store_true", default=True)
    args = ap.parse_args()

    summary_lines = []
    passed = True

    # 1) Sentry
    code, _ = run([sys.executable, "bin/ds_sentry.py"])
    summary_lines.append(f"Sentry: {'PASS' if code == 0 else 'FAIL'}")
    passed &= (code == 0)

    # 2) Heatmap coverage
    hm = heatmap_ok()
    if hm is None:
        summary_lines.append("Heatmap: N/A (no traces)")
    else:
        summary_lines.append(f"Heatmap coverage (≥1 pass & fail per rule): {'PASS' if hm else 'FAIL'}")
        passed &= hm

    # 3) Perf contract
    code, _ = run(["pytest", "tests/datasanity/test_perf_contract.py", "-q"])
    summary_lines.append(f"Perf budget: {'PASS' if code == 0 else 'FAIL'}")
    passed &= (code == 0)

    # 4) Flakiness scout
    code, out = run([sys.executable, "bin/flaky_scout.py"])
    summary_lines.append(f"Flakiness: {'PASS' if code == 0 else 'FAIL'}")
    passed &= (code == 0)

    # 5) Train readiness tests
    code, _ = run(["pytest", "tests/train_readiness", "-m", "sanity", "-q"])
    summary_lines.append(f"Train-readiness tests: {'PASS' if code == 0 else 'FAIL'}")
    passed &= (code == 0)

    # 6) Coverage (optional)
    cov = parse_coverage()
    if cov:
        line_ok = (cov["line"] or 0) >= 90.0
        branch_ok = (cov["branch"] or 0) >= 85.0
        summary_lines.append(f"Coverage: line={cov['line']:.1f}% (≥90), branch={cov['branch'] if cov['branch'] is not None else 'N/A'}% (≥85)")
        passed &= line_ok and (cov["branch"] is None or branch_ok)
    else:
        summary_lines.append("Coverage: N/A")

    # 7) Mutation (optional)
    mut = parse_mutmut_junit()
    if mut and mut["score"] is not None:
        mut_ok = mut["score"] >= 70.0
        summary_lines.append(f"Mutation score: {mut['score']:.1f}% (≥70)")
        passed &= mut_ok
    else:
        summary_lines.append("Mutation score: N/A")

    # Output
    title = f"## Gatekeeper: Ready to {args.target.capitalize()} — {'✅' if passed else '❌'}\n"
    body = "\n".join(f"- {line}" for line in summary_lines)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a") as f:
            f.write("\n" + title + body + "\n")
    else:
        print(title)
        print(body)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()


