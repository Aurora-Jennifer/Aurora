#!/usr/bin/env python3
import os
import subprocess
import sys

RULES = [
    "DUP_TS",
    "NON_MONO_INDEX",
    "INF_VALUES",
    "NAN_VALUES",
]


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main():
    failures = []
    for rule in RULES:
        env = dict(os.environ)
        env["DS_DISABLE_RULE"] = rule
        print(f"== Checking that tests catch disabled rule {rule} ==")
        r = run(["pytest", "tests/datasanity", "-m", f"rule_{rule}", "-q"], env=env)
        if r.returncode == 0:
            failures.append(rule)
            sys.stdout.write(r.stdout.decode())
            print(f"[SENTRY] ❌ Tests passed with {rule} disabled — not adequate.")
        else:
            print(f"[SENTRY] ✅ Tests failed as expected for {rule}.")

    if failures:
        print("\nSENTRY SUMMARY: missing coverage for rules:", ", ".join(failures))
        sys.exit(1)
    print("\nSENTRY SUMMARY: all targeted rules are enforced.")
    sys.exit(0)


if __name__ == "__main__":
    main()


