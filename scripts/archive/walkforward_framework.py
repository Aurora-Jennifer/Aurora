#!/usr/bin/env python
"""
Deprecated module with compatibility exports.
Prefer: scripts/multi_walkforward_report.py

This module re-exports legacy API symbols expected by older code/tests.
When executed as a script, it forwards to the new runner and warns.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import warnings

# Emit import-time deprecation notice once
warnings.warn(
    "`scripts.walkforward_framework` is deprecated; import from `scripts.multi_walkforward_report` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export expected symbols from the compat layer
from scripts.walkforward_framework_compat import *  # noqa: F401,F403


def main() -> None:
    NEW = "scripts/multi_walkforward_report.py"
    msg = f"`{__file__}` is deprecated. Use `{NEW}`."
    warnings.warn(msg, category=DeprecationWarning, stacklevel=1)
    print(msg, file=sys.stderr)
    cmd = f"python -u {shlex.quote(NEW)} " + " ".join(map(shlex.quote, sys.argv[1:]))
    raise SystemExit(subprocess.call(cmd, shell=True))


if __name__ == "__main__":
    main()


