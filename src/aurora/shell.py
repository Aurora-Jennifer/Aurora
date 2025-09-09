"""
Secure subprocess wrapper - no shell, validated argv
"""
import subprocess
import shlex
from typing import Sequence


def run_cmd(argv: Sequence[str]) -> str:
    """
    Run command safely - no shell=True, caller must pass sanitized args
    """
    assert isinstance(argv, (list, tuple)) and all(isinstance(a, str) for a in argv)
    # No shell=True, caller must pass sanitized args
    out = subprocess.run(list(argv), check=True, capture_output=True, text=True)
    return out.stdout
