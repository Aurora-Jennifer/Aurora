#!/usr/bin/env python3
"""
Aurora Run Wrapper
Wraps any command and records telemetry.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure repository root is on sys.path for package imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.telemetry.runlog import create_run_logger  # noqa: E402, I001

def main():
    parser = argparse.ArgumentParser(description="Wrap a command with telemetry")
    parser.add_argument("command", nargs="+", help="Command to run")
    parser.add_argument("--run-id", help="Custom run ID")
    args = parser.parse_args()

    # Create run logger
    logger = create_run_logger()

    try:
        # Log command start
        logger.log_event("command_start", {
            "command": " ".join(args.command),
            "cwd": str(Path.cwd())
        })

        # Run command
        result = subprocess.run(args.command, check=False)

        # Log command completion
        logger.log_event("command_completed", {
            "exit_code": result.returncode,
            "stdout_length": len(getattr(result, "stdout", b"") or b"") if hasattr(result, "stdout") else 0,
            "stderr_length": len(getattr(result, "stderr", b"") or b"") if hasattr(result, "stderr") else 0
        })

        # Finish logging
        logger.finish(result.returncode)

        # Exit with same code
        sys.exit(result.returncode)

    except Exception as e:
        # Log error
        logger.log_event("command_error", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        logger.finish(1)
        raise

if __name__ == "__main__":
    main()
