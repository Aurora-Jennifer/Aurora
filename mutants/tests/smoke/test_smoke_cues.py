import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_smoke_emits_expected_cues():
    cmd = [
        sys.executable,
        "scripts/multi_walkforward_report.py",
        "--smoke",
        "--validate-data",
        "--datasanity-profile",
        "walkforward_smoke",
        "--allow-zero-trades",
        "--log-level",
        "INFO",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "DataSanity validation enabled (walkforward_smoke profile)" in out
    assert "SMOKE_OHLC_GUARD_OK: train fold finite float64 enforced" in out
    assert "SMOKE_OHLC_GUARD_OK: test fold finite float64 enforced" in out


