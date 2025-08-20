import subprocess
import sys


def test_smoke_contract_runs_smoke_profile_and_finite_ohlc():
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
    assert proc.returncode == 0, f"Smoke failed: {out}"
    assert "DataSanity validation enabled (walkforward_smoke profile)" in out, out
    assert "SMOKE_OHLC_GUARD_OK: train fold finite float64 enforced" in out, out
    assert "SMOKE_OHLC_GUARD_OK: test fold finite float64 enforced" in out, out


