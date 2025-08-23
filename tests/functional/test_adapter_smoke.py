import json
import subprocess
import sys
from pathlib import Path


def test_adapter_smoke_cli(tmp_path: Path):
    out = tmp_path / "smoke.json"
    cmd = [sys.executable, "-m", "serve.adapter", "--csv", "fixtures/live_stub.csv", "--out", str(out)]
    r = subprocess.run(cmd, check=False)
    assert r.returncode == 0
    assert out.exists(), "adapter smoke did not write output"
    got = json.loads(out.read_text())
    assert got.get("rows", 0) >= 1
    # predictions may be zero if min_history_bars not yet met; ensure field exists
    assert "predictions_made" in got

