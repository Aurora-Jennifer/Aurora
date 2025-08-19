import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("symbols", ["SPY,TSLA"])
def test_canary_runner_isolated(tmp_path, symbols, monkeypatch):
    # isolate FS
    monkeypatch.chdir(tmp_path)

    # make sure notifications are NOOP in test
    monkeypatch.delenv("NTFY_URL", raising=False)

    # ensure prev_weights state starts clean in tmp
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    # Create minimal config files needed by the runner
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "base.yaml").write_text("""
models:
  enable: true
  selected: dummy_v1
  input_features: [ret_1d, ret_5d, vol_10d]
  score_to_weight: tanh
  max_abs_weight: 0.5
  min_history_bars: 60
live:
  per_trade_notional_pct: 0.01
  notional_daily_cap_pct: 0.10
paper:
  weight_spike_cap: 0.2
  turnover_cap: 0.8
""")
    (tmp_path / "config" / "models.yaml").write_text("""
registry:
  dummy_v1:
    kind: pickle
    path: artifacts/models/dummy_v1.pkl
    metadata:
      feature_order: [ret_1d, ret_5d, vol_10d]
""")

    # Create dummy model artifact
    (tmp_path / "artifacts/models").mkdir(parents=True, exist_ok=True)
    import pickle

    from ml.models.dummy_model import DummyModel

    (tmp_path / "artifacts/models/dummy_v1.pkl").write_bytes(
        pickle.dumps(DummyModel([0.5, -0.2, 0.1]))
    )

    # Run the canary runner for 2 iterations in shadow mode with dummy quotes
    # Use absolute path to script from repo root
    script_path = Path(__file__).parent.parent.parent / "scripts" / "canary_runner.py"
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--symbols",
        symbols,
        "--poll-sec",
        "0.05",
        "--profile",
        "live_canary",
        "--shadow",
        "--steps",
        "2",
    ]
    # ENV: avoid any network hooks; prefer deterministic paths
    env = os.environ.copy()
    # Set PYTHONPATH to include repo root so imports work
    repo_root = Path(__file__).parent.parent.parent
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    # If your runner chooses provider by env/flag, set it here (example):
    # env["QUOTES_PROVIDER"] = "dummy"

    res = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    # Artifacts should be created under tmp_path
    logs_dir = tmp_path / "logs" / "canary"
    meta_file = tmp_path / "reports" / "canary_run.meta.json"

    # At least one line in the canary log
    log_files = list(logs_dir.glob("*.jsonl"))
    assert log_files, "expected canary jsonl logs"
    assert log_files[0].stat().st_size > 0

    # Meta should include basic fields
    meta = json.loads(meta_file.read_text())
    assert isinstance(meta, dict)
    assert meta.get("profile") == "live_canary"
    # model block is optional, but if enabled should have id/hash
    if meta.get("model_enabled"):
        m = meta.get("model", {})
        assert m.get("id") and m.get("artifact_sha256")
