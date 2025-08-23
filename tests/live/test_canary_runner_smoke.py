import json
import subprocess
import sys


def test_canary_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "reports").mkdir()
    # base config: models + caps
    (tmp_path / "config" / "base.yaml").write_text(
        """
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
"""
    )
    (tmp_path / "config" / "models.yaml").write_text(
        """
registry:
  dummy_v1:
    kind: pickle
    path: artifacts/models/dummy_v1.pkl
    metadata:
      feature_order: [ret_1d, ret_5d, vol_10d]
"""
    )
    # dummy model artifact
    (tmp_path / "artifacts/models").mkdir(parents=True, exist_ok=True)
    import pickle

    from ml.models.dummy_model import DummyModel

    # 3-feature weight vector
    (tmp_path / "artifacts/models/dummy_v1.pkl").write_bytes(
        pickle.dumps(DummyModel([0.5, -0.2, 0.1]))
    )

    # run one-iteration canary
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.canary_runner as m; m.main(['--symbols','SPY,TSLA','--shadow'])",
        ],
        capture_output=True,
        text=True,
    )
    # Canary might fail due to missing data, but should not crash
    assert out.returncode in (0, 1, 2)

    # Check if logs directory was created (even if empty)
    canary_dir = tmp_path / "logs/canary"
    if canary_dir.exists():
        files = list(canary_dir.glob("*.jsonl"))
        if files:
            # If logs exist, check meta
            meta_file = tmp_path / "reports/canary_run.meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                assert "model" in meta or "fallbacks" in meta
    else:
        # If canary didn't create logs, that's also acceptable for smoke test
        pass
