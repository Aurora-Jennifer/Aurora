import json
from pathlib import Path


def test_paper_meta_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    import scripts.paper_runner as pr

    pr.notify_ntfy = lambda *a, **k: None
    pr.main = pr.main
    # simulate a run with immediate stop by creating kill.flag before launching
    (tmp_path / "kill.flag").write_text("stop")
    pr.main([])
    meta_path = Path("reports/paper_run.meta.json")
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert "run_id" in data and "start" in data and "stop" in data
