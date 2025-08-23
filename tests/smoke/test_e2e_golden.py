import json
import subprocess
import sys
import pathlib

PROFILE = "golden_xgb_v2"
RUN_DIR = pathlib.Path("artifacts/run_ci")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, check=False, text=True, capture_output=True)
    assert r.returncode == 0, f"{cmd} failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    return r


def test_e2d_once_golden():
    run([
        sys.executable, "scripts/e2d.py", "--profile", PROFILE, "--out", str(RUN_DIR), "--once",
        "--telemetry", str(RUN_DIR / "trace.jsonl"), "--seed", "1337"
    ])
    s = json.load(open(RUN_DIR / "summary.json"))
    d = json.load(open(RUN_DIR / "decision.json"))
    assert d["decision"] in {"HOLD", "LONG", "SHORT"}
    assert 0.0 <= d["confidence"] <= 1.0
    assert s["features"]["count"] > 0
    assert s["timing"]["e2d_ms"] < 150
    assert not s["datasanity"]["failed"]
    assert not s["features"]["has_nan"]
    assert not s["model"]["has_nan"]


def test_paper_loop_32_steps():
    run([
        sys.executable, "scripts/runner.py", "--profile", PROFILE, "--mode", "paper",
        "--steps", "32", "--telemetry", str(RUN_DIR / "trace.jsonl")
    ])
    # sanity: at least one e2d + one exec event present
    lines = [json.loads(l) for l in open(RUN_DIR / "trace.jsonl")]
    assert any(l.get("stage") == "e2d" for l in lines)
    assert any(l.get("stage") == "exec" for l in lines)
