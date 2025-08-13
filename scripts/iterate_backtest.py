#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

BASELINE = Path("config/baseline.json")
RESULTS_DIR = Path("results/backtest")

IDEAS = [
    {
        "desc": "Increase confidence_threshold by +0.05",
        "path": ["strategy_params", "regime_aware_ensemble", "confidence_threshold"],
        "delta": 0.05,
    },
    {
        "desc": "Increase rolling_window by +20",
        "path": ["strategy_params", "regime_aware_ensemble", "rolling_window"],
        "delta": 20,
    },
]
TARGET_WIN_RATE = 0.35  # stop when overall win rate >= 35%
MAX_ITERS = 2


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def apply_delta(cfg_obj: dict, path_keys: list, delta):
    cur = cfg_obj
    for k in path_keys[:-1]:
        cur = cur.setdefault(k, {})
    last = path_keys[-1]
    cur[last] = type(cur.get(last, 0))(cur.get(last, 0)) + delta


def run_backtest(start: str, end: str, symbols: list, config_file: str):
    cmd = (
        ["python", "backtest.py", "--start-date", start, "--end-date", end, "--symbols"]
        + symbols
        + ["--config", config_file]
    )
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def main():
    baseline = read_json(BASELINE)
    base_cfg_path = Path(baseline["config_file"])  # do not modify
    start, end = baseline["start_date"], baseline["end_date"]
    symbols = baseline["symbols"]

    for i, idea in enumerate(IDEAS[:MAX_ITERS], start=1):
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_cfg_path = Path(f"config/{run_id}.json")
        cfg = read_json(base_cfg_path)
        apply_delta(cfg, idea["path"], idea["delta"])
        write_json(run_cfg_path, cfg)

        ok = run_backtest(start, end, symbols, str(run_cfg_path))
        if not ok:
            print("Backtest failed", file=sys.stderr)
            sys.exit(2)

        tag_dir = Path(f"results/{run_id}")
        tag_dir.mkdir(parents=True, exist_ok=True)
        for name in ["trades.csv", "ledger.csv", "summary.txt", "results.json"]:
            src = RESULTS_DIR / name
            if src.exists():
                shutil.copy(src, tag_dir / name)

        # postmortem returns printed output; we rely on run_log.csv for win rate
        subprocess.run(
            ["python", "scripts/postmortem.py", run_id, idea["desc"]], check=True
        )

        # inspect run_log.csv for last line's win_rate
        import pandas as pd

        log_df = pd.read_csv("results/run_log.csv")
        win_rate = float(log_df.tail(1)["win_rate"].iloc[0])
        print(f"Iteration {i}: win_rate={win_rate:.3f}")
        if win_rate >= TARGET_WIN_RATE:
            print("Target win rate achieved; stopping iterations.")
            break

    print("Done iterations.")


if __name__ == "__main__":
    main()
