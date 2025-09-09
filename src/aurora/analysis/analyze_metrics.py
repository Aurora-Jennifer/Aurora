import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def warn(line: str) -> None:
    print(f"WARN: {line}")


def analyze_fold(fold: dict[str, Any], th: dict[str, float]) -> None:
    fid = fold.get("fold_id", "unknown")
    sharpe = fold.get("sharpe")
    mdd = fold.get("max_drawdown")
    win = fold.get("win_rate")
    vol = fold.get("volatility")
    if sharpe is not None and sharpe < th["sharpe_min"]:
        warn(f"fold={fid} sharpe {sharpe:.3f} < {th['sharpe_min']}")
    if mdd is not None and mdd > th["drawdown_max"]:
        warn(f"fold={fid} max_drawdown {mdd:.3f} > {th['drawdown_max']}")
    if win is not None and win < th["winrate_min"]:
        warn(f"fold={fid} win_rate {win:.3f} < {th['winrate_min']}")
    if vol is not None and vol > th["vol_max"]:
        warn(f"fold={fid} volatility {vol:.3f} > {th['vol_max']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reports", required=True)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    th = cfg.get("observability", {}).get("thresholds")
    if not th:
        print("ERROR: observability.thresholds missing in config", file=sys.stderr)
        return 2

    reports_dir = Path(args.reports)
    folds_dir = reports_dir / "folds"
    if not folds_dir.exists():
        # Fallback: analyze single run if available
        run_file = reports_dir / "run.json"
        if run_file.exists():
            try:
                run = json.loads(run_file.read_text())
                if isinstance(run.get("folds"), list):
                    for f in run["folds"]:
                        analyze_fold(f, th)
                    return 0
            except Exception:
                pass
        print("WARN: no folds found to analyze")
        return 0

    count = 0
    for p in sorted(folds_dir.glob("*.json")):
        try:
            fold = json.loads(p.read_text())
            analyze_fold(fold, th)
            count += 1
        except Exception:
            warn(f"failed to parse {p.name}")

    if count == 0:
        print("WARN: no fold files found")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


