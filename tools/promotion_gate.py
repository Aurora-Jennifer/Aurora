#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def load_thresholds(cfg_path: Path) -> dict:
    cfg = load_yaml(cfg_path)
    return cfg.get("thresholds", {})


def load_metrics() -> dict:
    candidates = [
        Path("results/run.json"),
        Path("artifacts/multi_symbol/summary_results.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


def check_returns_type_is_percent(base_cfg: Path) -> bool:
    try:
        cfg = load_yaml(base_cfg)
        rt = cfg.get("data", {}).get("returns", {}).get("type", "percent")
        return rt == "percent"
    except Exception:
        return False


def has_quarantined_records() -> bool:
    q = Path("quarantine")
    if not q.exists():
        return False
    return any(p.is_file() for p in q.rglob("*"))


def invariance_smoke_test() -> float:
    # Synthetic additive shift correlation on random series
    import numpy as np
    import pandas as pd

    from core.metrics.returns import percent_returns

    rng = np.random.default_rng(42)
    base = 100 + np.cumsum(rng.normal(0, 1.0, 500))
    s = pd.Series(base)
    r1 = percent_returns(s)
    r2 = percent_returns(s + 50.0)
    corr = np.corrcoef(r1.values, r2.values)[0, 1]
    return float(corr)


def write_promotion_report(payload: dict) -> Path:
    date_dir = Path("docs/audit") / datetime.utcnow().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    report = date_dir / "promotion_report.md"
    lines = [
        "# Promotion Gate Report\n",
        "\n## Metrics\n",
        json.dumps(payload.get("metrics", {}), indent=2),
        "\n\n## Thresholds\n",
        json.dumps(payload.get("thresholds", {}), indent=2),
        "\n\n## Checks\n",
        json.dumps(payload.get("checks", {}), indent=2),
    ]
    report.write_text("\n".join(lines))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Aurora Promotion Gate")
    parser.add_argument("--fail-on-quarantine", action="store_true")
    parser.add_argument("--report", action="store_true", help="Emit promotion_report.md")
    args = parser.parse_args()

    th = load_thresholds(Path("config/promotion.yaml"))
    m = load_metrics()
    if not m:
        print("No metrics found; failing gate")
        return 1

    sharpe = float(m.get("sharpe", m.get("Sharpe", 0.0)))
    max_dd = float(m.get("max_drawdown", m.get("MaxDrawdown", 1.0)))
    cagr = float(m.get("cagr", m.get("CAGR", 0.0)))
    oos_win = float(m.get("oos_win_rate", m.get("OOSWinRate", 0.0)))

    corr = invariance_smoke_test()
    returns_type_ok = check_returns_type_is_percent(Path("config/base.yaml"))
    quarantined = has_quarantined_records()

    checks = {
        "sharpe_ok": sharpe >= float(th.get("min_sharpe", 0.9)),
        "max_dd_ok": max_dd <= float(th.get("max_drawdown", 0.25)),
        "cagr_ok": cagr >= float(th.get("min_cagr", 0.05)),
        "oos_win_ok": oos_win >= float(th.get("min_oos_win_rate", 0.52)),
        "add_corr_ok": corr >= float(th.get("min_add_corr", 0.98)),
        "returns_type_percent": returns_type_ok,
        "no_quarantine": (not quarantined) or (not args.fail_on_quarantine),
    }

    passed = all(checks.values())

    output = {
        "pass": passed,
        "metrics": {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "cagr": cagr,
            "oos_win_rate": oos_win,
            "additive_corr_smoke": corr,
        },
        "thresholds": th,
        "checks": checks,
    }
    print(json.dumps(output, indent=2))

    if args.report:
        path = write_promotion_report(output)
        print(f"report: {path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
