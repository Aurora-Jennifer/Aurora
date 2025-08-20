#!/usr/bin/env python3
"""
Daily maintenance runner for Aurora.
- Preflight checks (ports, locks, env)
- Short paper + canary shadow sessions (bounded by --minutes)
- Rollups + validators
- NTFY summary + optional GitHub issue on anomalies
- Housekeeping (rotate old logs/reports)

Usage (typical):
  python tools/daily_maintenance.py --symbols SPY,TSLA --minutes 15 --quotes ibkr --ntfy
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.name


def sh(cmd: list[str], timeout: int | None = None, env: dict | None = None) -> tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return 124, out, f"[TIMEOUT] {err}"
    return p.returncode, out, err


def port_open(host: str, port: int, timeout=1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def rollup_safe():
    sh([sys.executable, "tools/rollup_posttrade.py"], timeout=120)
    sh([sys.executable, "tools/rollup_canary.py"], timeout=120)


def validate_canary() -> tuple[bool, str]:
    rc, out, err = sh([sys.executable, "tools/validate_canary.py"], timeout=60)
    ok = rc == 0
    return ok, out + err


def notify_ntfy(title: str, body: str):
    url = os.environ.get("NTFY_URL")
    if not url:
        return
    try:
        import urllib.request

        req = urllib.request.Request(url, data=body.encode("utf-8"), headers={"Title": title})
        urllib.request.urlopen(req, timeout=5).read()  # nosec B310  # URL from config; timeout set
    except Exception:
        pass  # best effort


def open_issue_if_needed(title: str, payload: dict):
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not (tok and repo):
        return
    # delegate to your existing tool (keeps behavior consistent)
    sh([sys.executable, "tools/gh_issue.py", "--title", title, "--body", json.dumps(payload)])


def rotate(paths: list[Path], days_keep: int):
    cutoff = datetime.now(UTC) - timedelta(days=days_keep)
    for p in paths:
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    ts = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                    if ts < cutoff:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,TSLA")
    ap.add_argument("--minutes", type=int, default=15, help="Duration for each session")
    ap.add_argument("--paper-profile", default="paper_strict")
    ap.add_argument("--live-profile", default="live_canary")
    ap.add_argument("--quotes", default="ibkr", choices=["ibkr", "dummy"])
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument("--ntfy", action="store_true")
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--skip-canary", action="store_true")
    ap.add_argument("--keep-days", type=int, default=14, help="Retention for logs/reports")
    args = ap.parse_args()

    # --- Preflight ---
    lock = ROOT / "runlocks" / "live.lock"
    kill = ROOT / "kill.flag"
    (ROOT / "runlocks").mkdir(exist_ok=True)
    if lock.exists():
        msg = "Session lock present; remove runlocks/live.lock to proceed."
        print(msg)
        notify_ntfy(f"{REPO}: maintenance SKIP", msg)
        sys.exit(0)
    # TWS/Gateway check if using IBKR quotes
    if args.quotes == "ibkr":
        host = os.environ.get("IB_HOST", "127.0.0.1")
        port = int(os.environ.get("IB_PORT", "7497"))
        if not port_open(host, port, 1.0):
            msg = f"IBKR Gateway not reachable at {host}:{port}"
            print(msg)
            notify_ntfy(f"{REPO}: maintenance FAIL", msg)
            sys.exit(2)
    kill.unlink(missing_ok=True)

    # Compute steps
    steps = max(1, int((args.minutes * 60) // args.poll_sec))
    summary = {"paper": {}, "canary": {}, "alerts": []}

    # --- Paper session ---
    if not args.skip_paper:
        print("[paper] starting")
        cmd = [
            sys.executable,
            "scripts/paper_runner.py",
            "--symbols",
            args.symbols,
            "--poll-sec",
            str(args.poll_sec),
            "--profile",
            args.paper_profile,
            "--steps",
            str(steps),
        ]
        if args.ntfy:
            cmd.append("--ntfy")
        rc, _, err = sh(cmd, timeout=args.minutes * 120)
        # write rollup
        rollup_safe()
        meta = read_json(ROOT / "reports/paper_run.meta.json")
        daily = read_json(ROOT / f"reports/paper_daily_{datetime.now(UTC).strftime('%Y%m%d')}.json")
        summary["paper"] = {
            "rc": rc,
            "fallbacks": meta.get("model_fallbacks", 0),
            "trades": daily.get("trades", 0),
        }
        if rc != 0 or meta.get("model_fallbacks", 0) > 0:
            summary["alerts"].append("paper_anomaly")

    # --- Canary shadow session ---
    if not args.skip_canary:
        print("[canary] starting (shadow)")
        cmd = [
            sys.executable,
            "scripts/canary_runner.py",
            "--symbols",
            args.symbols,
            "--poll-sec",
            str(args.poll_sec),
            "--profile",
            args.live_profile,
            "--quotes",
            args.quotes,
            "--shadow",
            "--steps",
            str(steps),
        ]
        rc, _, err = sh(cmd, timeout=args.minutes * 120)
        rollup_safe()
        ok, valog = validate_canary()
        meta = read_json(ROOT / "reports/canary_run.meta.json")
        daily = read_json(
            ROOT / f"reports/canary_daily_{datetime.now(UTC).strftime('%Y%m%d')}.json"
        )
        summary["canary"] = {
            "rc": rc,
            "validator_ok": ok,
            "fallbacks": meta.get("model_fallbacks", 0),
            "anomalies": meta.get("anomalies", []),
            "trades": daily.get("trades", 0),
        }
        if not ok or meta.get("model_fallbacks", 0) > 0 or meta.get("anomalies"):
            summary["alerts"].append("canary_anomaly")

    # --- Housekeeping ---
    rotate([ROOT / "logs", ROOT / "reports"], days_keep=args.keep_days)

    # --- Notify & auto-issue on problems ---
    title = (
        f"{REPO}: daily maintenance OK" if not summary["alerts"] else f"{REPO}: maintenance ALERT"
    )
    body = json.dumps(summary, indent=2)
    if args.ntfy:
        notify_ntfy(title, body)
    if summary["alerts"]:
        open_issue_if_needed(title, summary)
        print(body)
        sys.exit(3)

    print(body)
    sys.exit(0)


if __name__ == "__main__":
    main()
