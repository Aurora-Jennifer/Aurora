#!/usr/bin/env python3
"""
Observability Dashboard Generator
Collates all verification results: parity, bench, E2D, ablation, backtest
"""
import json
import os
from pathlib import Path
from typing import Any


def load_json_safe(path: str) -> dict[str, Any] | None:
    """Load JSON file safely."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def format_latency(lat_ms: float) -> str:
    """Format latency with color coding."""
    if lat_ms < 50:
        color = "green"
    elif lat_ms < 100:
        color = "orange"
    else:
        color = "red"
    return f'<span style="color: {color}">{lat_ms:.2f} ms</span>'


def format_status(ok: bool) -> str:
    """Format status with emoji."""
    return "✅ PASS" if ok else "❌ FAIL"


def generate_dashboard():
    """Generate the main dashboard HTML."""

    # Load all artifacts
    parity = load_json_safe("reports/experiments/parity.json")
    bench = load_json_safe("reports/experiments/bench.json")
    e2d_summary = load_json_safe("artifacts/e2d/last/summary.json")
    ablation = load_json_safe("reports/experiments/ablation_results.json")
    backtest = load_json_safe("reports/backtest/backtest.json")

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Aurora Trading Platform - Verification Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        .status-pass {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ margin: 20px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Aurora Trading Platform</h1>
        <h2>Verification Dashboard</h2>
        <p>Generated: {os.popen('date').read().strip()}</p>
    </div>

    <div class="section">
        <h2>📊 ONNX Parity & Performance</h2>
"""

    if parity:
        max_diff = parity.get("max_abs_diff", 0)
        nrows = parity.get("nrows", 0)
        parity_ok = parity.get("ok", False)
        html += f"""
        <div class="metric">
            <strong>Parity Status:</strong> {format_status(parity_ok)}<br>
            <strong>Max |Δ|:</strong> {max_diff:.2e}<br>
            <strong>Rows:</strong> {nrows:,}
        </div>
"""

    if bench:
        latency = bench.get("latency", {})
        if "32" in latency:
            p95_32 = latency["32"].get("p95_ms", 0)
            html += f"""
        <div class="metric">
            <strong>ONNX Latency:</strong> {format_latency(p95_32)}<br>
            <strong>Batch Size:</strong> 32
        </div>
"""

    html += """
    </div>

    <div class="section">
        <h2>⚡ End-to-Decision (E2D)</h2>
"""

    if e2d_summary:
        latency = e2d_summary.get("total_latency_ms", 0)
        decisions = e2d_summary.get("n_decisions", 0)
        ds_ok = e2d_summary.get("datasanity", {}).get("ok", None)
        failing_rules = e2d_summary.get("datasanity", {}).get("failing_rules", [])

        html += f"""
        <div class="metric">
            <strong>Latency:</strong> {format_latency(latency)}<br>
            <strong>Decisions:</strong> {decisions}<br>
            <strong>DataSanity:</strong> {format_status(ds_ok) if ds_ok is not None else "⚠️ Unknown"}
        </div>
"""

        if failing_rules:
            html += f"""
        <div class="warning">
            <strong>Failing Rules:</strong> {', '.join(failing_rules[:5])}{'...' if len(failing_rules) > 5 else ''}
        </div>
"""

    html += """
    </div>

    <div class="section">
        <h2>🔬 Feature Ablations</h2>
"""

    if ablation:
        groups = ablation.get("groups", {})
        if groups:
            html += """
        <table>
            <tr><th>Feature Group</th><th>ΔIC</th><th>Status</th></tr>
"""
            for group, data in groups.items():
                delta_ic = data.get("delta_ic", 0)
                harmful = data.get("harmful", False)
                status = "❌ Harmful" if harmful else "✅ Beneficial" if delta_ic > 0 else "➖ Neutral"
                html += f"""
            <tr>
                <td>{group}</td>
                <td>{delta_ic:.4f}</td>
                <td>{status}</td>
            </tr>
"""
            html += """
        </table>
"""

    html += """
    </div>

    <div class="section">
        <h2>📈 Backtest Results</h2>
"""

    if backtest:
        metrics = backtest.get("metrics", {})
        acceptance = backtest.get("acceptance", {})

        html += f"""
        <div class="metric">
            <strong>Sharpe:</strong> {metrics.get('sharpe', 0):.3f} {format_status(acceptance.get('sharpe_ok', False))}<br>
            <strong>Max DD:</strong> {metrics.get('max_dd', 0):.1%} {format_status(acceptance.get('max_dd_ok', False))}<br>
            <strong>IC:</strong> {metrics.get('ic', 0):.3f} {format_status(acceptance.get('ic_ok', False))}
        </div>
        <div class="metric">
            <strong>CAGR:</strong> {metrics.get('cagr', 0):.1%}<br>
            <strong>Vol:</strong> {metrics.get('vol', 0):.1%}<br>
            <strong>Hit Rate:</strong> {metrics.get('hit_rate', 0):.1%}
        </div>
        <div class="metric">
            <strong>Turnover:</strong> {metrics.get('turnover', 0):.1f}x {format_status(acceptance.get('turnover_ok', False))}<br>
            <strong>Avg Win:</strong> {metrics.get('avg_win', 0):.3f}<br>
            <strong>Avg Loss:</strong> {metrics.get('avg_loss', 0):.3f}
        </div>
"""

    html += """
    </div>

    <div class="section">
        <h2>🎯 Overall Status</h2>
"""

    # Calculate overall status
    all_checks = []
    if parity:
        all_checks.append(("ONNX Parity", parity.get("ok", False)))
    if e2d_summary:
        latency = e2d_summary.get("total_latency_ms", 0)
        all_checks.append(("E2D Latency", latency <= 150))
        ds_ok = e2d_summary.get("datasanity", {}).get("ok", None)
        if ds_ok is not None:
            all_checks.append(("DataSanity", ds_ok))
    if backtest:
        acceptance = backtest.get("acceptance", {})
        all_checks.extend([
            ("Backtest Sharpe", acceptance.get("sharpe_ok", False)),
            ("Backtest Max DD", acceptance.get("max_dd_ok", False)),
            ("Backtest IC", acceptance.get("ic_ok", False))
        ])

    passed = sum(1 for _, ok in all_checks if ok)
    total = len(all_checks)

    html += f"""
        <div class="metric">
            <strong>Overall:</strong> {passed}/{total} checks passed<br>
            <strong>Status:</strong> {format_status(passed == total) if total > 0 else "⚠️ No checks"}
        </div>

        <table>
            <tr><th>Check</th><th>Status</th></tr>
"""

    for check_name, check_ok in all_checks:
        html += f"""
            <tr>
                <td>{check_name}</td>
                <td>{format_status(check_ok)}</td>
            </tr>
"""

    html += """
        </table>
    </div>

    <div class="section">
        <h2>📋 Quick Actions</h2>
        <p>
            <a href="artifacts/e2d/last/summary.json">📄 E2D Summary</a> |
            <a href="reports/experiments/parity.json">🔍 Parity Details</a> |
            <a href="reports/backtest/backtest.json">📊 Backtest Data</a> |
            <a href="reports/experiments/ablation_results.json">🧬 Ablation Results</a>
        </p>
    </div>

</body>
</html>
"""

    return html


def main():
    """Generate and save dashboard."""
    print("[DASHBOARD] Generating verification dashboard...")

    # Create output directory
    output_dir = Path("reports/dashboard")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    html = generate_dashboard()

    # Save dashboard
    dashboard_path = output_dir / "index.html"
    with open(dashboard_path, "w") as f:
        f.write(html)

    print(f"[DASHBOARD] Dashboard saved: {dashboard_path}")

    # Also save a summary for CI
    summary_path = output_dir / "summary.json"
    summary = {
        "dashboard_path": str(dashboard_path),
        "generated_at": os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip(),
        "artifacts_found": {
            "parity": os.path.exists("reports/experiments/parity.json"),
            "bench": os.path.exists("reports/experiments/bench.json"),
            "e2d": os.path.exists("artifacts/e2d/last/summary.json"),
            "ablation": os.path.exists("reports/experiments/ablation_results.json"),
            "backtest": os.path.exists("reports/backtest/backtest.json")
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DASHBOARD] Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    exit(main())
