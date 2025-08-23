#!/usr/bin/env python3
import builtins
import contextlib
import glob
import json
import os
import statistics as stats


def read_heatmap(tsv_path: str) -> str:
    if not os.path.exists(tsv_path):
        return "No heatmap generated."
    with open(tsv_path) as f:
        lines = [line.rstrip() for line in f if line.strip()]
    if not lines:
        return "No heatmap generated."
    header = lines[0]
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return f"{header}\n{body}"


def summarize_benchmark_json() -> str:
    paths = glob.glob(".benchmarks/**/stats.json", recursive=True)
    if not paths:
        return "No benchmark stats found."
    try:
        with open(paths[0]) as f:
            data = json.load(f)
        means = [e["stats"]["mean"] for e in data.get("benchmarks", []) if "stats" in e]
        if not means:
            return "Benchmark stats unavailable."
        m = stats.mean(means)
        return f"Mean validate() time across cases: {m*1000:.2f} ms"
    except Exception:
        return "Benchmark stats unavailable."


def add_parity_line(f):
    import json
    import os
    p = "reports/experiments/parity.json"
    if os.path.exists(p):
        with open(p) as f:
            js = json.load(f)
        status = "OK" if js.get("ok") else "FAIL"
        f.write(f"\n### ONNX Parity\n{status} · max|Δ|={js.get('max_abs_diff', 0):.2e} on {js.get('nrows', 0)} rows\n")


def add_bench(f):
    import json
    import os
    p = "reports/experiments/bench.json"
    if os.path.exists(p):
        with open(p) as f:
            js = json.load(f)
        lat = js.get("latency", {})
        if "32" in lat:
            f.write(f"\n### ONNX Latency\np95@32 = {lat['32']['p95_ms']:.2f} ms\n")


def add_serve(f):
    import json
    import os
    p = "artifacts/serve/preds.jsonl"
    if not os.path.exists(p):
        return
    latencies = []
    with open(p) as fh:
        for line in fh:
            with contextlib.suppress(builtins.BaseException):
                latencies.append(json.loads(line)["lat_ms"])
    if latencies:
        latencies.sort()
        p95 = latencies[int(max(0, 0.95 * len(latencies) - 1))]
        f.write(f"\n### Serve Telemetry\np95 ≈ {p95:.2f} ms over {len(latencies)} calls\n")


def add_e2d(f):
    import json
    import os
    p = "artifacts/e2d/last/summary.json"
    if not os.path.exists(p):
        return
    with open(p) as f:
        s = json.load(f)
    ds = s.get("datasanity", {})
    f.write(f"\n### E2D\nlatency={s.get('total_latency_ms','?')} ms · ds_ok={ds.get('ok','?')} · decisions={s.get('n_decisions','?')}\n")

def add_datasanity(f):
    import json
    import os
    p = "reports/datasanity/report.json"
    if not os.path.exists(p):
        return
    with open(p) as f:
        d = json.load(f)
    ok = "OK" if d["ok"] else "FAIL"
    f.write(f"\n### DataSanity\n{ok} · failing_rules={len(d['failing_rules'])}\n")


def main():
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    content = []
    content.append("## DataSanity Summary\n")
    content.append("\n### Rule coverage heatmap\n")
    content.append("```text\n" + read_heatmap("artifacts/trace_heatmap.tsv") + "\n```\n")
    content.append("\n### Perf\n" + summarize_benchmark_json() + "\n")

    if not summary:
        print("\n".join(content))
        return
    with open(summary, "a") as f:
        f.write("\n".join(content))
        add_parity_line(f)
        add_bench(f)
        add_serve(f)
        add_e2d(f)
        add_datasanity(f)


if __name__ == "__main__":
    main()


