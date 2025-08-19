#!/usr/bin/env python
import json
import os
import sys
import urllib.request


def create_issue(repo: str, title: str, body: dict, token: str) -> int:
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = {
        "title": title,
        "body": "```\n" + json.dumps(body, indent=2) + "\n```",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "aurora-paper-bot",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read().decode())
    return int(resp.get("number", 0))


def main():
    repo = os.getenv("GITHUB_REPOSITORY") or (len(sys.argv) > 1 and sys.argv[1]) or ""
    token = (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or (len(sys.argv) > 2 and sys.argv[2])
        or ""
    )
    if not repo or not token:
        print("missing repo or token; skipping")
        return 0
    meta = json.loads(open("reports/paper_run.meta.json").read())
    fallbacks = int(meta.get("model_fallbacks", 0))
    tw = meta.get("model_tripwire") or meta.get("model_tripwire_turnover")
    if fallbacks > 0 or tw:
        title = f"[paper] anomalies: fallbacks={fallbacks} tripwire={bool(tw)}"
        body = {"meta": meta, "reason": tw or {"fallbacks": fallbacks}}
        num = create_issue(repo, title, body, token)
        print(f"opened issue #{num}")
    else:
        print("no anomalies; no issue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
