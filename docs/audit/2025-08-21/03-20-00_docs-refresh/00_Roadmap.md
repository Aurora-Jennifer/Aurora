# Docs Refresh — Roadmap (2025-08-21 03:20)

**Prompt:** "Finish documentation cleanup; update and add docs to reflect current DataSanity architecture (engine switch, telemetry, canary)."

## Context
- DataSanity production rollout foundation landed (engine switch facade, telemetry JSONL, metrics, canary runner).
- Prior cleanup removed legacy docs; summaries and runbooks must reflect new architecture.

## Plan (now)
1) Update `docs/summaries/core.md` with Purpose, Entrypoints, Do-not-touch, and API list.
2) Add `docs/summaries/data_sanity.md` with Mermaid diagram and key APIs.
3) Add `docs/runbooks/datasanity_ops.md` covering ops procedures, telemetry paths/rotation, and regression triage.
4) Append today’s EOD timeline.

## Success criteria
- Summaries include a diagram and a concrete API section.
- Runbook provides actionable steps and paths.
- Docs-only minimal diffs.
