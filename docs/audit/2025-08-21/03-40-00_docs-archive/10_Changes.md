# Changes

## Actions
- Archived reports → `attic/docs/reports/` (kept `REPO_AUDIT_REPORT.md`).
- Archived older session docs → `attic/docs/sessions/` (kept CRITICAL_ERRORS, NEXT_SESSION_PLAN, CHANGELOG_REFACTORING_SESSION).
- Added `attic/docs/README.md` describing archival policy.

## Commands
```bash
git mv docs/reports/<many> attic/docs/reports/
ls docs/sessions | grep -v -E "(CRITICAL_ERRORS_NEXT_SESSION|NEXT_SESSION_PLAN|CHANGELOG_REFACTORING_SESSION)" | xargs -I{} git mv -f docs/sessions/{} attic/docs/sessions/{}
```
