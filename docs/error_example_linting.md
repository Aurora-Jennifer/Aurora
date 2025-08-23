# Error-Checking Example: Linting Issues

## **Real Example: Ruff Linting Failures**

### **INPUTS for Master Error-Checking Prompt:**

**1) Failure Symptom:**
Ruff linting fails with 14,118 errors, mostly line length (E501) and import sorting (I001) issues.

**2) Evidence:**
```
ruff check . --select E,F,I,UP,B,SIM
error: unexpected argument '--format' found

  tip: to pass '--format' as a value, use '-- --format'

Usage: ruff check --select <RULE_CODE> <FILES>...

For more information, try '--help'.
Cursor.AppImage: *** [Makefile:55: lint] Error 2
```

**3) Config/Flags:**
- Makefile target: `make gates-quick`
- Ruff version: 0.6.9
- Selected rules: E,F,I,UP,B,SIM

### **EXPECTED OUTPUT (JSON):**

```json
{
  "rca": {
    "summary": "Ruff command syntax error in Makefile",
    "likely_root_cause": "Invalid --format argument in ruff check command",
    "failure_mechanism": "Ruff 0.6.9 changed CLI syntax; --format is not valid for 'check' subcommand",
    "repro_cmd": "make gates-quick"
  },
  "classification": ["tooling"],
  "blast_radius": {
    "components": ["Makefile", "CI pipeline", "pre-commit hooks"],
    "data_integrity_risks": ["none"],
    "reproducibility_risks": ["none"],
    "pnl_impact": "no"
  },
  "fix": {
    "changes": [
      "Remove --format argument from ruff check command",
      "Use ruff format --check for format validation",
      "Update pre-commit config to match new syntax"
    ],
    "flag_plan": {
      "flag": "RUFF_LEGACY_SYNTAX",
      "default": "OFF",
      "profiles": ["dev"],
      "rollout": "immediate"
    },
    "rollback": "Set RUFF_LEGACY_SYNTAX=1 to use old syntax"
  },
  "patch_unified_diff": "<<<BEGIN_DIFF
--- a/Makefile
+++ b/Makefile
@@ -54,7 +54,7 @@ coverage:
 # Run linting checks
 lint:
-	ruff check . --select E,F,I,UP,B,SIM --format=github
+	ruff check . --select E,F,I,UP,B,SIM
 
 # Format code
 format:
--- a/.pre-commit-config.yaml
+++ b/.pre-commit-config.yaml
@@ -4,7 +4,7 @@ repos:
     hooks:
       - id: ruff
-        args: [--fix, --exit-non-zero-on-fix]
+        args: [--fix, --exit-non-zero-on-fix, --select, E,F,I,UP,B,SIM]
       - id: ruff-format
 END_DIFF>>>",
  "tests": {
    "add_or_update": [
      "Add linting gate test to verify ruff syntax",
      "Update CI to validate ruff command syntax"
    ],
    "ci_commands": [
      "make lint",
      "make format-check",
      "ruff --version"
    ],
    "parity_budget": "N/A - tooling only"
  },
  "validation": {
    "pre_merge": [
      "ruff --version compatibility check",
      "make gates-quick passes",
      "pre-commit hooks install correctly"
    ],
    "ci_targets": ["lint", "format-check"],
    "negative_controls": [
      "No change to actual code logic",
      "No change to test behavior"
    ]
  },
  "release_notes": {
    "note": "Fixed ruff linting command syntax for compatibility with ruff 0.6.9",
    "rollback_note": "Set RUFF_LEGACY_SYNTAX=1 to revert to old syntax"
  },
  "open_questions": [
    "Should we pin ruff version in requirements.txt?",
    "Do we need to update other CI environments?"
  ]
}
```

### **APPLICATION:**

1. **Copy the patch** from `patch_unified_diff`
2. **Apply with git**: `git apply - < patch.diff`
3. **Test**: `make gates-quick`
4. **Validate**: All linting gates pass

### **KEY BENEFITS:**

- **Systematic approach** to error diagnosis
- **Minimal, reversible changes** behind flags
- **Clear rollback path** with environment variable
- **Comprehensive validation** with CI gates
- **Structured output** for consistent handling

### **INTEGRATION WITH AURORA FRAMEWORK:**

This error-checking prompt integrates seamlessly with our systematic review framework:

- **Pre-commit hooks**: Use for linting/formatting failures
- **CI Gates**: Apply to any gate failures
- **Weekly Review**: Track error patterns and fixes
- **Documentation**: Maintain error resolution history

The prompt ensures we follow Aurora's core principles:
- **Small, deterministic changes**
- **Reversible behind flags**
- **Comprehensive testing**
- **Clear rollback paths**
