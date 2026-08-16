# Errors

Command failures and integration errors.

---

## [ERR-20260815-002] powershell_python_quoting

**Logged**: 2026-08-15T00:00:00+08:00
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
A read-only SQLite aggregation passed through nested PowerShell and Python quoting failed before execution.

### Error

```text
PowerShell ParserError: Missing argument in parameter list.
```

### Context
- The command embedded SQL strings containing single quotes inside a `python -c` string.
- No database was opened and no data was changed.

### Suggested Fix
Use a repository script or a PowerShell here-string instead of deeply nested inline quoting.

### Metadata
- Reproducible: yes
- Related Files: scripts/

---

## [ERR-20260815-001] parallel_environment_probe

**Logged**: 2026-08-15T00:00:00+08:00
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
An aggregated parallel environment probe returned exit code 1 while exposing only the successful Git branch output.

### Error

```text
Script error: Exit code: 1
Output: feat/ml-ranking
```

### Context
- Combined worktree status, local data inventory, Python dependency checks, schema reads, and threshold searches in one parallel tool call.
- The aggregation hid which child command failed.

### Suggested Fix
Run environment probes independently or make each read-only PowerShell probe explicitly tolerate missing optional paths.

### Metadata
- Reproducible: unknown
- Related Files: pyproject.toml, src/openbiliclaw/storage/database.py

---
