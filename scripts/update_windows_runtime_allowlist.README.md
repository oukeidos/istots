# Windows Runtime Allowlist Updater

This script is a standalone maintenance tool for the Windows managed
`llama.cpp` allowlist.

It is intentionally separate from the main application runtime path. The app
does not import this script. The script reads the committed allowlist source
file, keeps its own local test ledger, and only edits the allowlist file when
you explicitly run the `apply` command.

The tool lives at:

```text
scripts/update_windows_runtime_allowlist.py
```

## What It Does

The updater has two commands:

1. `scan`
   - looks at recent upstream `ggml-org/llama.cpp` releases
   - checks only the supported Windows families:
     - `x64/cpu`
     - `x64/vulkan`
     - `x64/cuda12`
   - skips tags already in the committed allowlist
   - skips tags already tested in the local ledger
   - validates only as many new candidates as needed for the configured
     per-family target, within a bounded attempt budget
   - records local pass/fail results under a gitignored artifact directory

2. `apply`
   - reads the local ledger
   - takes the top pending passing tags per family
   - prepends them into
     `src/istots/gui/windows_runtime_allowlist.py`

## Why It Is Bounded

This tool is designed for occasional manual maintenance, not continuous
monitoring.

A normal run intentionally stops early:

- it scans only a recent upstream window
- it does not sweep the whole release history
- it stops once each family reaches the configured promotion target for that
  run
- it also stops if the per-family or global attempt budget is exhausted

That keeps the maintenance cost predictable for a solo project.

## Requirements

- Run it on Windows.
- Run it from the repository root.
- Use the project `uv` environment.
- Expect it to download and probe official Windows `llama.cpp` release assets.

The `scan` command is Windows-only because it validates Windows
`llama-server.exe` binaries locally.

## Artifact Location

The updater writes local-only outputs under:

```text
build/windows_runtime_allowlist_automation/
```

This directory is gitignored.

Important files:

- `ledger.json`
  - persistent local memory of already tested `tag + variant` results
- `latest_scan_summary.md`
  - latest human-readable scan summary
- `latest_scan_summary.json`
  - latest machine-readable scan summary
- `latest_apply_summary.md`
  - latest human-readable apply summary
- `latest_apply_summary.json`
  - latest machine-readable apply summary
- `runs/<run-id>/summary.md`
  - per-run scan summary archive

## Cleanup Behavior

The updater does not keep downloaded runtime archives or extracted runtime
directories after validation.

During `scan`:

- each candidate is downloaded into a temporary per-candidate work directory
- the extracted runtime is probed there
- that temporary directory is removed immediately after the candidate finishes
- the per-run `work/` directory is removed at the end of the scan
- stale `work/` directories from interrupted earlier runs are removed at the
  start of the next scan

The persistent artifact directory keeps summaries and the ledger only.

## Typical Workflow

### 1. Scan Recent Releases

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan
```

This uses the default policy:

- recent release window: `120` days
- overlap from the previous successful scan: `14` days
- upstream release cap: `40`
- promotion target per family: `3`
- attempt budget per family: `8`
- global attempt budget: `18`

### 2. Read the Summary

After the scan, inspect:

```text
build/windows_runtime_allowlist_automation/latest_scan_summary.md
```

That file tells you:

- which releases were considered
- which candidates were actually tested
- which ones passed
- which pending passing tags are now available for promotion

### 3. Apply the Top Pending Passing Tags

```bash
uv run python scripts/update_windows_runtime_allowlist.py apply
```

This updates:

```text
src/istots/gui/windows_runtime_allowlist.py
```

It only applies pending local passing tags that are not already in the
committed allowlist.

## Useful Options

### Per-family target

Collect or apply fewer tags for one family:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan ^
  --target x64/cpu=2 ^
  --target x64/vulkan=2 ^
  --target x64/cuda12=1
```

The same override format also works with `apply`.

### Per-family attempt budget

Lower the max number of local validation attempts for one family:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan ^
  --attempt-budget x64/cpu=4 ^
  --attempt-budget x64/vulkan=6 ^
  --attempt-budget x64/cuda12=4
```

### Recent window and release cap

Tighten or widen one manual scan:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan ^
  --lookback-days 90 ^
  --release-limit 25
```

### Global attempt budget

Bound total local validation work for one run:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan ^
  --global-attempt-budget 12
```

## Safety Notes

- `scan` does not modify the committed allowlist source file.
- `apply` is the only command that edits the committed allowlist.
- The local ledger is advisory memory, not source control truth.
- If you want to start fresh, you can remove
  `build/windows_runtime_allowlist_automation/` and run `scan` again.

## Quick Commands

Default scan:

```bash
uv run python scripts/update_windows_runtime_allowlist.py scan
```

Default apply:

```bash
uv run python scripts/update_windows_runtime_allowlist.py apply
```
