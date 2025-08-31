# Way of Working (You & ChatGPT)
_Last updated: 2025-08-31_

## Principles
- **Deliver in complete folders**, not micro-snippets. For Stage 2+, you’ll request:  
  “Create Stage 2 folder for <scope>” → I deliver a ready-to-drop folder (code + runner + config + docs).
- **Windows + PowerShell** workflow; code authored/edited in **VS Code**; Python **3.11** venv at `C:\aqf311\.venv`.
- **Debug in reasonable chunks** (a script/module per iteration), not line-by-line.

## Request pattern (Stage 2+)
1) You specify scope/constraints (APIs, SLAs, data sources, NFRs).  
2) I deliver:
   - A full `stage2/` (or feature) folder with code, `scripts\` runner, config, and docs.
   - Optionally a zip or a PowerShell bootstrap to reconstruct locally.
3) We validate by running the runner end-to-end; on errors, you paste the last ~20 lines (incl. `CMD:`); we patch targeted files.

## Branch & commit policy
- Branches: `feat/<name>`, `fix/<name>`, `chore/<name>`  
- Push with `git push -u origin HEAD` (avoids placeholder branch mistakes).  
- Commits: imperative messages, scoped (e.g., `feat(stage2): add provider X`).

## Environment conventions
- `.env`: `DATA_ROOT, MODELS_ROOT, CACHE_ROOT, OPENAQ_API_KEY` (secrets **not** committed).  
- Paths with spaces must be quoted; keep heavy outputs under `C:\aqf311\...` (not in Drive-synced paths).  
- Corporate proxies: set `HTTP_PROXY`/`HTTPS_PROXY` before `pip`/HTTP calls when needed.

## Quality bar
- Runs on Python 3.11; **pyarrow** for Parquet; timestamps in **UTC**.  
- Clear errors for 401/429/timeouts; polite retry where applicable.  
- Every delivery includes: runner script, docs, and commands to run/debug/inspect outputs.

## Documentation expectations
- Update `docs/` with:
  - Overview (human + system/ChatGPT)  
  - Commands cheat-sheet  
  - Provider-specific notes/NFRs (as needed)
