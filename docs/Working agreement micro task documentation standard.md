# Working Agreement & Micro‑Task Documentation Standard

This document defines **how we work** and the **minimum documentation** required for every micro‑step. It is aimed at a beginner engineer and an engineering manager who need clarity, reproducibility, and auditability.

---

## 1) Working Agreement (How We Build)

**Principles**

- **One micro‑task at a time.** Scope ≤ 50 lines of code or a small config change.
- **Single responsibility.** Each script does one thing.
- **Idempotent.** Safe to re‑run without breaking existing outputs.
- **Traceable.** One commit per micro‑task with a descriptive message.
- **Windows‑first.** Python code is written in the IDE Visual Stage Code. The rest is done with Windows PowerShell without admin rights.
- **CI‑safe.** Code must run locally and not break the Stage 0 smoke/CI checks.
- **Documentation‑driven.** Each step is documented before/while coding.

**Standard Workflow (Step‑by‑Step)**

1. Pick the next micro‑task from the plan.
2. Create a micro‑task doc (template below) and fill the top sections (What/Why/Big Picture, Inputs/Outputs, Acceptance Criteria).
3. Implement the change (≤ 50 LOC). Include a docstring and `--help` for CLIs.
4. Run the script locally with **PowerShell commands, one line at a time**.
5. Verify outputs match Acceptance Criteria; update the doc with results.
6. Commit & push: single commit per task.
7. Ensure CI passes. Mark the micro‑task **Done**.

---

## 2) Documentation Standard (What we write every time)

For **every** micro‑task, we create a file under `docs/microtasks/` named:

```
<YYMM>-<TaskID>-<slug>.md
```

Example: `2508-MT01-setup-scaffold.md`

Place related scripts under the appropriate folder (e.g., `scripts/`, `apps/etl/`, `apps/features/`, `apps/train/`, `apps/eval/`, `apps/infer/`, `apps/verify/`, `apps/publish/`).

### Micro‑Task Doc Template (copy/paste)

````markdown
# <TaskID> – <Title>

## 1. What this script/config does (plain English)
- One or two sentences a non‑engineer can understand.

## 2. Why we need it (business & modeling rationale)
- How this contributes to project goals.

## 3. Where it fits in the big picture
- Pipeline stage: ETL → Features → Train → Infer → Verify → Publish.
- Upstream dependencies:
- Downstream consumers:

## 4. Inputs & Outputs
- **Inputs**: files, tables, environment variables, configs.
- **Outputs**: files, directories, logs, database keys.
- Storage paths (relative), partitioning and naming conventions.

## 5. Preconditions & Assumptions
- Required folders, configs, environment variables.
- Any provider availability/latency assumptions.

## 6. How it works (algorithm/logic)
- High‑level steps (3–8 bullets). Keep jargon to a minimum.
- Link to code sections if helpful.

## 7. Idempotency & Safety
- What happens if we run it twice?
- Overwrite vs. create‑if‑missing behavior.
- Checksums or schema guards when applicable.

## 8. Edge Cases & Failure Modes
- Network timeouts, empty data, malformed rows, missing columns, time zone issues.
- How the script handles/raises each case.

## 9. Performance & Cost Considerations
- Expected runtime and data size.
- Any batching/chunking decisions.

## 10. Validation & Acceptance Criteria (Definition of Done)
- File(s) created/updated where expected.
- Schema/row counts/metrics thresholds (if relevant).
- Human‑readable log message indicating success.

## 11. How to run (PowerShell — one line at a time)
```powershell
# cd to repo root first
cd "G:\My Drive\sync\air quality forecast\Git_repo"
# (add the exact run commands here)
````

## 12. Test Plan

- Unit tests added/updated (paths).
- Manual verification steps.

## 13. Version Control

```powershell
git add <files>
git commit -m "stage1: <TaskID> <short description>"
git push
```

## 14. Artifacts Produced

- Paths and filenames; include examples.

## 15. Links & References

- Related ADRs/PRD/metrics/features docs.

```

> **Manager note:** Every micro‑task must include sections 1–11 at minimum. Sections 12–15 are strongly recommended where applicable.

---

## 3) Definition of Ready / Definition of Done

**Definition of Ready (before coding):**
- Task ID, title, and owner set.
- Sections 1–5 of the template drafted.
- Acceptance Criteria written as concrete, testable checks.

**Definition of Done (before marking complete):**
- Script/config merged; ≤ 50 LOC change criterion respected (or justified).
- Successful local run with outputs verified.
- CI green.
- Template sections updated with actual results.
- Single commit with clear message.

---

## 4) Review Checklist (use before merging)
- [ ] Script has docstring and clear CLI `--help` (if applicable).
- [ ] Inputs/outputs documented with paths.
- [ ] Idempotency behavior explicit and tested.
- [ ] Edge cases listed; failure modes handled or surfaced with actionable errors.
- [ ] Logs include at least one success line and one line per major step.
- [ ] PowerShell runbook tested line‑by‑line on Windows.
- [ ] No secrets or absolute local paths in code; use env/config.
- [ ] Commit message follows `stage1: <TaskID> <action>`.

---

## 5) Naming, Locations & Conventions
- **Docs**: `docs/microtasks/` with the template above.
- **Scripts**: place under the correct `apps/` subfolder or `scripts/` for general utilities.
- **Config**: `config/` for YAML/JSON schemas. No secrets in git; use `.env`.
- **Data**: `data/` and `models/` are git‑ignored; never commit large artifacts.
- **Logging**: print one concise success line including task ID and output path(s).

**Commit message format**
```

stage1:  short, imperative description

````
Examples: `stage1: MT02 add data_sources.yaml + validator`, `stage1: MT05 parquet writer with schema guard`.

---

## 6) Example – Filled Micro‑Task Doc (MT01 – Setup Scaffold)

> **Files**: `scripts/setup_scaffold.py`, `config/local_config.yaml`, `.env.example`

**1. What:** Create initial folders (`data/raw`, `data/processed`, `artifacts`, `logs`, `config`) and seed two config files if missing.

**2. Why:** Unblocks all later tasks by standardizing paths and local configuration; ensures idempotent setup on any machine.

**3. Big Picture:** Foundation for ETL/Features/Train/Infer/Verify/Publish. Upstream: none. Downstream: all pipeline steps expect these paths.

**4. Inputs/Outputs**
- Inputs: none.
- Outputs: directories above; `config/local_config.yaml`; `.env.example`.

**5. Preconditions**
- Python 3.11 available; repository cloned locally.

**6. How it works (logic)**
- Create directories with `exist_ok=True`.
- Write default config files only if they don’t exist.
- Print final summary of created/verified resources.

**7. Idempotency & Safety**
- Re‑runs don’t overwrite existing files; only create‑if‑missing behavior.

**8. Edge Cases**
- Insufficient permissions on paths → shows clear error and aborts.

**9. Performance**
- <1s; negligible cost.

**10. Acceptance Criteria**
- Listed folders exist; two files present.
- Console prints `Scaffold OK:` with paths.

**11. How to run (PowerShell)**
```powershell
cd "G:\My Drive\sync\air quality forecast\Git_repo"
python .\scripts\setup_scaffold.py
````

**12. Test Plan**

- Manual: verify with `gi .\data, .\artifacts, .\logs, .\config, .\.env.example` (PowerShell `Get-Item`).
- (Optional) Unit: a small test that fakes a temp repo root and asserts created paths.

**13. Version Control**

```powershell
git add scripts\setup_scaffold.py config\local_config.yaml .env.example
git commit -m "stage1: MT01 setup scaffold + local config"
git push
```

**14. Artifacts**

- `data/raw/`, `data/processed/`, `artifacts/`, `logs/`, `config/local_config.yaml`, `.env.example`.

**15. Links**

- See Stage 1 plan; storage and scheduler ADRs; providers/features docs.

---

## 7) FAQ

- **Why ≤ 50 LOC?** To keep each step easy to verify and review, and to maintain velocity.
- **What if a task can’t fit?** Split it into multiple micro‑tasks, each with its own doc and acceptance criteria.
- **How do we handle secrets?** Never in code or git; use `.env` locally and cloud secrets later.

---

*End of document.*

