\# <TaskID> – <Title>



\## 1. What this script/config does (plain English)

\- One or two sentences a non-engineer can understand.



\## 2. Why we need it (business \& modeling rationale)

\- How this contributes to Stage 1 goals.



\## 3. Where it fits in the big picture

\- Pipeline stage: ETL → Features → Train → Infer → Verify → Publish.

\- Upstream dependencies:

\- Downstream consumers:



\## 4. Inputs \& Outputs

\- \*\*Inputs\*\*: files, tables, environment variables, configs.

\- \*\*Outputs\*\*: files, directories, logs, database keys.

\- Storage paths (relative), partitioning and naming conventions.



\## 5. Preconditions \& Assumptions

\- Required folders, configs, environment variables.

\- Any provider availability/latency assumptions.



\## 6. How it works (algorithm/logic)

\- High-level steps (3–8 bullets). Keep jargon to a minimum.

\- Link to code sections if helpful.



\## 7. Idempotency \& Safety

\- What happens if we run it twice?

\- Overwrite vs. create-if-missing behavior.

\- Checksums or schema guards when applicable.



\## 8. Edge Cases \& Failure Modes

\- Network timeouts, empty data, malformed rows, missing columns, time zone issues.

\- How the script handles/raises each case.



\## 9. Performance \& Cost Considerations

\- Expected runtime and data size.

\- Any batching/chunking decisions.



\## 10. Validation \& Acceptance Criteria (Definition of Done)

\- File(s) created/updated where expected.

\- Schema/row counts/metrics thresholds (if relevant).

\- Human-readable log message indicating success.



\## 11. How to run (PowerShell — one line at a time)

```powershell

\# cd to repo root first

cd "G:\\My Drive\\sync\\air quality forecast\\Git\_repo"

\# (add the exact run commands here)



