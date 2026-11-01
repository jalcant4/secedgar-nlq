# Suggested Pi skill: SEC EDGAR NLQ

The repository contains no existing Pi skill file to patch, so adapt the following into your Pi agent's skill location and registration format. Keep this document as the authoritative behavior contract, not an assumption that Pi automatically discovers it.

## Trigger

Use for user questions that require SEC EDGAR 10-K, 10-Q, 8-K, XBRL Company Facts, or sourced financial disclosures. Do not invoke for general web news or private contracts that are not SEC filings.

## Execution

1. Verify that `src/config.toml` points to a running local llama.cpp OpenAI-compatible endpoint. Check `/health` before launching a long task.
2. Run `python /absolute/path/to/secedgar-nlq/src/secedgar_agent.py "<verbatim user question>"` using Pi's command tool with a generous **overall** wall-clock budget. Pass the question as one safely quoted argument; do not interpolate shell metacharacters unsafely.
3. Preserve stdout/stderr, process exit code, and source accession IDs. Avoid treating a printed provisional result as verified accounting.
4. If the agent says `PROVISIONAL RESULT`, report its arithmetic and the `NEEDS ACCOUNTING REVIEW` warning. Never relabel it fully verified.
5. If it says `RESULT (source checks passed; interpretation not independently verified)`, treat cited excerpts/XBRL values as checked for provenance, but independently review substantive interpretations, causal claims, and dates.
6. If inference times out, report the local model problem and inspect server health; do not fabricate an answer or silently retry an identical request repeatedly.

## Key interface rules

- `get_recent_filings`: accession is `accessionNumber`, filing date is `filingDate`, report period is `reportDate`.
- XBRL observations use `accn`, `start`, `end`, `val`; `select_observation` returns a **dict**, not a scalar.
- `find_concepts`: supported semantic metrics are `net_income_loss` and `total_interest_bearing_debt` only.
- `financial_trends(cik, periods=3)`: returns a **dict**, with `rows` and `changes`; metric values are nested at `row['metrics'][name]['value']`.
- `make_trend_indicators(trends, evidence)`: generates numeric XBRL indicators with `evidence_type='xbrl'` and SEC accession; these do not require invented verbatim text excerpts.
- `research_filings` findings are keyword hits, **not** conclusions.
- A project-specific causal claim requires explicit evidence linking the specific event and financial condition, not a nearby mention of debt or force majeure.

## Output

Present reporting period, value/unit, filing accession and URL, calculation, and limitations. Separate **documented facts**, **financial pressure indicators**, and **direct causal evidence**. For annual debt ratios, disclose which debt definition was used and whether the debt footnote has been reconciled. Do not assert a negative SEC disclosure search was exhaustive unless the search actually covered all relevant filings.

## Recommended Pi tool wrapper

Use Pi's native command tool with an argument array (rather than `shell=True`), an absolute script path, a bounded stdout capture, and an explicit timeout. The skill should not edit the repository or mutate the persistent REPL state from another process.

### Inspecting the internal REPL

The SEC agent exposes `inspect()` to its own model-generated Python code.
Pi should not attempt to access the agent's in-memory REPL directly. When a
run fails, ask the SEC agent to inspect `mode="last_error"` or
`mode="validation"`, inspect dictionary keys, and reuse persistent variables
rather than restarting. The agent's result validator remains authoritative
for acceptance; an inspection result is diagnostic, not financial evidence.
