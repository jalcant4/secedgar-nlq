---
name: secedgar
description: Research public companies using SEC EDGAR filings and financial data. Use when the user asks about 10-K/10-Q/8-K disclosures, XBRL Company Facts, financial ratios, or sourced financial research on a public company.
---

# SEC EDGAR Research

When the user asks for financial information from SEC filings, use the local
SEC research agent. Do **not** use this skill for general web news or private
company data that is not in SEC filings.

This skill is also available as the `/secedgar "<question>"` prompt template
(or `/skill:secedgar`), which routes the question here.

## What the agent does

The agent at `~/secedgar-nlq/src/secedgar_agent.py` runs an iterative Python
REPL backed by a local OpenAI-compatible model (configured in
`~/secedgar-nlq/src/config.toml`). It retrieves SEC Company Facts and filing
metadata, computes ratios, searches primary documents, and validates results
against deterministic accounting checks.

Supported question types:

- **Financial ratios** — e.g. *"What is Apple's net-income-to-debt ratio for
  its latest annual period?"* The agent picks the latest 10-K, extracts
  `NetIncomeLoss` and the chosen debt strategy components, verifies the
  arithmetic and accession provenance, and runs a debt-footnote review.
  Output is **provisional** until the footnote review is reconciled.
- **Concept discovery** — e.g. *"List all US-GAAP XBRL concepts matching net
  income and debt for company X."* The agent returns exact concept names found
  in the Company Facts API.
- **Qualitative research** — e.g. *"Investigate whether Oracle's financial
  conditions contributed to the Project Jupiter force majeure."* The agent
  retrieves and searches primary 10-K / 10-Q / 8-K documents, builds a
  source-backed narrative, and classifies the conclusion as
  `documented_causation`, `circumstantial_evidence`, or
  `insufficient_evidence`.

## Execution

Run the agent directly — no wrapper is needed;
`src/` is beside `secedgar_agent.py` so imports resolve without `PYTHONPATH`:

```
python ~/secedgar-nlq/src/secedgar_agent.py "USER QUESTION"
```

Replace `USER QUESTION` with the verbatim user question, safely quoted so
shell metacharacters are not interpolated. Use a generous overall wall-clock
timeout: a timeout means the local model did not complete inference, not that
the SEC rejected a filing.

The script exits 0 on a verified or provisionally verified result and 1 on
unsupported question types, inference timeout, or validation failure. Preserve
stdout/stderr, the exit code, and source accession IDs. Do not edit the
repository or attempt to mutate the agent's in-memory REPL from another
process; `inspect()` is for the agent's own model-generated code.

## Interpreting output

- **`=== RESULT (source checks passed; interpretation not independently verified) ===`**
  — arithmetic and accession provenance are checked. Treat cited XBRL values
  and verbatim excerpts as source-verified, but independently review any
  causal claims, date selections, or accounting interpretations.
- **`=== PROVISIONAL RESULT (accounting review required) ===`**
  — ratio arithmetic is correct but the debt footnote has not been fully
  reconciled. Report the value with the debt definition used and the
  `NEEDS ACCOUNTING REVIEW` warnings. Do **not** relabel it fully verified.
- **`=== DEBT FOOTNOTE REVIEW ===`**
  — shows retrieved debt-footnote excerpts. Use this to assess whether
  additional borrowings (leases, company-specific extensions) were missed.
- **`=== FINAL ANSWER ===`**
  — research-type questions produce a cited paragraph plus an APA References
  section using the actual SEC filing metadata.
- **`=== STALLED: identical rejected result N times ===`**
  — the model could not produce a valid result after retries. Report the
  rejection feedback and do not fabricate an answer.

If inference times out, report the local model problem and check the
llama.cpp server health; do not silently retry an identical request
repeatedly.

## Key interface rules

- `get_recent_filings(cik, "10-K")` — accession is `accessionNumber`, filing
  date is `filingDate`, report period is `reportDate`.
- XBRL observations use `accn`, `start`, `end`, `val`; `select_observation`
  returns a **dict**, not a scalar — extract the number via `obs["val"]`.
- `find_concepts(cik, metric)` — supported semantic metrics are
  `net_income_loss` and `total_interest_bearing_debt` only.
- `financial_trends(cik, periods=3)` — returns a **dict** with `rows` and
  `changes`; metric values are at `row["metrics"][name]["value"]`.
- `make_trend_indicators(trends, evidence)` — generates numeric XBRL
  indicators with `evidence_type='xbrl'` and SEC accession where the accession
  appears in retrieved SEC sources.
- `research_filings` findings are keyword hits, **not** conclusions. The
  agent interprets at most five distinct excerpts into substantive claims.
- A causal claim requires an explicit statement linking the specific event
  and financial condition in the filing text — a nearby mention of debt or
  liquidity is not sufficient.
- Missing XBRL values are **not** zero. Report them as missing.

## Output format

Always include:

1. Reporting period and filing accession / URL.
2. Value with unit (USD).
3. The debt definition used (e.g. `current_plus_noncurrent`) and whether
   the footnote was reconciled.
4. Separation of **documented facts**, **financial pressure indicators**,
   and **direct causal evidence**.
5. Limitations — state the filing-search scope and any retrieval failures.
   Do not assert an exhaustive negative search unless it actually covered
   all relevant filings.

## Prerequisites

- `~/secedgar-nlq/src/config.toml` must point to a running local
  OpenAI-compatible endpoint (`[llm].endpoint`). Check `/health` before
  launching a long task.
- The SEC contact email in `src/edgar.py` (`SEC_USER_AGENT`) should be
  valid for production use.
- Python 3.11+, `requests`.
