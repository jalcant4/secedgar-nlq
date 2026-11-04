# SEC EDGAR Natural-Language Query Agent

A local Python REPL agent for retrieving, calculating, and investigating public-company SEC EDGAR disclosures. It uses a configurable OpenAI-compatible local `llama.cpp` endpoint and deterministic validation of supported result contracts.

## Layout

```text
src/
  secedgar_agent.py   # CLI, orchestration, completion contracts, citations
  llm.py              # local model client
  repl.py             # persistent Python REPL
  edgar.py            # SEC HTTP / CIK / Company Facts
  sec_data.py         # filing metadata and XBRL observations
  accounting.py       # debt methods, validation, footnote review
  filing_text.py      # filing retrieval, search, evidence preparation
  financial_trends.py # accession-matched financial trends
  config.toml         # local model endpoint
 tests/               # offline pytest suite
 docs/pi-skill.md     # Pi skill install notes
 skills/secedgar/     # Pi skill (SKILL.md)
 skills/prompts/      # Pi prompt template for the /secedgar command
```

This refactor folds `debt_review.py` into `accounting.py` and `evidence_synthesis.py` into `filing_text.py`. No shim modules are necessary. Imports referencing the old names must be updated.

## Install and run

Requires Python 3.11+ and `requests` and `pytest`. Edit `src/config.toml` to point to an active OpenAI-compatible server; configure a valid SEC contact in the client before production use.

```bash
python -m pip install requests pytest
python -m pytest -q
python src/secedgar_agent.py "What is CoreWeave's net-income-to-debt ratio for its latest annual period?"
python src/secedgar_agent.py "Investigate Oracle's financial conditions and Project Jupiter force majeure."
```

`llm.py` requests `/models` and `/chat/completions`. A timeout means inference did not complete, not that the SEC rejected a filing. The agent now terminates cleanly on inference errors; monitor the llama.cpp server if generation is slow.

## Contracts and evidence

**Ratio:** Use `get_recent_filings(cik, "10-K")` and exact `accessionNumber`/`reportDate`; `select_observation` returns a dict with `val`. `build_debt_composition` supports alternative debt representations. Its arithmetic and accession matching are checked, but a successful result remains **provisional** until debt-note coverage is reconciled. `review_debt_footnote` retrieves passages and identifies categories; it does **not** certify that all borrowings, lease obligations, or company-specific XBRL extensions were accounted for. Do not add lease balances to debt without assessing classification and double counting.

**Research:** `research_filings` returns raw search hits, not claims. `financial_trends` returns a dictionary with `rows`, `changes`, `units`, and `notes`. Each row has `period`, `accession`, `metrics` (e.g. `capital_expenditures: {value, concept}`), and `missing`. `make_trend_indicators(trends, evidence)` generates numeric XBRL indicators with period and accession, but only where that accession appears in retrieved SEC sources. The harness checks the numeric indicator against the live REPL trend data. For text-based findings, provide an exact verbatim excerpt and the matching SEC accession; the harness checks it against retrieved document text. Model interpretation and causal claims are **not** independently fact-checked by the validator.

The ontology `find_concepts` only accepts `net_income_loss` and `total_interest_bearing_debt`. These are **not** the `financial_trends` keys, and derived `free_cash_flow` is not an XBRL concept. Free cash flow = operating cash flow minus capital expenditures only when both exist for the same accession and annual period. Missing values remain missing. `debt_carrying_amount` is not automatically exhaustive interest-bearing debt.

For causal research, distinguish reported financial conditions from evidence directly linking them to an event. A generic liquidity or data-center keyword match does not establish project-level causation. State filing-search scope and retrieval failures when making negative claims.

## Tests and static checks

```bash
python -m pytest -q
python -m compileall -q src tests
python -m pip install ruff
ruff check src tests
```

The local tests are offline and use synthetic fixtures; they do not test live SEC availability, model quality, or accounting exhaustiveness.

## Pi integration

See [docs/pi-skill.md](docs/pi-skill.md). The repository ships a Pi skill
(`skills/secedgar/SKILL.md`) and a prompt template (`skills/prompts/secedgar.md`)
that exposes the agent as the `/secedgar "<question>"` command. The skill
invokes the CLI, captures exit status and bounded output, and enforces
evidence rules rather than relying solely on a skill prompt.

## REPL inspection and observation

The REPL maintains variables across model steps. `inspect()` is a read-only
helper injected into its namespace, allowing the model to inspect values
without guessing their structure:

```python
print(inspect())                          # variable names and types
print(inspect("latest", mode="keys"))    # dictionary keys
print(inspect("latest"))                 # bounded repr preview
print(inspect("latest", mode="type"))    # Python type
print(inspect(mode="last_error"))        # latest execution traceback
print(inspect(mode="validation"))        # most recent result validation
```

Inspection does not change existing variables. The preview and key counts are
bounded; it is not a security sandbox and does not prevent malicious Python
from modifying the process. The agent now preserves the tail of tracebacks in
recent observations, and stores validator errors for explicit inspection.
This reduces repeated guesses about SEC filing keys (`accessionNumber`, not
`accession` or `accn`) and supports repairing a result without recomputing it.

Run `python -m pytest -q` and `ruff check src tests` after changes. Offline
unit tests do not establish live SEC availability or accounting completeness.
