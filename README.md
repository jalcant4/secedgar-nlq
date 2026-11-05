SEC EDGAR Natural-Language Query Agent

⚠️ secedgar-nlq is a proof of concept, not a production system.

1. It converts a natural-language query into SEC EDGAR API queries that are executed inside a Python environment (a persistent REPL driven by a local LLM).
2. It enforces the LLM to programmatically decompose the question and produce a programmatic solution — Python code against the SEC tooling — rather than answering from memory.
3. The inference is programmatically correct: arithmetic, accession provenance, and result contracts are checked deterministically. But the final answer is still natural-language output, which is not programmatically ensured. The validator cannot guarantee that the prose faithfully represents the computed values, or that interpretations and causal claims are sound.
4. Output requires grounding — tying every claim to a specific retrieved SEC source — which is currently not supported as-is. Citations are checked for provenance, but end-to-end grounding of the narrative is not enforced.

Do not rely on its output for financial or legal decisions without independent verification against the underlying SEC filings.

A local Python agent for retrieving, calculating, and investigating public-company SEC EDGAR disclosures, backed by a configurable OpenAI-compatible local llama.cpp endpoint.

How it works

The agent loop is: LLM proposes → harness executes → LLM observes → repeat.

* The LLM may only respond with Python code (executed in a persistent REPL with SEC tools injected) or a FINAL: answer.
* Code runs against real SEC EDGAR APIs (edgar.py, sec_data.py, filing_text.py) with deterministic helpers for ratios, debt composition, and filing search.
* Result contracts are validated programmatically (arithmetic, accession matching, period consistency); failures are fed back to the model as observations.
* A DRAFT: → verification → FINAL: protocol asks the model to submit an answer for deterministic checking before it is accepted.

What is checked: numeric arithmetic, accession/report-date provenance, debt-composition consistency, and the shape of final answers.

What is not checked: the natural-language interpretation of results, causal claims, or that the final prose matches the verified numbers (see the Proof of Concept disclaimer above).

Layout

src/
  secedgar_agent.py   # CLI, orchestration, completion contracts, citations
  llm.py              # local model client
  repl.py             # persistent Python REPL
  edgar.py            # SEC HTTP / CIK / Company Facts
  sec_data.py         # filing metadata and XBRL observations
  accounting.py       # debt methods, validation, footnote review
  filing_text.py      # filing retrieval, search, evidence preparation
  financial_trends.py # accession-matched financial trends
  config.toml.example # settings template (copy to config.toml, gitignored)
tests/               # offline pytest suite
docs/pi-skill.md     # Pi skill install notes
skills/secedgar/     # Pi skill (SKILL.md)
skills/prompts/      # Pi prompt template for the /secedgar command

Install and run

Requires Python 3.11+, requests, and pytest. The real src/config.toml is gitignored so personal details never enter the repository — create it from the template:

cp src/config.toml.example src/config.toml

Set [llm].endpoint and [user].email.

[user].email is used as the SEC User-Agent contact (override with the SEC_CONTACT_EMAIL environment variable). Point [llm].endpoint at an active OpenAI-compatible server.

python -m pip install requests pytest
python -m pytest -q
python src/secedgar_agent.py "What is CoreWeave's net-income-to-debt ratio for its latest annual period?"
python src/secedgar_agent.py "Investigate Oracle's financial conditions and Project Jupiter force majeure."

A timeout means inference did not complete, not that the SEC rejected a filing. The agent terminates cleanly on inference errors; monitor the llama.cpp server if generation is slow.

Pi integration

See docs/pi-skill.md.

The repository ships a Pi skill (skills/secedgar/SKILL.md) and a prompt template (skills/prompts/secedgar.md) that expose the agent as the /secedgar "<question>" command.

The skill captures exit status and bounded output and enforces evidence rules rather than relying solely on a skill prompt.

Key interface rules

* get_recent_filings(cik, "10-K") — accession is accessionNumber, filing date is filingDate, report period is reportDate.
* XBRL observations use accn, start, end, val; select_observation returns a dict, not a scalar — extract the number via obs["val"].
* find_concepts supports net_income_loss and total_interest_bearing_debt only.
* financial_trends(cik, periods=3) returns a dict with rows and changes; metric values are at row["metrics"][name]["value"].
* review_debt_footnote retrieves passages but does not certify that all borrowings, lease obligations, or company-specific XBRL extensions were accounted for. Ratio results stay provisional until the footnote is reconciled.
* research_filings findings are keyword hits, not conclusions. Missing XBRL values are not zero.

Tests and static checks

python -m pytest -q
python -m compileall -q src tests
ruff check src tests

The tests are offline and use synthetic fixtures; they do not test live SEC availability, model quality, or accounting exhaustiveness.

License

secedgar-nlq is licensed under the PolyForm Noncommercial License 1.0.0.

This project is source-available and may be used, modified, and distributed for purposes permitted by the license, including personal use, noncommercial experimentation, and qualifying educational and research activities.

Commercial use

Commercial use outside the purposes permitted by PolyForm Noncommercial 1.0.0 requires a separate written license from the copyright holder.

This includes commercial financial research, business-to-business consulting, integration into commercial products, and paid services when those activities are not otherwise permitted by the license.

There is no automatic commercial-use exemption based on annual business revenue.

For commercial licensing inquiries, please open an issue in this repository.

Disclaimer

This project is experimental software and is provided without warranties, as specified in the license.

Its outputs should be independently verified against authoritative SEC filings before use in financial, investment, or legal decisions.

See the repository’s LICENSE file for the complete legally operative terms.
