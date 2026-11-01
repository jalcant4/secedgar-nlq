"""Iterative SEC research: LLM decides what to do, harness executes."""

import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request

from accounting import DEBT_STRATEGIES, build_debt_composition, find_concepts, review_debt_footnote
from edgar import get_cik, get_company_facts, get_filings
from filing_text import filing_url, get_filing_text, prepare_evidence, research_filings, search_filing
from financial_trends import financial_trends
from llm import llm_query
from repl import REPL
from sec_data import get_annual_observations, get_observations, get_recent_filings, normalize_concept, select_observation

try:
    import requests as _requests
    _WEB_SEARCH_AVAILABLE = True
except ImportError:
    _WEB_SEARCH_AVAILABLE = False
    _requests = None

MAX_STEPS = 30
MAX_INVALID_RESPONSES = 5


def _search_wikipedia(query: str, max_results: int = 5) -> list[dict]:
    """Search Wikipedia for background context."""
    try:
        q = urllib.parse.quote(query)
        url = (f"https://en.wikipedia.org/w/api.php?action=query&list=search&"
               f"srsearch={q}&srlimit={max_results}&format=json&origin=*")
        req = urllib.request.Request(url, headers={
            "User-Agent": "secedgar-nlq/1.0 (SEC-research-agent)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = []
            for r in data.get("query", {}).get("search", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("fullurl", ""),
                    "snippet": (r.get("snippet", "") or "")[:200],
                })
            return results
    except Exception as exc:
        return [{"title": f"wikipedia error: {exc}", "url": "", "snippet": ""}]


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML (when not rate-limited)."""
    if _requests is None:
        return []
    try:
        resp = _requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        matches = re.findall(
            r'class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>',
            resp.text, re.DOTALL
        )
        results = []
        for raw_url, title in matches[:max_results]:
            title = re.sub(r"<[^>]+>", "", title).strip()[:120]
            if "uddg=" in raw_url:
                encoded = raw_url.split("uddg=")[1].split("&")[0]
                url = urllib.parse.unquote(encoded)
            else:
                url = raw_url
            results.append({"title": title, "url": url, "snippet": ""})
        return results
    except Exception:
        return []
    finally:
        time.sleep(1)


def web_search_query(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for background context. Returns [{title, url, snippet}].

    Web results are supplementary context only — they do NOT replace SEC filings
    as primary evidence. Never cite a web result as an SEC source.
    """
    results = _search_wikipedia(query, max_results)
    if results and any(r["title"] not in ("", "No results") for r in results):
        return results
    results = _search_duckduckgo(query, max_results)
    if results:
        return results
    return [{"title": f"no results for: {query}", "url": "", "snippet": ""}]


def extract_code(response: str) -> str:
    """Extract executable Python from an LLM response."""
    match = re.search(
        r'<parameter\s+name=["\']code["\']\s*>\s*(.*?)\s*</parameter>',
        response, re.DOTALL,
    )
    if not match:
        match = re.search(r"```(?:python|py|json)?\s*\n(.*?)```", response, re.DOTALL)
    code = (match.group(1) if match else response).strip()
    # Strip a leading language tag the model sometimes emits as plain text.
    if code.lower().startswith('python\n'):
        code = code[len('python'):].lstrip('\n')
    if code.startswith('{'):
        try:
            wrapper = json.loads(code)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON code wrapper: {exc}") from exc
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get('code'), str):
            raise ValueError("JSON code wrapper must contain a string 'code' field")
        code = wrapper['code'].strip()
    if not code:
        raise ValueError("The model returned no code")
    ast.parse(code)
    return code


def is_final_response(response: str) -> bool:
    """Check whether the raw LLM response signals a terminal answer."""
    return response.strip().startswith("FINAL:")


def sec_tools() -> dict:
    """Return the SEC tool namespace available to the model in the REPL."""
    retrieved_documents = {}
    def tracked_filing_text(cik, accession, primary_document):
        document = get_filing_text(cik, accession, primary_document)
        retrieved_documents[accession] = document
        return document
    def tracked_research_filings(cik, forms=('10-K', '10-Q', '8-K'),
                                 terms=('Project Jupiter', 'force majeure', 'New Mexico',
                                        'data center', 'lease commitments', 'liquidity'),
                                 max_filings=30):
        return research_filings(cik, forms=forms, terms=terms, max_filings=max_filings,
                                fetch=tracked_filing_text)
    def tracked_debt_review(cik, accession):
        return review_debt_footnote(cik, accession, fetch=tracked_filing_text)
    return {
        "research_filings": tracked_research_filings,
        "prepare_evidence": prepare_evidence,
        "review_debt_footnote": tracked_debt_review,
        "financial_trends": financial_trends,
        "web_search": web_search_query,
        "SUPPORTED_ONTOLOGY_METRICS": ('net_income_loss', 'total_interest_bearing_debt'),
        "get_cik": get_cik, "get_filings": get_filings,
        "get_company_facts": get_company_facts,
        "get_recent_filings": get_recent_filings,
        "get_observations": get_observations,
        "get_annual_observations": get_annual_observations,
        "find_concepts": find_concepts,
        "DEBT_STRATEGIES": DEBT_STRATEGIES,
        "normalize_concept": normalize_concept,
        "select_observation": select_observation,
        "build_debt_composition": build_debt_composition,
        "get_filing_text": tracked_filing_text,
        "search_filing": search_filing,
        "filing_url": filing_url,
        "_retrieved_documents": retrieved_documents,
    }


RESPONSE_PROTOCOL = """Response protocol — two modes only:

MODE 1 — Working step: Return executable Python code inside a markdown code fence.
The harness executes the code in a persistent REPL. Use this to call SEC helpers,
inspect data, compute values, and build up state across steps.

MODE 2 — Final answer: Return a line beginning with `FINAL:` followed by your
natural-language answer. The harness accepts this as the terminal response.
No result dictionary, no validation, no prescribed structure.

Examples:
  ```python
  cik = get_cik("Apple")
  facts = get_company_facts(cik)
  print(facts["entityName"])
  ```

  FINAL:
  Apple's net-income-to-debt ratio for fiscal year 2025 was 1.24,
  based on net income of $102.0 billion and total debt of $82.5 billion.

Rules:
  - Return ONLY a code fence with Python code, or a `FINAL:` line.
  - Do not include commentary, XML, or prose outside the code fence or FINAL: line.
  - The model decides when it has sufficient evidence to stop.
"""

TOOL_REFERENCE = """SEC tools available in the REPL:
get_cik(company), get_company_facts(cik), get_recent_filings(cik, form=None),
get_observations(cik, concept, unit="USD"), get_annual_observations(cik, concept),
find_concepts(cik, metric), research_filings(cik, forms, terms, max_filings),
review_debt_footnote(cik, accession), financial_trends(cik, periods),
prepare_evidence(evidence, trends), web_search(query, max_results),
build_debt_composition(cik, period, accession, method), select_observation(observations, ...),
filing_url(cik, accession, primary_document), get_filing_text(cik, accession, primary_document).
SUPPORTED_ONTOLOGY_METRICS contains the only supported metric names for find_concepts.
Filing metadata (get_recent_filings):
  Returns a LIST of dicts. Each dict has keys:
  accessionNumber, filingDate, reportDate, form, primaryDocument.
  Use these keys directly — do NOT treat the result as a dict.
  reportDate is the reporting-period END, not the start.
  primaryDocument is the actual SEC filename; use it with filing_url().

Filing metadata (get_filings):
  Returns a dict with SEC parallel arrays (advanced use only).
  Keys: accessionNumber, filingDate, reportDate, form — each is a LIST.
  Prefer get_recent_filings() for most tasks.

SEC URLs (filing_url):
  filing_url(cik, accession, primary_document) returns the canonical SEC URL.
  Example: filing_url('0001769628', '0001769628-26-000366', 'crwv-20260630.htm')
  Do NOT construct filing URLs by manually removing hyphens from accessions.
Concept names may be unprefixed or use the us-gaap: prefix; both are accepted.
Use EXACT concept names from find_concepts as XBRL concept names.
find_concepts(cik, metric) accepts semantic metrics; get_annual_observations(cik, concept)
accepts ONLY exact US-GAAP XBRL concepts, e.g. NetIncomeLoss (not net_income_loss).
Use select_observation() to pick a specific observation from a list;
  it returns a dict with a "val" key. Observations contain val, start (for durations),
  end, form, filed, accn.
Financial helpers:
  Read the official company name from get_company_facts(cik)["entityName"];
  never invent, substitute, or hard-code a company name.
Calculate ratios using full numeric precision. Do NOT round numeric values.
Rounding is permitted ONLY in human-readable print output.
Use review_debt_footnote(cik, accession) to inspect a 10-K debt footnote.
Use build_debt_composition(cik, period=period, accession=accession, method="current_plus_noncurrent")
to get total_debt, debt_method, and debt_components. The helper raises on missing components.
Other strategies are in DEBT_STRATEGIES. Use bare concept names in debt_components.
SEC XBRL values are under facts['facts']['us-gaap'][concept]['units']['USD'].
Observations contain val, start (for durations), end, form, filed, accn.
Print only information needed for the next decision.
For large collections, print the count, keys, and a small sample.
Filter data before printing. Do not dump entire SEC submissions or Company Facts objects.
For background context outside SEC filings, call web_search(query, max_results=5).
Web results are supplementary — they provide context but NEVER replace SEC filings as primary evidence.
Never cite a web result as an SEC source.
"""


MAX_OUTPUT_CHARS = 2000


def build_prompt(question: str, history: list, last_feedback: str = "", remaining: int | None = None) -> str:
    tail = f"{last_feedback}\n" if last_feedback else ""
    recent = " ".join(history[-3:]) if history else "(none)"
    lines = [f"{RESPONSE_PROTOCOL}\n\n{TOOL_REFERENCE}\n\nQuestion:\n{question}\n\n"
             f"Recent steps and feedback:\n{tail}{recent}"]
    if remaining is not None and remaining <= 3:
        lines.append(
            f"\nOnly {remaining} execution steps remain. "
            "Use existing REPL observations to produce FINAL: if possible. "
            "If essential evidence is missing, explain that limitation "
            "rather than restarting the investigation."
        )
    else:
        lines.append(
            "Decide whether the question has been answered using the observations. "
            "If yes, return FINAL: with the completed answer. "
            "Otherwise execute one Python step to obtain missing information. "
            "Do not repeat a completed calculation or retrieval. "
            "Inspect persistent state with print(inspect()), print(inspect('latest', mode='keys')), "
            "print(inspect(mode='last_error')) when needed. "
            "Do not reset successful work."
        )
    return "".join(lines)


def _snapshot(repl: REPL) -> str:
    """Brief state snapshot: variable names, types, and key metadata."""
    vars_summary = {
        k: v.__class__.__name__
        for k, v in repl.namespace.items()
        if not k.startswith("_") and not callable(v)
    }
    return f"State: {vars_summary}"


def _observation(code: str, output: str, repl: REPL) -> str:
    """Build a rich observation from a REPL execution step."""
    lines = []
    code_tail = code[:500]
    if len(code) > 500:
        code_tail = "..." + code_tail
    lines.append(f"CODE:\n{code_tail}")
    if output.strip():
        out_tail = output[-MAX_OUTPUT_CHARS:] if len(output) > MAX_OUTPUT_CHARS else output
        lines.append(f"OUTPUT:\n{out_tail}")
    if repl.last_failed and repl.last_error:
        err_tail = repl.last_error[-MAX_OUTPUT_CHARS:]
        lines.append(f"ERROR:\n{err_tail}")
    lines.append(_snapshot(repl))
    return "\n\n".join(lines)


def _error_signature(error: str) -> str:
    """Normalize a traceback for comparison — strip line numbers and file paths."""
    # Keep the exception type and message; drop file/line info.
    lines = error.split('\n')
    sig_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Collapse file paths and line numbers: keep only the exception line.
        if stripped.startswith('File "') or stripped.startswith('  File "'):
            continue
        sig_lines.append(stripped)
    return '\n'.join(sig_lines)


def research(question: str, max_steps: int = MAX_STEPS) -> str | None:
    """Iterative SEC research: model decides what to do, harness executes."""
    repl = REPL(namespace=sec_tools())
    history: list[str] = []
    invalid_count = 0
    last_error_sig: str | None = None
    repeated_errors = 0

    while len(history) + invalid_count < max_steps:
        remaining = max_steps - len(history) - invalid_count
        prompt = build_prompt(question, history, remaining=remaining)
        try:
            response = llm_query(prompt)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"=== INFERENCE UNAVAILABLE === {type(exc).__name__}: {exc}")
            return None
        if is_final_response(response):
            answer = response.strip()
            print(answer)
            return answer
        try:
            code = extract_code(response)
        except (SyntaxError, ValueError) as exc:
            invalid_count += 1
            feedback = f"Invalid Python: {exc}. Return only executable Python."
            history.append(feedback)
            print(f"INVALID CODE: {exc}", flush=True)
            if invalid_count >= MAX_INVALID_RESPONSES:
                print(f"Research stopped: {MAX_INVALID_RESPONSES} consecutive invalid responses.")
                return None
            continue
        output = repl.execute(code)
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if repl.last_failed:
            error_sig = _error_signature(repl.last_error or output)
            if error_sig == last_error_sig:
                repeated_errors += 1
            else:
                repeated_errors = 1
                last_error_sig = error_sig
            if repeated_errors >= 3:
                feedback = (
                    "Repeated execution failure. Stop repeating this operation. "
                    "Inspect the relevant variable types, use a different method, "
                    "or return FINAL: with the evidence already obtained."
                )
                history.append(feedback)
                print(f"REPEATED ERROR: {feedback}", flush=True)
                last_error_sig = None
                repeated_errors = 0
            else:
                err_tail = (repl.last_error or output)[-MAX_OUTPUT_CHARS:]
                feedback = f"Execution failed: {err_tail}"
                history.append(_observation(code, output, repl))
        else:
            last_error_sig = None
            repeated_errors = 0
            history.append(_observation(code, output, repl))

    print(f"Research stopped after {max_steps} steps without a final answer.")
    return None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: python secedgar_agent.py "Your SEC research question"')
    research(question)
