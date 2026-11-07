"""Iterative SEC research: LLM decides what to do, harness executes."""

import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from accounting import (
    DEBT_STRATEGIES,
    build_debt_composition,
    find_concepts,
    review_debt_footnote,
)
from compare import compare_debt_ratios
from edgar import get_cik, get_company_facts, get_filings
from filing_text import (
    filing_url,
    get_filing_text,
    prepare_evidence,
    research_filings,
    search_filing,
)
from financial_trends import financial_trends
from llm import llm_query
from repl import REPL
from sec_data import (
    get_annual_observations,
    get_observations,
    get_recent_filings,
    normalize_concept,
    select_observation,
)

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
    except (OSError, ValueError):
        # Return no results so web_search_query falls through to the
        # DuckDuckGo fallback instead of surfacing an error record as a hit.
        return []


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
    except (OSError, ValueError):
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


def is_draft_response(response: str) -> bool:
    """Check whether the raw LLM response is a draft for verification."""
    return response.strip().startswith("DRAFT:")


def extract_draft(response: str) -> str:
    """Extract the draft content after the DRAFT: marker."""
    stripped = response.strip()
    if stripped.startswith("DRAFT:"):
        return stripped.removeprefix("DRAFT:").strip()
    return stripped.strip()


@dataclass(frozen=True)
class Evidence:
    """Lightweight provenance record for SEC research findings."""
    claim: str
    value: Any
    source: str
    method: str = "retrieved"  # "retrieved" | "derived" | "inferred" | "unsupported"


def verify_claim(claim_dict: dict, source_list: list[dict], key_field: str = "accessionNumber") -> dict:
    """Compare claimed values against original retrieved records.

    Returns a dict mapping each claimed key to a verification result:
      {key: {"match": bool, "expected": <claimed>, "actual": <retrieved>}}
    """
    results = {}
    for key, claimed_value in claim_dict.items():
        found = None
        for record in source_list:
            if record.get(key_field) == claim_dict.get(key_field):
                found = record
                break
        actual_value = found.get(key) if found else None
        results[key] = {
            "match": actual_value == claimed_value,
            "expected": claimed_value,
            "actual": actual_value,
        }
    return results


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
        "compare_debt_ratios": compare_debt_ratios,
        "get_filing_text": tracked_filing_text,
        "search_filing": search_filing,
        "filing_url": filing_url,
        "Evidence": Evidence,
        "verify_claim": verify_claim,
        "_retrieved_documents": retrieved_documents,
    }


RESPONSE_PROTOCOL = """Response protocol — three modes:

MODE 1 — PYTHON: Return executable Python code inside a markdown code fence.
The harness executes the code in a persistent REPL. Use this to call SEC helpers,
inspect data, compute values, and build up state across steps.

MODE 2 — DRAFT:: Propose a preliminary answer for verification. The harness
runs consistency checks and returns the results. The model then decides whether
to refine the answer (more PYTHON steps) or finalize it with FINAL:.

MODE 3 — FINAL:: Return the completed, verified answer. The harness checks
it before accepting: every cited accession number and material number
(dollar amounts, comma-grouped values, multi-decimal ratios) must be
traceable to observations retrieved in this session. Untraceable claims are
rejected with feedback; after 3 rejections the research stops without an
answer. Ground every figure before finalizing.

When structured data is requested, present the table, CSV, or other
requested format first. End substantive research answers with
a Research Summary and References.

Structure:
  FINAL:
  [Title / company name]
  [Brief context]

  [CSV TABLE or structured data if requested]

  RESEARCH SUMMARY
  [Natural-language explanation of findings. Interpret results,
   clarify important distinctions, identify limitations.
   Support factual claims with APA-style author-date citations.]

  REFERENCES
  [APA 7-style references for sources actually used.
   Include direct source URLs where available.]

Examples:
  ```python
  cik = get_cik("Apple")
  facts = get_company_facts(cik)
  print(facts["entityName"])
  ```

  DRAFT:
  Apple Inc.
  Net-income-to-debt ratio for fiscal year 2025

  ratio = 1.24

  RESEARCH SUMMARY
  Apple's net-income-to-debt ratio for fiscal year 2025 was 1.24,
  based on net income of $102.0 billion and total debt of $82.5 billion
  (Apple Inc., 2025).

  REFERENCES
  Apple Inc. (2025). Form 10-K for the fiscal year ended September 28, 2025.
  U.S. Securities and Exchange Commission. https://www.sec.gov/...

  [After verification feedback, if correct:]
  FINAL:
  Apple Inc.
  Net-income-to-debt ratio for fiscal year 2025
  ...

Citation rules:
  - Cite the original source supporting each factual claim.
  - Use SEC filings as primary sources for SEC financial research.
  - Use the actual filing metadata and primaryDocument filename.
  - Do not fabricate authors, publication dates, titles, or URLs.
  - Do not cite a source that was not retrieved or inspected.
  - Distinguish retrieved facts from calculations and inferences.
  - If a claim cannot be substantiated, acknowledge the limitation.
  - When multiple sources share the same author and year, disambiguate
    with lowercase suffixes (2026a, 2026b, 2026c) in both the text
    citations and the reference list. The suffix order follows the
    chronological order of filing dates.

Evidence handling:
  Never replace missing observations with guessed or inferred values.
  A derived value must identify its inputs and assumptions.
  If a requested field is unavailable, use N/A and explain why.
  If evidence contradicts an earlier calculation, correct before FINAL:.
  Do not present inferred values as retrieved SEC observations.
  Before returning FINAL:, review your conclusions against the
  observations already obtained. Verify that every factual claim
  in the RESEARCH SUMMARY is supported by an observation in the
  REPL state. If a claim references a filing, confirm that you
  actually retrieved that filing's metadata.

Verification pass:
  When you receive DRAFT feedback, use Python to check your claims:
  - Compare dates, accession numbers, and values against original data.
  - Run check_period(start, end) for fiscal-period consistency.
  - Use verify_claim() to compare conclusions against retrieved records.
  - Resolve any contradictions before returning FINAL:.

Rules:
  - Return ONLY a code fence with Python code, a `DRAFT:` line, or a `FINAL:` line.
  - Do not include commentary, XML, or prose outside the code fence, DRAFT: line, or FINAL: line.
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

Standard library only: Use Python's built-in modules for CSV generation.
Do not import pandas, numpy, or other optional third-party packages.
If an import fails, do not retry it without changing the environment.

Source attribution:
  get_recent_filings() provides filing metadata, including accessionNumber,
  filingDate, reportDate, and primaryDocument.
  filing_url(cik, accession, primary_document) produces a direct SEC filing URL
  suitable for references.
  get_filing_text() and search_filing() can provide documentary evidence for
  claims beyond filing metadata.
  Company Facts observations identify reporting periods and their associated
  filings, but comparative observations are not necessarily separate filings.
  Use the retrieved company name, filing form, filing date, reporting-period end,
  and URL to construct APA references.
  Do not infer a fiscal-period start solely from reportDate.
  A citation should support the specific claim being made.

Evidence provenance (optional helpers):
  Evidence(claim, value, source, method) — frozen dataclass for tracking
  whether a finding was retrieved from SEC, derived from other values,
  inferred from assumptions, or unsupported. The method field distinguishes
  these categories: "retrieved" | "derived" | "inferred" | "unsupported".
  Example:
    rec = Evidence(
        claim="Q2 2026 reporting-period end",
        value="2026-06-30",
        source=filing_url(cik, accn, doc),
        method="retrieved",
    )
  verify_claim(claim_dict, source_list, key_field) — compares a dict of
  claimed values against original retrieved records. Returns verification
  results per key: {"match": bool, "expected": <claimed>, "actual": <retrieved>}.
  Use this before FINAL: to catch contradictions between your conclusions
  and the actual retrieved data.
  Never silently promote an inference into a retrieved observation.
"""


MAX_OUTPUT_CHARS = 2000


GROUNDING_GUIDANCE = """Evidence-grounded research:
Use retrieved source data as the basis for factual claims.
Distinguish four kinds of information:
- Retrieved: explicitly present in an original source.
- Derived: calculated from identified source values.
- Inferred: reasoned from evidence but not directly stated.
- Unsupported: not established by available evidence.
When a requested value is missing:
- Investigate an appropriate primary source when useful.
- Do not invent values to complete a table or calculation.
- If the value remains unavailable, report it as unknown.
For derived values, preserve the inputs and calculation.
For inferred values, identify the assumption or uncertainty.
An Evidence object records provenance but does not itself prove a claim.
verify_claim() can compare claims against original retrieved records.
Do not verify a claim against another model-generated claim.
Before returning FINAL:
- Compare important conclusions with retrieved observations.
- Resolve contradictions when evidence permits.
- Check that citations support the corresponding claims.
- Identify material limitations and unresolved uncertainty.
The model decides whether further research, verification, or finalization is appropriate.
No specific research workflow is required.
"""


def build_prompt(question: str, history: list, last_feedback: str = "", remaining: int | None = None, draft: str = "") -> str:
    tail = f"{last_feedback}\n" if last_feedback else ""
    recent = " ".join(history[-3:]) if history else "(none)"
    draft_block = f"\nDRAFT to verify:\n{draft}\n" if draft else ""
    lines = [(f"{RESPONSE_PROTOCOL}\n\n{TOOL_REFERENCE}\n\n{GROUNDING_GUIDANCE}\n\nQuestion:\n{question}\n\n{draft_block}"
              f"Recent steps and feedback:\n{tail}{recent}")]
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
        if stripped.startswith(('File "', '  File "')):
            continue
        sig_lines.append(stripped)
    return '\n'.join(sig_lines)


def verify_draft(draft: str, repl: REPL) -> str:
    """Run consistency checks on a draft answer and return verification feedback."""
    issues = []

    # Check for date inconsistencies in CSV-like rows
    date_rows = re.findall(r'(\d{4}-\d{2}-\d{2})\s*→\s*(\d{4}-\d{2}-\d{2})', draft)
    for start, end in date_rows:
        try:
            from datetime import date
            if date.fromisoformat(start) > date.fromisoformat(end):
                issues.append(f"Impossible date range: {start} → {end} (start after end)")
        except ValueError:
            pass

    # Check for obvious fabrication indicators
    suspicious_patterns = [
        (r'\b202[0-9]-(0[13578]|1[02])-(0[13579])\b', 'End-of-month date not tied to a filing'),
        (r'\bfiscal period start is \d{4}-\d{2}-\d{2}', 'Invented fiscal-period start'),
    ]
    for pattern, desc in suspicious_patterns:
        if re.search(pattern, draft):
            issues.append(f"Suspicious pattern: {desc}")

    # Check that any cited accession numbers match retrieved data
    accn_pattern = r'(\d{10}-\d{2}-\d{6})'
    cited_accns = set(re.findall(accn_pattern, draft))
    if cited_accns and 'filings' in repl.namespace:
        retrieved_accns = {f['accessionNumber'] for f in repl.namespace['filings']}
        for accn in cited_accns:
            if accn not in retrieved_accns:
                issues.append(f"Cited accession {accn} not found in retrieved filings")

    if issues:
        return (
            "VERIFICATION ISSUES:\n" + "\n".join(f"  - {i}" for i in issues) + "\n"
            "Fix these issues or run Python to verify claims. "
            "Then return FINAL: with corrected evidence."
        )
    return (
        "VERIFICATION PASSED: No obvious consistency issues found.\n"
        "Run additional checks if needed (verify_claim, check_period, etc.).\n"
        "Return FINAL: when confident the answer is evidence-grounded."
    )


FINAL_ACCESSION_RE = re.compile(r'\d{10}-\d{2}-\d{6}')
FINAL_MONEY_RE = re.compile(r'(-?)\s*\$\s?\(?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?')
FINAL_NUM_RE = re.compile(r'(?<![\d,$.])(-?)(\d{1,3}(?:,\d{3})+)(?:\.(\d+))?')
FINAL_DEC_RE = re.compile(r'(?<![\d$])(-?)(\d+)\.(\d{2,})(?!\d)')
OBS_NUM_RE = re.compile(r'(?<![\d.,])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\d.,])')


def _state_numbers(repl: REPL, cap: int = 2000) -> list[float]:
    """All finite numbers persisted in the REPL namespace (bounded)."""
    from math import isfinite
    values: list[float] = []

    def walk(obj, depth: int = 0) -> None:
        if depth > 8 or len(values) >= cap:
            return
        if isinstance(obj, bool) or callable(obj):
            return
        if isinstance(obj, (int, float)) and isfinite(obj):
            values.append(float(obj))
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)

    for value in repl.namespace.values():
        walk(value)
    return values


def _state_accessions(repl: REPL) -> set[str]:
    """All accession numbers present anywhere in retrieved REPL state."""
    found: set[str] = set()

    def walk(obj, depth: int = 0) -> None:
        if depth > 10 or callable(obj):
            return
        if isinstance(obj, str):
            found.update(FINAL_ACCESSION_RE.findall(obj))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str):
                    found.update(FINAL_ACCESSION_RE.findall(key))
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)

    for value in repl.namespace.values():
        walk(value)
    return found


def _derived_values(state_values: list[float], cap: int = 300, limit: int = 4000) -> list[float]:
    """Arithmetic combinations of state values, in two levels.

    Level 1: sums, differences, products, and ratios of state values.
    Level 2: ratios of level-1 results (e.g. net income / (current +
    noncurrent debt)). Bounded to keep the gate fast.
    """
    from math import isfinite
    base = state_values[:cap]

    def combine(pool: list[float], with_ratios: bool, bound: int) -> list[float]:
        out: list[float] = []
        for i, a in enumerate(pool):
            for b in pool[i:]:
                for candidate in (a + b, a - b, b - a, a * b):
                    if isfinite(candidate):
                        out.append(candidate)
                if with_ratios and b != 0 and isfinite(a / b):
                    out.append(a / b)
                if len(out) >= bound:
                    return out
        return out

    level1 = combine(base, with_ratios=True, bound=limit)
    level2 = combine(level1[:200], with_ratios=True, bound=limit)
    return level1 + level2


def _answer_numbers(answer: str) -> list[tuple[float, str]]:
    """Material numbers in an answer: dollar amounts, comma-grouped, and multi-decimal values."""
    found: list[tuple[float, str]] = []
    seen_spans: list[tuple[int, int]] = []

    def add(match: re.Match, negative_paren: bool = False) -> None:
        if any(start <= match.start() < end for start, end in seen_spans):
            return
        text = match.group(2).replace(",", "")
        fraction = match.group(3)
        value = float(text + ("." + fraction if fraction else ""))
        sign = -1.0 if (match.group(1) == "-" or negative_paren) else 1.0
        seen_spans.append(match.span())
        found.append((sign * value, match.group(0)))

    for match in FINAL_MONEY_RE.finditer(answer):
        add(match, negative_paren="(" in match.group(0))
    for match in FINAL_NUM_RE.finditer(answer):
        add(match)
    for match in FINAL_DEC_RE.finditer(answer):
        add(match)
    return found


def _is_grounded(value: float, state_values: list[float], derived: list[float]) -> bool:
    """A number is grounded when it matches state evidence under unit scaling,
    sign convention (losses as positives), or a simple arithmetic combination."""
    from math import isclose
    for candidate in (value, -value):
        for pool in (state_values, derived):
            for state_value in pool:
                for mult in (1.0, 1e3, 1e6, 1e9, 1e-2):
                    if isclose(candidate * mult, state_value, rel_tol=1e-3, abs_tol=1.0):
                        return True
    return False


def _observation_numbers(text: str) -> list[float]:
    """All numbers in observation output (accession patterns excluded)."""
    cleaned = FINAL_ACCESSION_RE.sub(" ", text)
    values: list[float] = []
    for match in OBS_NUM_RE.finditer(cleaned):
        try:
            values.append(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    return values


def _history_observations(history: list[str]) -> str:
    """Concatenated OUTPUT sections of the session history."""
    parts: list[str] = []
    for entry in history:
        index = entry.find("OUTPUT:\n")
        if index == -1:
            continue
        rest = entry[index + len("OUTPUT:\n"):]
        stop = rest.find("ERROR:\n")
        parts.append(rest if stop == -1 else rest[:stop])
    return "\n".join(parts)


def verify_final(answer: str, repl: REPL, observation_text: str = "") -> list[str]:
    """Deterministic gate for FINAL: answers.

    Every cited accession number and material number must be traceable to
    evidence retrieved in this session — persisted REPL state or printed
    observation output — directly, under unit scaling, or as a simple
    arithmetic combination. Returns a list of issues; an empty list means
    the answer is released.
    """
    issues: list[str] = []
    state_accns = {a.replace("-", "") for a in _state_accessions(repl)}
    state_accns |= {a.replace("-", "") for a in FINAL_ACCESSION_RE.findall(observation_text)}
    cited_hyphen = set(FINAL_ACCESSION_RE.findall(answer))
    for accn in sorted(cited_hyphen):
        if accn.replace("-", "") not in state_accns:
            issues.append(f"Cited accession {accn} was not retrieved in this session.")
    state_values = _state_numbers(repl) + _observation_numbers(observation_text)
    derived = _derived_values(state_values)
    for value, rendered in _answer_numbers(answer):
        if abs(value) < 1000 and "." not in rendered:
            continue  # Small integers (counts, fiscal years) are not material.
        if not _is_grounded(value, state_values, derived):
            issues.append(f"Number {rendered!r} cannot be traced to retrieved observations or a calculation from them.")
    return issues


def check_period(start: str, end: str) -> bool:
    """Check that a fiscal-period start is not after its end date."""
    from datetime import date
    if not start or start in (None, "N/A", "unknown"):
        return True
    return date.fromisoformat(start) <= date.fromisoformat(end)


def research(question: str, max_steps: int = MAX_STEPS) -> str | None:
    """Iterative SEC research: model decides what to do, harness executes."""
    repl = REPL(namespace=sec_tools())
    history: list[str] = []
    invalid_count = 0
    last_error_sig: str | None = None
    repeated_errors = 0
    draft = ""
    final_rejections = 0

    while len(history) < max_steps:
        remaining = max_steps - len(history)
        prompt = build_prompt(question, history, remaining=remaining, draft=draft)
        try:
            response = llm_query(prompt)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"=== INFERENCE UNAVAILABLE === {type(exc).__name__}: {exc}")
            return None
        if is_final_response(response):
            answer = response.strip()
            issues = verify_final(answer, repl, _history_observations(history))
            if issues:
                final_rejections += 1
                feedback = (
                    "FINAL REJECTED: the following claims cannot be traced to retrieved "
                    "evidence in this session:\n"
                    + "\n".join(f"  - {issue}" for issue in issues)
                    + "\nCorrect them from the retrieved observations, or FINAL: an answer "
                    "whose every cited accession and material number is grounded."
                )
                history.append(feedback)
                print(feedback, flush=True)
                if final_rejections >= 3:
                    print("Research stopped: final answer failed verification 3 times.")
                    return None
                continue
            print(answer)
            return answer
        if is_draft_response(response):
            draft = extract_draft(response)
            feedback = verify_draft(draft, repl)
            history.append(feedback)
            print(f"DRAFT: {draft}", flush=True)
            print(feedback, flush=True)
            draft = ""  # Clear draft so it's not echoed again
            continue
        try:
            code = extract_code(response)
            invalid_count = 0
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


def main() -> None:
    """CLI entry point: secedgar "Your SEC research question"."""
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: secedgar "Your SEC research question"')
    research(question)


if __name__ == "__main__":
    main()
