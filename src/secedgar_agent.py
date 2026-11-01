"""Iterative SEC research with frozen, harness-owned completion contracts."""

import ast
import json
import re
import sys
from dataclasses import dataclass
from math import isclose, isfinite

from accounting import (
    DEBT_STRATEGIES,
    build_debt_composition,
    find_concepts,
    review_debt_footnote,
    validate_debt_composition,
)
from edgar import get_cik, get_company_facts, get_filings
from filing_text import (
    filing_url,
    get_filing_text,
    prepare_evidence,
    research_filings,
    search_filing,
    validate_synthesis,
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

MAX_STEPS = 15
RESEARCH_STEPS = 30
MAX_OBSERVATION_CHARS = 2000
MAX_IDENTICAL_REJECTIONS = 3

# These are explicit lexical discovery rules, NOT accounting classifications.
CONCEPT_TERMS = {
    "net_income_concepts": ("netincome", "netloss", "incomeloss"),
    "debt_concepts": ("debt", "notespayable", "borrowings"),
}


@dataclass(frozen=True)
class CompletionContract:
    kind: str
    requirements: tuple[str, ...]
    scope: str
    latest_annual: bool = False


def make_contract(question: str) -> CompletionContract:
    """Select only task types with implemented deterministic checks.

    Ambiguous questions are unsupported instead of silently misclassified.
    """
    q = question.casefold()
    ratio = any(term in q for term in ("ratio", "divided by", "divide by"))
    discovery = (any(term in q for term in ("concept", "xbrl"))
                 and any(term in q for term in ("income", "earnings"))
                 and "debt" in q)
    research_terms = ('investigate', 'force majeure', 'precipitat', 'causal',
                      'contributed to', 'financial conditions', 'project jupiter')
    if any(term in q for term in research_terms):
        return CompletionContract('financial_research',
            ('company', 'cik', 'question', 'findings', 'financial_indicators',
             'direct_evidence', 'limitations', 'conclusion', 'sources'),
            'filing_text_and_financial_observations')
    if ratio:
        return CompletionContract(
            "financial_ratio",
            ("company", "cik", "period", "net_income", "total_debt",
             "ratio", "sources", "debt_components", "debt_method"),
            "annual_10k_observations",
            latest_annual=any(term in q for term in ("latest", "most recent", "newest")),
        )
    if discovery:
        return CompletionContract(
            "concept_discovery",
            ("company", "cik", "net_income_concepts", "debt_concepts"),
            "all_matching_us_gaap_concepts",
        )
    return CompletionContract("unsupported", (), "no_automatic_verifier")


def extract_code(response: str) -> str:
    match = re.search(
        r'<parameter\s+name=["\']code["\']\s*>\s*(.*?)\s*</parameter>',
        response, re.DOTALL,
    )
    if not match:
        match = re.search(r"```(?:python|py|json)?\s*\n(.*?)```", response, re.DOTALL)
    code = (match.group(1) if match else response).strip()
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


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def expected_concepts(facts: dict) -> dict[str, list[str]]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    return {
        field: sorted(name for name in gaap
                      if any(term in name.casefold() for term in terms))
        for field, terms in CONCEPT_TERMS.items()
    }


def validate_structure(result: object, contract: CompletionContract) -> list[str]:
    if not isinstance(result, dict):
        return ["Assign a dictionary named result containing the final answer."]
    return [f"Missing result[{key!r}]." for key in contract.requirements
            if result.get(key) is None]


def validate_coverage(result: dict, contract: CompletionContract, facts: dict) -> list[str]:
    if contract.kind != "concept_discovery":
        return []
    errors = []
    expected = expected_concepts(facts)
    for field, required in expected.items():
        actual = result.get(field)
        if not isinstance(actual, list) or any(not isinstance(c, str) for c in actual):
            errors.append(f"{field} must be a list of concept-name strings.")
            continue
        missing = sorted(set(required) - set(actual))
        extra = sorted(set(actual) - set(required))
        if len(actual) != len(set(actual)):
            errors.append(f"{field} contains duplicate concepts.")
        if missing:
            errors.append(f"{field} is missing {len(missing)} matching concepts: {missing}.")
        if extra:
            errors.append(f"{field} contains {len(extra)} out-of-scope concepts: {extra}.")
    return errors


def validate_reporting_period(result: dict, contract: CompletionContract) -> list[str]:
    """Enforce latest 10-K fiscal year when the question explicitly requires it.

    Filing selection is harness-owned; the model cannot waive this requirement.
    """
    if not contract.latest_annual:
        return []
    filings = get_recent_filings(result["cik"], form="10-K")
    eligible = [f for f in filings if f.get("reportDate") and f.get("accessionNumber")]
    if not eligible:
        return ["No dated 10-K filings found; cannot verify latest annual period."]
    latest_period = max(f["reportDate"] for f in eligible)
    if result["period"] != latest_period:
        return [(f"Wrong reporting period: latest 10-K fiscal year ends {latest_period}; "
                f"submitted {result['period']}. Use the latest fiscal year, not a comparative year.")]
    latest_accessions = {f["accessionNumber"] for f in eligible
                         if f["reportDate"] == latest_period}
    if not isinstance(result["sources"], list) or not latest_accessions.intersection(result["sources"]):
        return [(f"Latest period {latest_period} must cite its 10-K accession "
                f"({', '.join(sorted(latest_accessions))}).")]
    return []


def validate_ratio(result: dict, facts: dict) -> list[str]:
    """Verify arithmetic and provenance, not semantic correctness of debt grouping."""
    errors = []
    for key in ("net_income", "total_debt", "ratio"):
        if not _number(result[key]):
            errors.append(f"{key} must be a finite number.")
    if errors:
        return errors
    if result["total_debt"] <= 0:
        errors.append("Total debt must be positive.")
    elif not isclose(result["ratio"], result["net_income"] / result["total_debt"],
                     rel_tol=1e-9, abs_tol=1e-12):
        errors.append("Ratio does not equal net_income / total_debt.")

    period = result["period"]
    if not isinstance(period, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
        return errors + ["Period must be YYYY-MM-DD."]
    sources = result["sources"]
    if (not isinstance(sources, list) or not sources or
        any(not isinstance(s, str) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", s)
            for s in sources)):
        return errors + ["Sources must be a nonempty list of SEC accession numbers."]
    components = result["debt_components"]
    if not isinstance(components, list) or not components:
        return errors + ["debt_components must be a nonempty list."]

    gaap = facts.get("facts", {}).get("us-gaap", {})
    source_ids = set(sources)

    def matches(concept, value, annual):
        from datetime import date as _date
        for obs in gaap.get(concept, {}).get("units", {}).get("USD", []):
            if (obs.get("form") != "10-K" or obs.get("end") != period or
                obs.get("accn") not in source_ids or not _number(obs.get("val"))):
                continue
            if annual:
                start = obs.get("start")
                if not isinstance(start, str):
                    continue
                try:
                    days = (_date.fromisoformat(period) - _date.fromisoformat(start)).days + 1
                except ValueError:
                    continue
                if not 350 <= days <= 380:
                    continue
            if isclose(obs["val"], value, rel_tol=1e-9, abs_tol=1):
                return True
        return False

    if not matches("NetIncomeLoss", result["net_income"], True):
        errors.append("Net income is not supported by a matching annual 10-K NetIncomeLoss observation.")
    return errors


def make_trend_indicators(trends: dict, evidence: dict, *, limit: int = 8) -> list[dict]:
    """Construct numeric, accession-backed findings from annual XBRL data.

    No prose model or filing-text quotation is needed to establish the numeric
    observation. The result validator separately verifies each against REPL trends.
    """
    if not isinstance(trends, dict) or not isinstance(trends.get('rows'), list):
        raise TypeError("Expected financial_trends dictionary with rows")
    source_accessions = {s.get('accession') for s in evidence.get('sources', []) if isinstance(s, dict)}
    indicators = []
    for row in trends['rows']:
        if not isinstance(row, dict) or row.get('accession') not in source_accessions:
            continue
        for metric, item in row.get('metrics', {}).items():
            if not isinstance(item, dict) or not _number(item.get('value')):
                continue
            value = item['value']
            indicators.append({
                'statement': f"The company reported {metric.replace('_', ' ')} of ${value / 1e9:,.3f} billion for the period ended {row['period']}; this documents financial exposure but does not establish event causation.",
                'metric': metric, 'value': value, 'period': row['period'],
                'accession': row['accession'], 'evidence_type': 'xbrl',
            })
            if len(indicators) >= limit:
                return indicators
    return indicators


def validate_research(result: dict) -> list[str]:
    """Validate source-backed substantive claims; interpretation remains model-owned."""
    errors = []
    if not isinstance(result.get('question'), str) or not result['question'].strip():
        errors.append('question must describe the investigation.')
    for field in ('findings', 'financial_indicators', 'direct_evidence', 'limitations'):
        if not isinstance(result.get(field), list):
            errors.append(f'{field} must be a list.')
    if result.get('conclusion') not in ('documented_causation', 'circumstantial_evidence', 'insufficient_evidence'):
        errors.append('Invalid conclusion classification.')
    sources = result.get('sources')
    if not isinstance(sources, list) or not sources:
        return errors + ['sources must be a nonempty list of SEC filing references.']
    valid_sources = set()
    for i, source in enumerate(sources):
        if not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k] for k in ('accession','url','form','filing_date')):
            errors.append(f'sources[{i}] requires accession, url, form, filing_date.')
            continue
        if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', source['accession']):
            errors.append(f'sources[{i}] invalid accession.')
        if not source['url'].startswith('https://www.sec.gov/Archives/edgar/data/'):
            errors.append(f'sources[{i}] invalid SEC document URL.')
        valid_sources.add(source['accession'])
    generic = re.compile(r'^(?:filing|document|report)\s+(?:mentions|references|contains|discusses|includes)\b', re.IGNORECASE)
    for field in ('findings', 'financial_indicators', 'direct_evidence'):
        for i, item in enumerate(result.get(field, []) if isinstance(result.get(field), list) else []):
            if not isinstance(item, dict):
                errors.append(f'{field}[{i}] must be a dictionary.')
                continue
            statement = item.get('statement')
            if not isinstance(statement, str) or len(statement.split()) < 8 or generic.search(statement):
                errors.append(f'{field}[{i}] must state a substantive disclosure or analytical observation, not a keyword hit.')
            if item.get('accession') not in valid_sources:
                errors.append(f'{field}[{i}] must cite a listed accession.')
            if item.get('evidence_type') == 'xbrl' and field == 'financial_indicators':
                if not isinstance(item.get('metric'), str) or not _number(item.get('value')) or not isinstance(item.get('period'), str):
                    errors.append(f'{field}[{i}] requires numeric XBRL value, metric and period.')
            elif not isinstance(item.get('excerpt'), str) or not item['excerpt'].strip():
                errors.append(f'{field}[{i}] requires a supporting verbatim excerpt.')
    if result.get('conclusion') == 'documented_causation' and not result.get('direct_evidence'):
        errors.append('Documented causation requires direct evidence.')
    errors.extend(validate_synthesis(result))
    return errors


def date_from_iso(value: str) -> str:
    """APA date: year, full month name, day."""
    from datetime import datetime
    return datetime.strptime(value, '%Y-%m-%d').strftime('%Y, %B %-d')


def format_research_answer(result: dict) -> str:
    """Concise paragraph, cited claims, and APA references for *used* sources only."""
    by_accession = {s['accession']:s for s in result['sources']}
    items = [i for field in ('findings','financial_indicators','direct_evidence')
             for i in result.get(field, []) if isinstance(i,dict) and i.get('accession') in by_accession]
    # Only include the most informative claims, not every search hit.
    unique = []
    seen = set()
    for item in items:
        key = (item['accession'], item.get('statement','').casefold().strip())
        if key not in seen:
            seen.add(key)
            unique.append(item)
        if len(unique) >= 5:
            break
    used = {i['accession'] for i in unique}
    if not used:  # Cite the actual reviewed filings for the bounded negative conclusion.
        used = {result['sources'][0]['accession']}
    ordered = sorted((by_accession[a] for a in used), key=lambda s:(s['filing_date'],s['form'],s['accession']))
    # APA 7: distinguish same-author, same-year references with a/b/c suffixes.
    from collections import Counter, defaultdict
    counts = Counter(s['filing_date'][:4] for s in ordered)
    index = defaultdict(int)
    labels = {}
    for source in ordered:
        year = source['filing_date'][:4]
        suffix = chr(97 + index[year]) if counts[year] > 1 else ''
        index[year] += 1
        labels[source['accession']] = year + suffix
    author = result['company']
    def cite(acc):
        return f"({author}, {labels[acc]})"
    conclusions = {
        'documented_causation':'The reviewed filings contain evidence explicitly linking financial conditions to the reported event.',
        'circumstantial_evidence':'The reviewed filings indicate potential financial pressure, but do not establish that it caused the reported event.',
        'insufficient_evidence':'The reviewed filings do not establish that financial conditions precipitated the reported event.',
    }
    sentences = []
    for item in unique:
        sentences.append(item['statement'].strip().rstrip('.') + ' ' + cite(item['accession']) + '.')
    summary_cites = '; '.join(f'{author}, {labels[s["accession"]]}' for s in ordered)
    sentences.append(conclusions[result['conclusion']].rstrip('.') + f' ({summary_cites}).')
    if result.get('limitations'):
        sentences.append('This conclusion is limited to the filings examined; absence of a mention does not establish that no private notice exists.')
    _author = author.rstrip('.')
    references = [f"{_author}. ({date_from_iso(s['filing_date'])}). "
                  f"*Form {s['form']}* [SEC filing]. U.S. Securities and Exchange Commission. {s['url']}"
                  for s in ordered]
    return ' '.join(sentences) + '\n\nReferences\n' + '\n'.join(references)


def _fmt_dollars(value: float) -> str:
    """Format a dollar amount as a human-readable string."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f} billion"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f} million"
    return f"${value:,.2f}"


def format_ratio_answer(result: dict, retrieved_documents: dict | None = None) -> str:
    """Human-readable APA-style paragraph for a financial_ratio result."""
    company = result.get('company', 'Unknown')
    period = result.get('period', 'unknown')
    net_income = result.get('net_income', 0)
    total_debt = result.get('total_debt', 0)
    ratio = result.get('ratio', 0)
    sources = result.get('sources', [])
    debt_method = result.get('debt_method', 'unknown')
    debt_components = result.get('debt_components', [])
    debt_review = result.get('debt_review', {})

    # Build source metadata lookup from retrieved documents or debt review.
    # retrieved_documents may have URL but no filing_date; debt_review fills that gap.
    source_meta: dict[str, dict] = {}
    if retrieved_documents:
        for acc, doc in retrieved_documents.items():
            if isinstance(doc, dict) and doc.get('url'):
                source_meta[acc] = {'url': doc['url'], 'filing_date': '', 'form': '10-K'}
    if isinstance(debt_review, dict) and debt_review.get('source'):
        src = debt_review['source']
        acc = src.get('accession', '')
        if acc:
            source_meta[acc] = {
                'url': src.get('url', source_meta.get(acc, {}).get('url', '')),
                'filing_date': src.get('filing_date', source_meta.get(acc, {}).get('filing_date', '')),
                'form': src.get('form', source_meta.get(acc, {}).get('form', '10-K')),
            }

    # APA labels: same-year disambiguation.
    from collections import Counter, defaultdict
    ordered = sorted(
        (source_meta.get(acc, {'url': '', 'filing_date': '', 'form': '10-K'})
         for acc in sources if acc),
        key=lambda s: (s['filing_date'], s['form'], next((a for a in sources if source_meta.get(a, {}) == s), '')),
    )
    # Deduplicate by accession.
    seen_accs: set[str] = set()
    unique_sources = []
    for acc in sources:
        if acc in seen_accs:
            continue
        seen_accs.add(acc)
        meta = source_meta.get(acc, {'url': '', 'filing_date': '', 'form': '10-K'})
        unique_sources.append((acc, meta))
    ordered = [m for _, m in unique_sources]

    counts = Counter(s['filing_date'][:4] for s in ordered if s.get('filing_date'))
    index: defaultdict[str, int] = defaultdict(int)
    labels: dict[str, str] = {}
    for acc, meta in unique_sources:
        year = meta.get('filing_date', '')[:4]
        if year:
            suffix = chr(97 + index[year]) if counts[year] > 1 else ''
            index[year] += 1
            labels[acc] = f"{company}, {year}{suffix}"
        else:
            labels[acc] = company

    # Build narrative sentences.
    sentences = []
    cite_str = ', '.join(labels.get(acc, company) for acc, _ in unique_sources)
    sentences.append(
        f"For the fiscal year ending {period}, {company} reported net income of "
        f"{_fmt_dollars(net_income)} and total interest-bearing debt of "
        f"{_fmt_dollars(total_debt)} ({debt_method.replace('_', ' ')}), yielding a "
        f"net-income-to-debt ratio of {ratio:.4f} {cite_str}."
    )

    # Debt components detail.
    if debt_components:
        comp_parts = []
        for c in debt_components:
            concept = c.get('concept', 'unknown')
            value = c.get('value', 0)
            comp_parts.append(f"{_fmt_dollars(value)} in {concept}")
        sentences.append(
            f"Total debt comprises {'; '.join(comp_parts[:-1])}, and "
            f"{comp_parts[-1]}."
        )

    # Debt review findings.
    if isinstance(debt_review, dict):
        status = debt_review.get('status', '')
        reason = debt_review.get('reason', '')
        if status == 'review_required' and reason:
            sentences.append(f"Debt footnote review: {reason}")
        elif status == 'not_reviewed' and reason:
            sentences.append(f"Debt footnote review could not be completed: {reason}")

    sentences.append(
        "This ratio is derived from XBRL Company Facts and does not independently "
        "verify exhaustive interest-bearing debt coverage; lease liabilities and "
        "company-specific extensions may affect the total."
    )

    # References.
    references = []
    for acc, meta in unique_sources:
        year = meta.get('filing_date', '')[:4]
        form = meta.get('form', '10-K')
        url = meta.get('url', '')
        _company = company.rstrip('.')
        ref = f"{_company}. ({year}). " if year else f"{_company}. (n.d.). "
        ref += f"*Form {form}* [SEC filing]. U.S. Securities and Exchange Commission. {url}"
        references.append(ref)

    return ' '.join(sentences) + '\n\nReferences\n' + '\n'.join(references)


# Backward-compatible name for callers of the previous extension.
def format_research_sentence(result: dict) -> str:
    return format_research_answer(result)


def research_complete(result: object, contract: CompletionContract,
                      facts: dict | None) -> list[str]:
    """Empty errors means all *implemented* checks passed."""
    if contract.kind == "unsupported":
        return ["Unsupported question type: no deterministic completion validator exists."]
    errors = validate_structure(result, contract)
    if errors:
        return errors
    if facts is None:
        return ["SEC company facts are unavailable for verification."]
    if not isinstance(result["cik"], str) or result["cik"] != str(facts.get("cik", "")).zfill(10):
        errors.append("CIK does not match SEC company facts.")
    if (not isinstance(result["company"], str) or
        result["company"].casefold() != str(facts.get("entityName", "")).casefold()):
        errors.append("Company name does not match SEC company facts.")
    if contract.kind == "concept_discovery":
        errors.extend(validate_coverage(result, contract, facts))
    elif contract.kind == "financial_research":
        errors.extend(validate_research(result))
    elif contract.kind == "financial_ratio":
        errors.extend(validate_reporting_period(result, contract))
        errors.extend(validate_ratio(result, facts))
        assessment = validate_debt_composition(result, facts)
        errors.extend(assessment.errors)
    return errors


def research(question: str, max_steps: int = MAX_STEPS) -> dict | None:
    contract = make_contract(question)  # Freeze before the model is called.
    if contract.kind == "unsupported":
        print("Unsupported for automatic verification: only net-income/debt concept discovery "
              "and financial-ratio questions are currently supported.")
        return None

    if contract.kind == 'financial_research' and max_steps == MAX_STEPS:
        max_steps = RESEARCH_STEPS
    repl = REPL()
    repl.validation = {"status": "not_checked"}
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
    repl.namespace.update({"research_filings": tracked_research_filings,
                           "prepare_evidence": prepare_evidence,
                           "make_trend_indicators": make_trend_indicators,
                           "review_debt_footnote": tracked_debt_review,
                           "financial_trends": financial_trends,
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
                           "filing_url": filing_url})
    if contract.kind == "concept_discovery":
        schema = ("result = {'company': company_name, 'cik': cik, "
                  "'net_income_concepts': complete_list, 'debt_concepts': complete_list}")
        scope = ("Find ALL us-gaap concept names matching these case-insensitive substrings: "
                 "net_income_concepts: netincome, netloss, incomeloss; "
                 "debt_concepts: debt, notespayable, borrowings. "
                 "These are lexical matches, not a classification of debt balances. "
                 "Submit the COMPLETE lists, not selected examples.")
    elif contract.kind == 'financial_research':
        schema = ("result = {'company': company_name, 'cik': cik, 'question': question_text, "
                  "'findings': [{'statement': text, 'accession': accession, 'excerpt': quoted_excerpt}], "
                  "'financial_indicators': [...], 'direct_evidence': [...], "
                  "'limitations': [text, ...], "
                  "'conclusion': 'insufficient_evidence', "
                  "'sources': [{'accession': accession, 'url': sec_filing_url, "
                  "'form': '10-K', 'filing_date': 'YYYY-MM-DD'}, ...]}")
        scope = ("This is qualitative research, NOT a financial ratio task. "
                 "First call evidence = research_filings(cik, forms=['10-K','10-Q','8-K'], "
                 "terms=['Project Jupiter','force majeure','New Mexico','data center','lease commitments','liquidity'], max_filings=30). "
                 "This deterministic helper returns sources, findings with verbatim excerpts, failures, searched and limitations. "
                 "Reuse evidence across steps; do not manually loop over filings or repeat retrieval. "
                 "Do NOT assign evidence['findings'] directly to result['findings']; these are keyword hits, not conclusions. "
                 "Interpret at most five distinct material excerpts into substantive disclosure statements with the original accession and exact verbatim excerpt. "
                 "Call trends = financial_trends(cik, periods=3); its schema is a dictionary with rows (each row has period, accession, metrics dictionary) and changes list, NOT a list of periods. Call prepared = prepare_evidence(evidence, trends) for compact evidence. "
                 "For financial_indicators use make_trend_indicators(trends, evidence) to produce accession-verified quantitative observations directly from XBRL; do not invent filing-text excerpts for XBRL numbers. Each indicator has statement, metric, value, period, accession, evidence_type=xbrl, and comparison provenance. These are documented financial indicators, NOT proof of Project Jupiter causation. "
                 "Use sources=evidence['sources']; financial_indicators and direct_evidence may be empty if unsupported. "
                 "A valid insufficient_evidence conclusion requires no causal proof. "
                 "If no sources were retrieved, report failure rather than invent sources. "
                 "Never treat a search-term match as documented causation; direct_evidence requires an explicit causal statement. "
                 "The ontology supports ONLY net_income_loss and total_interest_bearing_debt for find_concepts; financial_trends is a SEPARATE helper with six financial metrics, including capital_expenditures, operating_cash_flow, free_cash_flow, cash_and_equivalents, debt_carrying_amount and net_income. Never call get_annual_observations with a derived metric name. "
                 "never call find_concepts for capex, lease, cash flow, or other metrics. "
                 "Avoid calling build_debt_composition for research tasks unless specifically needed. "
                 "Do not claim that a source was searched if retrieval failed.")
    else:
        schema = ("result = {'company': company_name, 'cik': cik, 'period': 'YYYY-MM-DD', "
                  "'net_income': number, 'total_debt': number, 'ratio': number, "
                  "'sources': [SEC_accession_number, ...], "
                  "'debt_method': 'current_plus_noncurrent', 'debt_components': [{'concept': us_gaap_name, 'value': number}, ...]}")
        scope = ("Use annual 10-K NetIncomeLoss and year-end debt observations from the same "
                 "period; do not double-count overlapping debt concepts. "
                 "Choose debt_method from: {', '.join(DEBT_STRATEGIES)}. Use ONLY the exact components for that method. Explain your debt definition in printed findings. "
                 "Verification checks observations and arithmetic, not accounting judgment.")

    instructions = f"""You are researching an SEC EDGAR question in a persistent Python REPL.
Use the provided functions: get_cik(company), get_company_facts(cik),
get_recent_filings(cik, form=None), get_observations(cik, concept, unit="USD"),
get_annual_observations(cik, concept), find_concepts(cik, metric="total_interest_bearing_debt"),
find_concepts(cik, metric="net_income_loss").
SUPPORTED_ONTOLOGY_METRICS contains the ONLY supported metric names; do not invent others.
get_filings(cik) is available for advanced exploration but returns SEC parallel arrays.
Concept names may be unprefixed or use the us-gaap: prefix; both are accepted.
Use EXACT concept names from find_concepts, never metric names or guessed aliases as XBRL concepts.
find_concepts(cik, metric) accepts semantic metrics; get_annual_observations(cik, concept)
accepts ONLY exact US-GAAP XBRL concepts, e.g. NetIncomeLoss (not net_income_loss).
Use select_observation(get_annual_observations(cik, "NetIncomeLoss"),
period=period, accession=accession, duration=True) for annual net income.
IMPORTANT: select_observation() returns an observation DICTIONARY, not a number.
Extract the numeric value using ni_observation["val"] before calculating a ratio.
Read the official company name from get_company_facts(cik)["entityName"];
never invent, substitute, or hard-code a company name in the submitted result.
Calculate ratios using full numeric precision: ratio = net_income / total_debt.
Do NOT round ratio or any numeric value in the result dictionary.
Rounding is permitted ONLY in human-readable print output.
For RATIO tasks, use review_debt_footnote(cik, accession) to inspect the 10-K debt footnote; this provides evidence for accounting review but cannot independently certify exhaustive coverage.
For RATIO tasks only, call build_debt_composition(cik, period=period, accession=accession,
method="current_plus_noncurrent") to get total_debt, debt_method, debt_components.
The helper raises on missing components, never substitutes zero. Other available
strategies are in DEBT_STRATEGIES; a strategy is not proof of complete debt coverage.
Use bare names in submitted debt_components for clarity.
Prefer normalized helpers. Do NOT access a nonexistent observations key: SEC XBRL values are under units["USD"].
Do not import SEC libraries or make HTTP requests. Reuse persistent variables.
Company facts: facts['entityName'], facts['cik'], facts['facts']['us-gaap'][concept]['units']['USD'].
Observations contain val, start (for durations), end, form, filed, accn.
Find concepts inside the us-gaap dictionary. Print short observations only (at most 5 items).
Use get_recent_filings(cik, "10-K") to find the latest annual report.
For RESEARCH tasks prefer research_filings(cik, forms, terms, max_filings), which retrieves and searches SEC primary documents deterministically.
The final research answer may be a sentence or paragraph. The harness will append APA-style in-text citations and a References section using verified SEC filing metadata. Provide concise, specific, source-backed statements.
When the question requests latest, you MUST use its reportDate, not an earlier comparative year.
Use get_annual_observations for 10-K facts; match report end dates and filing accessions.
Use find_concepts to discover ontology-tagged debt candidates, then choose ONE compatible debt strategy.
Never sum all concepts containing "debt". Never use (value or 0) for missing financial values: explicitly reject None. Reuse variables from earlier steps.
Completion contract (fixed by harness): {contract.kind}; scope: {contract.scope}.
{scope}
Generate ONE executable Python step, code only; no XML or commentary.
When evidence is sufficient, assign a dictionary named result:
{schema}
Do not set research_complete; only the harness decides when to stop.
"""
    observations: list[str] = []
    # Keep critical failure details visible; the REPL holds complete objects.
    def observation(step: int, feedback: str, output: str) -> str:
        tail = output[-1200:] if "Traceback" in output else output[:900]
        return f"Step {step}: {feedback}\nREPL output: {tail}"
    last_rejection = None
    identical_rejections = 0
    for step in range(1, max_steps + 1):
        prompt = (f"{instructions}\nQuestion:\n{question}\n\nRecent steps and feedback:\n"
                  f"{' '.join(observations[-3:]) or '(none)'}\n"
                  "Continue with one Python step; do not repeat completed work. Inspect persistent state with print(inspect()), print(inspect('latest', mode='keys')), print(inspect(mode='last_error')), or print(inspect(mode='validation')) when needed. Do not reset successful work.")
        try:
            response = llm_query(prompt)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"=== INFERENCE UNAVAILABLE === {type(exc).__name__}: {exc}")
            return None
        try:
            code = extract_code(response)
        except (SyntaxError, ValueError) as exc:
            feedback = f"Invalid Python: {exc}. Return only executable Python."
            print(f"STEP {step}: INVALID CODE — {feedback}", flush=True)
            observations.append(feedback)
            continue
        repl.namespace.pop("result", None)
        output = repl.execute(code)
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        failed = repl.last_failed
        candidate = repl.namespace.get("result")
        if failed:
            err_tail = (repl.last_error or output)[-400:]
            feedback = f"Execution failed: {err_tail}\nReuse variables already set; inspect(inspect(mode=\"last_error\"), inspect(mode=\"validation\"), inspect(\"cik\", mode=\"keys\")) to repair without redoing work."
            repl.validation = {"status": "execution_failed", "errors": [repl.last_error or output]}
        elif candidate is not None:
            cik = candidate.get("cik") if isinstance(candidate, dict) else None
            try:
                facts = get_company_facts(cik) if isinstance(cik, str) else None
                errors = research_complete(candidate, contract, facts)
                if contract.kind == 'financial_research' and isinstance(candidate, dict):
                    for source in candidate.get('sources', []) if isinstance(candidate.get('sources'), list) else []:
                        if isinstance(source, dict):
                            doc = retrieved_documents.get(source.get('accession'))
                            if not doc or doc['url'] != source.get('url'):
                                errors.append('Every cited SEC document must be retrieved using get_filing_text in this run.')
                    trend_rows = repl.namespace.get('trends', {})
                    trend_rows = trend_rows.get('rows', []) if isinstance(trend_rows, dict) else []
                    xbrl_expected = {
                        (r['accession'], r['period'], metric): item['value']
                        for r in trend_rows if isinstance(r, dict)
                        for metric, item in r.get('metrics', {}).items()
                        if isinstance(item, dict) and 'value' in item
                    }
                    for indicator in candidate.get('financial_indicators', []):
                        if isinstance(indicator, dict) and indicator.get('evidence_type') == 'xbrl':
                            key = (indicator.get('accession'), indicator.get('period'), indicator.get('metric'))
                            if key not in xbrl_expected or indicator.get('value') != xbrl_expected[key]:
                                errors.append('XBRL indicator does not match retrieved accession, period and metric.')
                    for field in ('findings', 'financial_indicators', 'direct_evidence'):
                        for finding in candidate.get(field, []) if isinstance(candidate.get(field), list) else []:
                            if isinstance(finding, dict) and finding.get('evidence_type') != 'xbrl' and isinstance(finding.get('excerpt'), str):
                                doc = retrieved_documents.get(finding.get('accession'))
                                if not doc or finding['excerpt'].strip() not in doc['text']:
                                    errors.append(f'{field} excerpt is not verbatim text from its cited filing.')
            except Exception as exc:  # noqa: BLE001
                errors = [f"SEC verification failed: {type(exc).__name__}: {exc}"]
            repl.validation = {"status": "rejected" if errors else "accepted",
                               "errors": list(errors),
                               "result_type": contract.kind}
            if not errors:
                if contract.kind == 'financial_research':
                    print(format_research_answer(candidate))
                elif contract.kind == 'financial_ratio':
                    # Compute debt review first so formatter can cite it.
                    accession = candidate.get('sources', [None])[0]
                    review = repl.namespace.get('debt_review')
                    if not isinstance(review, dict) or not review.get('source') or review['source'].get('accession') != accession:
                        review = tracked_debt_review(candidate['cik'], accession)
                    candidate['debt_review'] = review
                    print(format_ratio_answer(candidate, retrieved_documents))
                else:
                    print(candidate)
                return candidate
            feedback = "Result rejected: " + "; ".join(errors)
            # Repeated identical validation errors indicate no progress, even
            # if the model changes code or resubmits equivalent values.
            fingerprint = tuple(sorted(errors))
            if fingerprint == last_rejection:
                identical_rejections += 1
            else:
                last_rejection = fingerprint
                identical_rejections = 1
            if identical_rejections >= MAX_IDENTICAL_REJECTIONS:
                print(f"STALLED: {identical_rejections} identical rejections. {feedback}", flush=True)
                return None
        else:
            feedback = "Continue research; no final result submitted yet."
            repl.validation = {"status": "not_submitted"}
        observations.append(observation(step, feedback, output))
        if feedback and feedback != "Continue research; no final result submitted yet.":
            print(f"STEP {step}: VALIDATION — {feedback}", flush=True)
    print(f"Research stopped after {max_steps} steps without a verified result.")
    return None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: python secedgar_agent.py "Your SEC research question"')
    research(question)
