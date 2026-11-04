"""Company-independent, conservative accounting rules for SEC US-GAAP facts.

These rules establish internally consistent debt representations, not proof that
all financing obligations are disclosed in the Company Facts API.
"""
import re
from dataclasses import dataclass
from math import isclose, isfinite

from filing_text import search_filing
from sec_data import normalize_concept

# Accepted representations are alternatives, never additive to one another.
# The two-component strategy is accepted only when both parts reconcile.
DEBT_STRATEGIES = {
    "current_plus_noncurrent": ("LongTermDebtCurrent", "LongTermDebtNoncurrent"),
    "reported_long_term_debt": ("LongTermDebt",),
    "reported_debt_carrying_amount": ("DebtInstrumentCarryingAmount",),
}

# These concepts cannot be added to an outstanding debt balance.
EXCLUDED_CONCEPT_FRAGMENTS = (
    "interestexpense", "payments", "proceeds", "maturities", "repayments",
    "gain", "loss", "unamortized", "availableforsale", "conversion",
)


@dataclass(frozen=True)
class DebtAssessment:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _finite_number(value):
    return type(value) in (int, float) and isfinite(value)


def _observed(facts, concept, value, period, sources):
    for obs in facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", []):
        if (obs.get("form") == "10-K" and obs.get("end") == period
                and obs.get("accn") in sources and _finite_number(obs.get("val"))
                and isclose(obs["val"], value, rel_tol=1e-9, abs_tol=1)):
            return True
    return False


def validate_debt_composition(result: dict, facts: dict) -> DebtAssessment:
    """Validate debt strategy, provenance, reconciliation, and obvious exclusions.

    A valid strategy is only *internally consistent*. It cannot prove coverage
    of company extensions, other borrowings, or debt footnote disclosures.
    """
    errors, warnings = [], []
    method = result.get("debt_method")
    components = result.get("debt_components")
    period = result.get("period")
    sources = result.get("sources")
    total = result.get("total_debt")
    if method not in DEBT_STRATEGIES:
        return DebtAssessment((f"debt_method must be one of {list(DEBT_STRATEGIES)}.",), ())
    if not isinstance(components, list):
        return DebtAssessment(("debt_components must be a list.",), ())
    if not isinstance(period, str) or not isinstance(sources, list):
        return DebtAssessment(("Missing period or SEC accession sources.",), ())
    expected = DEBT_STRATEGIES[method]
    names = []
    for component in components:
        if not isinstance(component, dict):
            continue
        try:
            names.append(normalize_concept(component.get("concept")))
        except ValueError as exc:
            errors.append(f"Invalid debt concept identifier: {exc}.")
            names.append(None)
    if len(names) != len(components) or len(names) != len(set(names)):
        errors.append("Debt components must be unique concept/value dictionaries.")
    if set(names) != set(expected) or len(names) != len(expected):
        errors.append(f"Method {method} requires exactly {list(expected)}; do not add other debt concepts.")
    subtotal = 0
    for component in components:
        if not isinstance(component, dict):
            continue
        raw_name, value = component.get("concept"), component.get("value")
        try:
            name = normalize_concept(raw_name)
        except ValueError:
            continue  # Identifier error already reported above.
        if not _finite_number(value):
            errors.append("Each debt component needs a concept string and finite number.")
            continue
        if any(fragment in name.casefold() for fragment in EXCLUDED_CONCEPT_FRAGMENTS):
            errors.append(f"{name} is not an outstanding debt balance.")
        if value < 0:
            errors.append(f"{name} has a negative debt balance.")
        if not _observed(facts, name, value, period, set(sources)):
            errors.append(f"{name} has no matching 10-K observation for period {period}, source accession and submitted value.")
        subtotal += value
    if not _finite_number(total) or not isclose(subtotal, total, rel_tol=1e-9, abs_tol=1):
        errors.append("total_debt must equal the permitted component sum.")

    gaap = facts.get("facts", {}).get("us-gaap", {})
    if method == "current_plus_noncurrent" and "LongTermDebt" in gaap:
        aggregate = [o for o in gaap["LongTermDebt"].get("units", {}).get("USD", [])
                     if o.get("form") == "10-K" and o.get("end") == period
                     and o.get("accn") in sources and _finite_number(o.get("val"))]
        if aggregate and not any(isclose(o["val"], total, rel_tol=1e-9, abs_tol=1) for o in aggregate):
            errors.append("Current + noncurrent debt does not reconcile to reported LongTermDebt.")
    if method == "reported_long_term_debt" and all(c in gaap for c in ("LongTermDebtCurrent", "LongTermDebtNoncurrent")):
        warnings.append("Check that reported LongTermDebt reconciles to current and noncurrent balances.")
    warnings.append("Company Facts alone cannot establish exhaustive interest-bearing debt coverage; inspect the debt footnote and company-specific extensions before calling this fully verified.")
    return DebtAssessment(tuple(errors), tuple(warnings))

# Discovery metadata describes *candidate* balance concepts, not a guarantee
# that a given company's full debt is covered by these XBRL tags.
METRIC_CONCEPT_ROLES = {
    "net_income_loss": {"NetIncomeLoss": "net_income_attributable_to_parent"},
    "total_interest_bearing_debt": {
        "LongTermDebt": "aggregate",
        "DebtInstrumentCarryingAmount": "aggregate_candidate",
        "LongTermDebtCurrent": "current",
        "LongTermDebtNoncurrent": "noncurrent",
    },
}

DEBT_CONCEPT_ROLES = {
    "LongTermDebt": "aggregate",
    "DebtInstrumentCarryingAmount": "aggregate_candidate",
    "LongTermDebtCurrent": "current",
    "LongTermDebtNoncurrent": "noncurrent",
}


def find_concepts(cik: str, metric: str = "total_interest_bearing_debt") -> list[dict]:
    """Find available ontology-tagged concepts with compact evidence summaries."""
    if metric not in METRIC_CONCEPT_ROLES:
        raise ValueError(f"Unsupported ontology metric: {metric}; supported: {', '.join(METRIC_CONCEPT_ROLES)}")
    from edgar import get_company_facts
    gaap = get_company_facts(cik).get("facts", {}).get("us-gaap", {})
    matches = []
    for name, role in METRIC_CONCEPT_ROLES[metric].items():
        if name not in gaap:
            continue
        observations = gaap[name].get("units", {}).get("USD", [])
        annual = [o for o in observations if o.get("form") == "10-K" and o.get("end")]
        periods = sorted({o["end"] for o in annual}, reverse=True)
        matches.append({"concept": name, "role": role, "periods": periods[:5],
                        "label": gaap[name].get("label", "")})
    return matches


def build_debt_composition(cik: str, *, period: str, accession: str,
                           method: str = 'current_plus_noncurrent') -> dict:
    """Resolve a declared strategy to exact US-GAAP tags and observed values.

    Does not claim to establish exhaustive interest-bearing debt coverage.
    """
    from sec_data import get_annual_observations, select_observation
    if method not in DEBT_STRATEGIES:
        raise ValueError(f'Unsupported debt strategy {method!r}; choose from {tuple(DEBT_STRATEGIES)}')
    components = []
    for concept in DEBT_STRATEGIES[method]:
        observation = select_observation(
            get_annual_observations(cik, concept), period=period, accession=accession)
        if observation['val'] < 0:
            raise ValueError(f'Negative debt balance for {concept}')
        components.append({'concept': concept, 'value': observation['val']})
    return {'debt_method': method, 'debt_components': components,
            'total_debt': sum(item['value'] for item in components)}


# Debt footnote retrieval and conservative review.
# Never upgrades a ratio to verified by keyword alone.

DEBT_TERMS = ('long-term debt', 'debt obligations', 'credit agreement',
              'senior secured', 'convertible notes', 'finance lease',
              'other borrowings', 'debt instruments', 'equipment financing',
              'senior notes', 'operating lease liabilities', 'debt maturities')

def review_debt_footnote(cik, accession, *, filings=None, fetch=None, max_excerpts=6):
    """Fetch the cited 10-K and return relevant verbatim excerpts for human/model review.

    A positive retrieval is NOT evidence of exhaustive debt coverage. Company-specific
    XBRL extensions and footnote tables may require separate inspection.
    """
    if filings is None:
        from sec_data import get_recent_filings
        filings = get_recent_filings(cik, '10-K')
    if fetch is None:
        from filing_text import get_filing_text
        fetch = get_filing_text
    filing = next((f for f in filings if f.get('accessionNumber') == accession and
                   f.get('form') == '10-K' and f.get('primaryDocument')), None)
    if filing is None:
        return {'status':'not_reviewed', 'reason':'Matching 10-K primary document unavailable', 'excerpts':[], 'source':None}
    try:
        doc = fetch(cik, accession, filing['primaryDocument'])
        if not isinstance(doc, dict) or not doc.get('text') or not doc.get('url'):
            raise ValueError('Empty filing text')
        hits = search_filing(doc['text'], DEBT_TERMS, window=550, max_hits=100)
        seen, excerpts = set(), []
        for hit in hits:
            # Deduplicate near-identical passages; prioritize different debt terms.
            key = (hit['term'].lower(), re.sub(r'\s+', ' ', hit['excerpt'][:160]).lower())
            if key in seen:
                continue
            seen.add(key)
            excerpts.append({'term':hit['term'], 'excerpt':hit['excerpt'], 'offset':hit['offset']})
            if len(excerpts) >= max_excerpts:
                break
        inventory = {
            'borrowings': [t for t in ('credit agreement', 'convertible notes', 'equipment financing', 'senior notes', 'other borrowings') if t in doc['text'].lower()],
            'finance_leases_discussed': 'finance lease' in doc['text'].lower(),
            'operating_leases_discussed': 'operating lease' in doc['text'].lower(),
            'reconciled_to_xbrl': False,
        }
        return {'inventory': inventory, 'status':'review_required', 'reason':(
            'Retrieved debt-related footnote excerpts; reconcile all borrowings, '
            'company-specific XBRL extensions, and lease classification before '
            'asserting exhaustive interest-bearing debt.'),
            'excerpts':excerpts, 'source':{'accession':accession,'url':doc['url'],
            'form':'10-K','filing_date':filing.get('filingDate','')}}
    except (OSError, ValueError, KeyError, TimeoutError) as exc:
        return {'status':'not_reviewed', 'reason':f'{type(exc).__name__}: {exc}', 'excerpts':[], 'source':None}
