"""Small, deterministic adapters for the SEC EDGAR JSON shapes.

Uses the existing edgar.py transport functions; no direct HTTP access here.
"""
from edgar import get_company_facts, get_filings


def normalize_concept(concept: str, taxonomy: str = "us-gaap") -> str:
    """Canonicalize a local US-GAAP tag; reject other taxonomies explicitly."""
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("Concept must be a nonempty string")
    concept = concept.strip()
    if ":" in concept:
        namespace, concept = concept.split(":", 1)
        if namespace != taxonomy or not concept or ":" in concept:
            raise ValueError(f"Concept namespace must be {taxonomy!r}; received {namespace!r}")
    return concept


def get_recent_filings(cik: str, form: str | None = None) -> list[dict]:
    recent = get_filings(cik)["filings"]["recent"]
    if not isinstance(recent, dict):
        raise TypeError("SEC recent filings must be a dictionary of parallel arrays")
    required = ("accessionNumber", "filingDate", "reportDate", "form")
    if any(not isinstance(recent.get(k), list) for k in required):
        raise ValueError("SEC recent filings missing required arrays")
    lengths = {len(v) for v in recent.values() if isinstance(v, list)}
    if len(lengths) != 1 or any(not isinstance(v, list) for v in recent.values()):
        raise ValueError("SEC recent filings arrays have inconsistent lengths")
    count = lengths.pop()
    rows = [{key: values[i] for key, values in recent.items()} for i in range(count)]
    if form is not None:
        rows = [row for row in rows if row["form"] == form]
    return sorted(rows, key=lambda row: (row.get("reportDate") or "", row.get("filingDate") or ""), reverse=True)


def get_observations(cik: str, concept: str, unit: str = "USD", taxonomy: str = "us-gaap") -> list[dict]:
    concept = normalize_concept(concept, taxonomy)
    facts = get_company_facts(cik)
    return list(facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, []))


def get_annual_observations(cik: str, concept: str, unit: str = "USD") -> list[dict]:
    """Return 10-K observations; caller must still distinguish annual flows from instants."""
    return [obs for obs in get_observations(cik, concept, unit) if obs.get("form") == "10-K"]


def select_observation(observations: list[dict], *, period: str, accession: str,
                       duration: bool = False) -> dict:
    """Select one SEC observation from the requested filing and period.

    Duration facts must have a start date and a roughly annual duration; instant
    balance-sheet facts must not have a start date. Conflicting values fail closed.
    """
    from datetime import date
    matches = []
    for obs in observations:
        if (obs.get('end') != period or obs.get('accn') != accession or
                obs.get('form') != '10-K' or
                type(obs.get('val')) not in (int, float)):
            continue
        from math import isfinite
        if not isfinite(obs['val']):
            continue
        start = obs.get('start')
        if duration:
            if not isinstance(start, str):
                continue
            try:
                days = (date.fromisoformat(period) - date.fromisoformat(start)).days + 1
            except ValueError:
                continue
            if not 350 <= days <= 380:
                continue
        elif start is not None:
            continue
        matches.append(obs)
    if not matches:
        raise ValueError(f'Missing matching {"annual duration" if duration else "instant"} 10-K observation for {period}, accession {accession}')
    values = {obs['val'] for obs in matches}
    if len(values) != 1:
        raise ValueError(f'Conflicting observations for {period}, accession {accession}: {sorted(values)}')
    return matches[0]
