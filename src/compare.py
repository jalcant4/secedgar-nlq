"""Cross-company comparisons from accession-matched 10-K observations.

Each row is built from a single company's most recent 10-K. Missing
observations are reported as errors, never substituted with zero or
inferred values.
"""
from accounting import DEBT_STRATEGIES
from edgar import get_cik, get_company_facts
from sec_data import get_recent_filings, select_observation


def _annual_observations(facts: dict, concept: str) -> list[dict]:
    return [
        obs
        for obs in facts.get("facts", {}).get("us-gaap", {}).get(concept, {})
        .get("units", {}).get("USD", [])
        if obs.get("form") == "10-K"
    ]


def _debt_total(facts: dict, period: str, accession: str) -> tuple[str, int]:
    """Resolve the first supported debt strategy with complete observations."""
    for method, concepts in DEBT_STRATEGIES.items():
        total = 0
        try:
            for concept in concepts:
                obs = select_observation(
                    _annual_observations(facts, concept),
                    period=period, accession=accession)
                if obs["val"] < 0:
                    raise ValueError(f"Negative debt balance for {concept}")
                total += obs["val"]
        except ValueError as exc:
            # A negative balance is a data-quality error and fails closed;
            # only missing/conflicting observations fall through to the next strategy.
            if "Negative debt" in str(exc):
                raise
            continue
        return method, total
    raise ValueError("No supported debt strategy has complete 10-K observations")


def ratio_row_from_data(name: str, cik: str, filing: dict, facts: dict) -> dict:
    """Compute one comparison row from already-retrieved data (pure)."""
    period = filing["reportDate"]
    accession = filing["accessionNumber"]
    net_income = select_observation(
        _annual_observations(facts, "NetIncomeLoss"),
        period=period, accession=accession, duration=True)["val"]
    method, total_debt = _debt_total(facts, period, accession)
    return {
        "company": name,
        "cik": cik,
        "period": period,
        "accession": accession,
        "net_income": net_income,
        "total_debt": total_debt,
        "ratio": net_income / total_debt if total_debt else None,
        "debt_method": method,
        "status": "ok",
    }


def company_debt_ratio_row(name: str) -> dict:
    """Retrieve the latest 10-K for a company and compute its ratio row."""
    cik = get_cik(name)
    filings = get_recent_filings(cik, "10-K")
    if not filings:
        raise ValueError(f"No 10-K filings found for {name} (CIK {cik})")
    facts = get_company_facts(cik)
    return ratio_row_from_data(name, cik, filings[0], facts)


def compare_debt_ratios(names: list[str]) -> list[dict]:
    """Net-income-to-debt ratio for each company's latest 10-K.

    Returns one row per company; retrieval or observation failures are
    reported in that row's status/error fields instead of aborting the
    whole comparison.
    """
    rows = []
    for name in names:
        try:
            rows.append(company_debt_ratio_row(name))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            rows.append({"company": name, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows
