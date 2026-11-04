"""SEC EDGAR API client for CIK resolution, filings, and XBRL company facts."""

import os
import tomllib
import time
from pathlib import Path

import requests


def _contact_email() -> str:
    """SEC contact from $SEC_CONTACT_EMAIL or src/config.toml [user].email."""
    env = os.environ.get("SEC_CONTACT_EMAIL")
    if env:
        return env
    try:
        with Path(__file__).with_name("config.toml").open("rb") as f:
            return tomllib.load(f)["user"]["email"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "set [user].email in src/config.toml"


# SEC User-Agent: application name and contact email as required by SEC policy
SEC_USER_AGENT = f"secedgar-nlq (contact: {_contact_email()})"

# SEC public API endpoints
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FILINGS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL_TEMPLATE = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
)

# Conservative rate-limiting delay between SEC API calls (seconds)
_RATE_LIMIT_DELAY = 0.35

# Request timeout (seconds)
_TIMEOUT = 30

# Module-level cache for company tickers to avoid repeated requests
_ticker_cache: dict | None = None


def _session() -> requests.Session:
    """Return a configured requests.Session with SEC User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json",
    })
    return session


def _fetch_json(url: str) -> dict:
    """Fetch and parse a JSON endpoint with timeout and error handling."""
    session = _session()
    response = session.get(url, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _delay() -> None:
    """Enforce a conservative delay between SEC API requests."""
    time.sleep(_RATE_LIMIT_DELAY)


def get_cik(company: str) -> str:
    """Resolve a company name or ticker to its SEC CIK.

    Args:
        company: Company name (case-insensitive partial match) or ticker symbol.

    Returns:
        The 10-digit CIK string (zero-padded).

    Raises:
        ValueError: If no match or multiple ambiguous matches are found.
        ConnectionError: If the SEC ticker endpoint is unreachable.
    """
    global _ticker_cache

    if _ticker_cache is None:
        data = _fetch_json(COMPANY_TICKERS_URL)
        # The SEC returns a flat dict keyed by index, e.g. {"0": {...}, "1": {...}}
        # Store full items so we can match on both ticker and company name/title.
        _ticker_cache = list(data.values())
        _delay()

    query = company.strip()
    if not query:
        raise ValueError("Company name or ticker must not be empty.")

    query_upper = query.upper()

    # Exact ticker match (case-insensitive)
    ticker_matches = [
        item for item in _ticker_cache if item["ticker"] == query_upper
    ]
    if len(ticker_matches) == 1:
        return str(ticker_matches[0]["cik_str"]).zfill(10)
    if len(ticker_matches) > 1:
        # Prefer plain tickers over share-class tickers (e.g. ORCL over ORCL-PD)
        plain = [m for m in ticker_matches if "-" not in m["ticker"]]
        if len(plain) == 1:
            return str(plain[0]["cik_str"]).zfill(10)
        tickers = ", ".join(item["ticker"] for item in ticker_matches)
        raise ValueError(
            f"Ambiguous ticker for '{company}': "
            f"matched {len(ticker_matches)} tickers. "
            f"Candidates: {tickers}"
        )

    # Partial name/title match (case-insensitive)
    # Expand common corporate suffix abbreviations so "Corporation" matches "CORP"
    import re as _re
    expanded = query
    expanded = _re.sub(r"\bCORPORATION\b", "CORP", expanded, flags=_re.IGNORECASE)
    expanded = _re.sub(r"\bINCORPORATED\b", "INC", expanded, flags=_re.IGNORECASE)
    expanded = _re.sub(r"\bLIMITED\b", "LTD", expanded, flags=_re.IGNORECASE)
    expanded = _re.sub(r"\bCOMPANY\b", "CO", expanded, flags=_re.IGNORECASE)
    expanded = _re.sub(r"\bENTERPRISES\b", "ENT", expanded, flags=_re.IGNORECASE)

    name_matches = [
        item for item in _ticker_cache
        if expanded.lower() in item["title"].lower()
    ]
    if len(name_matches) == 1:
        return str(name_matches[0]["cik_str"]).zfill(10)
    if len(name_matches) > 1:
        # Deduplicate by CIK so same company with multiple tickers isn't "ambiguous"
        by_cik: dict = {}
        for m in name_matches:
            by_cik.setdefault(m["cik_str"], m)
        if len(by_cik) == 1:
            return str(next(iter(by_cik.values()))["cik_str"]).zfill(10)
        names = ", ".join(item["title"] for item in name_matches)
        raise ValueError(
            f"Ambiguous match for '{company}': "
            f"found {len(name_matches)} companies. "
            f"Refine your query. Candidates: {names}"
        )

    raise ValueError(
        f"Could not identify CIK for '{company}': "
        f"no matching company or ticker found."
    )


def get_filings(cik: str) -> dict:
    """Retrieve company filing metadata from the SEC.

    Args:
        cik: The 10-digit CIK string (will be zero-padded if necessary).

    Returns:
        Raw filing metadata dictionary from the SEC filings endpoint.

    Raises:
        ValueError: If the CIK is invalid.
        ConnectionError: If the SEC filings endpoint is unreachable.
    """
    cik = cik.strip().zfill(10)
    if not cik.isdigit() or len(cik) != 10:
        raise ValueError(f"Invalid CIK: '{cik}'. Must be a 10-digit numeric string.")

    url = FILINGS_URL_TEMPLATE.format(cik=int(cik))
    _delay()
    return _fetch_json(url)


def get_company_facts(cik: str) -> dict:
    """Retrieve structured XBRL financial data for a company.

    Args:
        cik: The 10-digit CIK string (will be zero-padded if necessary).

    Returns:
        Raw company facts dictionary from the SEC XBRL endpoint.

    Raises:
        ValueError: If the CIK is invalid.
        ConnectionError: If the SEC company facts endpoint is unreachable.
    """
    cik = cik.strip().zfill(10)
    if not cik.isdigit() or len(cik) != 10:
        raise ValueError(f"Invalid CIK: '{cik}'. Must be a 10-digit numeric string.")

    url = COMPANY_FACTS_URL_TEMPLATE.format(cik=int(cik))
    _delay()
    return _fetch_json(url)
