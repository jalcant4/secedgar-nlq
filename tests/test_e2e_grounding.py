"""Evaluation helpers for Experiment B: evidence-grounded SEC research.

These tests compare a FINAL: answer against independently verified
filing metadata. They do not control the research loop — they only
evaluate the output after it is produced.
"""
import json
import re
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "coreweave_filings.json"


@pytest.fixture
def coreweave_filings():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def evaluate_filing_row(row: dict, source: dict) -> dict:
    """Compare a single CSV row against the source fixture record."""
    return {
        "accession": row.get("accession_number") == source["accessionNumber"],
        "filing_date": row.get("filing_date") == source["filingDate"],
        "period_end": row.get("fiscal_period_end") == source["reportDate"],
        "form": row.get("filing_form") == source["form"],
    }


def parse_csv_rows(final_answer: str) -> list[dict]:
    """Extract CSV rows from a FINAL: answer."""
    # Match a CSV block that is not preceded by a section header like
    # RESEARCH SUMMARY, REFERENCES, or FINAL.
    csv_match = re.search(
        r"^(?:FINAL:\s*)?(?:CSV TABLE\s*)?"
        r"([\w\-_]+(?:,\s*[\w\-_]+)*)\s*\n"
        r"((?:[^\n]+\n?)*)"
        r"(?=\n\s*(?:RESEARCH SUMMARY|REFERENCES|$))",
        final_answer,
        re.MULTILINE,
    )
    if not csv_match:
        return []
    header = [h.strip() for h in csv_match.group(1).split(",")]
    rows = []
    for line in csv_match.group(2).strip().split("\n"):
        if not line.strip():
            continue
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(header, values)))
    return rows


def test_fixture_has_expected_count(coreweave_filings):
    assert len(coreweave_filings) == 4


def test_fixture_contains_one_10k_and_three_10qs(coreweave_filings):
    forms = [f["form"] for f in coreweave_filings]
    assert forms.count("10-K") == 1
    assert forms.count("10-Q") == 3


def test_fixture_accession_numbers_are_unique(coreweave_filings):
    accessions = [f["accessionNumber"] for f in coreweave_filings]
    assert len(accessions) == len(set(accessions))


def test_fixture_report_dates_are_valid_dates(coreweave_filings):
    from datetime import datetime
    for f in coreweave_filings:
        datetime.strptime(f["reportDate"], "%Y-%m-%d")


def test_fixture_filing_dates_are_valid_dates(coreweave_filings):
    from datetime import datetime
    for f in coreweave_filings:
        datetime.strptime(f["filingDate"], "%Y-%m-%d")


def test_evaluate_filing_row_match(coreweave_filings):
    source = coreweave_filings[0]
    row = {
        "accession_number": source["accessionNumber"],
        "filing_date": source["filingDate"],
        "fiscal_period_end": source["reportDate"],
        "filing_form": source["form"],
    }
    result = evaluate_filing_row(row, source)
    assert all(result.values())


def test_evaluate_filing_row_mismatch(coreweave_filings):
    source = coreweave_filings[0]
    row = {
        "accession_number": "0000000000-00-000000",
        "filing_date": source["filingDate"],
        "fiscal_period_end": source["reportDate"],
        "filing_form": source["form"],
    }
    result = evaluate_filing_row(row, source)
    assert result["accession"] is False
    assert result["filing_date"] is True
    assert result["period_end"] is True
    assert result["form"] is True


def test_parse_csv_rows_basic():
    answer = """FINAL:

CSV TABLE
fiscal_period_start,fiscal_period_end,filing_form,filing_date,accession_number,sec_filing_url
unknown,2025-12-31,10-K,2026-03-02,0001769628-26-000104,https://example.com/1
unknown,2026-06-30,10-Q,2026-08-12,0001769628-26-000366,https://example.com/2

RESEARCH SUMMARY
..."""
    rows = parse_csv_rows(answer)
    assert len(rows) == 2
    assert rows[0]["fiscal_period_end"] == "2025-12-31"
    assert rows[0]["filing_form"] == "10-K"


def test_parse_csv_rows_empty():
    answer = "FINAL:\nNo data retrieved.\n"
    rows = parse_csv_rows(answer)
    assert rows == []


def test_parse_csv_rows_no_table():
    answer = """FINAL:
RESEARCH SUMMARY
CoreWeave had four filings."""
    rows = parse_csv_rows(answer)
    assert rows == []
