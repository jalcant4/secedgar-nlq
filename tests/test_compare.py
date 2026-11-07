"""Offline tests for cross-company comparison rows (pure data path)."""
from compare import ratio_row_from_data

CIK = "1769628"
FILING = {"accessionNumber": "0001769628-26-000104",
          "reportDate": "2025-12-31", "filingDate": "2026-03-02", "form": "10-K"}


def _facts(net_income, debt_current, debt_noncurrent, accession="0001769628-26-000104",
           period="2025-12-31"):
    def instant(concept, val):
        return [{"start": None, "end": period, "val": val, "accn": accession,
                 "form": "10-K", "fy": 2025, "fp": "FY"}]
    def duration(concept, val):
        return [{"start": "2025-01-01", "end": period, "val": val, "accn": accession,
                 "form": "10-K", "fy": 2025, "fp": "FY"}]
    gaap = {}
    if net_income is not None:
        gaap["NetIncomeLoss"] = {"label": "Net Income (Loss)", "units": {"USD": duration("NetIncomeLoss", net_income)}}
    if debt_current is not None:
        gaap["LongTermDebtCurrent"] = {"label": "Current", "units": {"USD": instant("LongTermDebtCurrent", debt_current)}}
    if debt_noncurrent is not None:
        gaap["LongTermDebtNoncurrent"] = {"label": "Noncurrent", "units": {"USD": instant("LongTermDebtNoncurrent", debt_noncurrent)}}
    return {"entityName": "Test Co", "facts": {"us-gaap": gaap}}


def test_ratio_row_ok():
    facts = _facts(net_income=-1_167_000_000, debt_current=6_708_000_000,
                   debt_noncurrent=14_665_000_000)
    row = ratio_row_from_data("Test Co", CIK, FILING, facts)
    assert row["status"] == "ok"
    assert row["net_income"] == -1_167_000_000
    assert row["total_debt"] == 21_373_000_000
    assert row["debt_method"] == "current_plus_noncurrent"
    assert abs(row["ratio"] - (-1_167_000_000 / 21_373_000_000)) < 1e-12
    assert row["accession"] == FILING["accessionNumber"]


def test_ratio_row_missing_net_income():
    facts = _facts(net_income=None, debt_current=6_708_000_000,
                   debt_noncurrent=14_665_000_000)
    try:
        ratio_row_from_data("Test Co", CIK, FILING, facts)
        raise AssertionError("expected ValueError for missing NetIncomeLoss")
    except ValueError:
        pass


def test_debt_falls_back_to_reported_long_term_debt():
    facts = _facts(net_income=100, debt_current=None, debt_noncurrent=None)
    # No current/noncurrent; add an aggregate LongTermDebt observation.
    facts["facts"]["us-gaap"]["LongTermDebt"] = {
        "label": "Long-term Debt",
        "units": {"USD": [{"start": None, "end": "2025-12-31", "val": 500,
                           "accn": "0001769628-26-000104", "form": "10-K"}]},
    }
    row = ratio_row_from_data("Test Co", CIK, FILING, facts)
    assert row["status"] == "ok"
    assert row["debt_method"] == "reported_long_term_debt"
    assert row["total_debt"] == 500
    assert row["ratio"] == 0.2


def test_debt_no_complete_strategy():
    facts = _facts(net_income=100, debt_current=6_708_000_000, debt_noncurrent=None)
    try:
        ratio_row_from_data("Test Co", CIK, FILING, facts)
        raise AssertionError("expected ValueError when no debt strategy is complete")
    except ValueError as exc:
        assert "No supported debt strategy" in str(exc)


def test_negative_debt_rejected():
    facts = _facts(net_income=100, debt_current=-5, debt_noncurrent=10)
    try:
        ratio_row_from_data("Test Co", CIK, FILING, facts)
        raise AssertionError("expected ValueError for negative debt")
    except ValueError as exc:
        assert "Negative debt" in str(exc)
