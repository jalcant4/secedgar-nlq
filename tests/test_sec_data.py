import sys
import types

import pytest

try:
    import edgar
except ImportError:
    edgar = types.ModuleType("edgar")
    edgar.get_company_facts = lambda cik: {}
    edgar.get_filings = lambda cik: {}
    sys.modules["edgar"] = edgar
import accounting
import sec_data

ACC = '0001769628-26-000104'

def test_recent_filings_parallel_arrays(monkeypatch):
    payload = {'filings': {'recent': {'accessionNumber': ['a', ACC], 'form': ['10-Q', '10-K'],
                                      'filingDate': ['2026-08-12', '2026-03-02'],
                                      'reportDate': ['2026-06-30', '2025-12-31']}}}
    monkeypatch.setattr(sec_data, 'get_filings', lambda cik: payload)
    rows = sec_data.get_recent_filings('0001769628', '10-K')
    assert len(rows) == 1 and rows[0]['accessionNumber'] == ACC

def test_recent_filings_malformed(monkeypatch):
    monkeypatch.setattr(sec_data, 'get_filings', lambda cik: {'filings': {'recent': {
        'accessionNumber': ['a', 'b'], 'filingDate': ['date'], 'reportDate': ['date'], 'form': ['10-K']}}})
    with pytest.raises(ValueError):
        sec_data.get_recent_filings('x')

def test_observations_and_ontology(monkeypatch):
    payload = {'facts': {'us-gaap': {
        'NetIncomeLoss': {'units': {'USD': [{'val': -1167, 'form': '10-K', 'end': '2025-12-31'}]}},
        'LongTermDebtCurrent': {'label': 'Current debt', 'units': {'USD': [{'val': 6708, 'form': '10-K', 'end': '2025-12-31'}]}},
        'InterestExpenseDebt': {'units': {'USD': [{'val': 1148, 'form': '10-K'}]}}}}}
    monkeypatch.setattr(sec_data, 'get_company_facts', lambda cik: payload)
    monkeypatch.setattr('edgar.get_company_facts', lambda cik: payload)
    assert sec_data.get_annual_observations('x', 'NetIncomeLoss')[0]['val'] == -1167
    found = accounting.find_concepts('x')
    assert [x['concept'] for x in found] == ['LongTermDebtCurrent']
    assert found[0]['role'] == 'current'

def test_qualified_observation_names(monkeypatch):
    payload = {'facts': {'us-gaap': {'NetIncomeLoss': {'units': {'USD': [{'val': -100, 'form': '10-K'}]}}}}}
    monkeypatch.setattr(sec_data, 'get_company_facts', lambda cik: payload)
    assert sec_data.get_observations('x', 'us-gaap:NetIncomeLoss') == sec_data.get_observations('x', 'NetIncomeLoss')
    with pytest.raises(ValueError, match='namespace'):
        sec_data.get_observations('x', 'dei:NetIncomeLoss')


def test_net_income_discovery(monkeypatch):
    payload = {'facts': {'us-gaap': {'NetIncomeLoss': {'label': 'Net income', 'units': {'USD': [{'val': -100, 'form': '10-K', 'end': '2025-12-31'}]}}}}}
    monkeypatch.setattr('edgar.get_company_facts', lambda cik: payload)
    matches = accounting.find_concepts('x', 'net_income_loss')
    assert len(matches) == 1 and matches[0]['concept'] == 'NetIncomeLoss'
