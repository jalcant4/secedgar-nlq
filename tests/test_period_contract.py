"""Period-selection regression tests using mocked SEC data (no network)."""
import secedgar_agent as agent

CIK = '0001769628'
ACC = '0001769628-26-000104'

def filing(period, accession):
    return {'reportDate': period, 'filingDate': '2026-03-02', 'accessionNumber': accession, 'form': '10-K'}

def candidate(period, accession=ACC):
    return {'cik': CIK, 'period': period, 'sources': [accession]}

def test_latest_2024_rejected_even_when_2024_is_valid_comparative(monkeypatch):
    monkeypatch.setattr(agent, 'get_recent_filings', lambda cik, form=None: [filing('2025-12-31', ACC), filing('2024-12-31', '0001769628-25-000001')])
    contract = agent.make_contract('latest annual net income to debt ratio')
    assert contract.latest_annual
    assert 'Wrong reporting period' in agent.validate_reporting_period(candidate('2024-12-31'), contract)[0]
    assert agent.validate_reporting_period(candidate('2025-12-31'), contract) == []

def test_wrong_accession_rejected(monkeypatch):
    monkeypatch.setattr(agent, 'get_recent_filings', lambda cik, form=None: [filing('2025-12-31', ACC)])
    errors = agent.validate_reporting_period(candidate('2025-12-31', '0001769628-25-000001'), agent.make_contract('latest earnings-to-debt ratio'))
    assert 'must cite its 10-K accession' in errors[0]

def test_historical_request_does_not_enforce_latest(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Should not fetch filings for historical request')
    monkeypatch.setattr(agent, 'get_recent_filings', forbidden)
    contract = agent.make_contract('2024 net income divided by total debt ratio')
    assert not contract.latest_annual
    assert agent.validate_reporting_period(candidate('2024-12-31'), contract) == []

def test_no_filings_fails_closed(monkeypatch):
    monkeypatch.setattr(agent, 'get_recent_filings', lambda cik, form=None: [])
    assert 'No dated 10-K' in agent.validate_reporting_period(candidate('2024-12-31'), agent.make_contract('latest ratio'))[0]
