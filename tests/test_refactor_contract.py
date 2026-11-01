from secedgar_agent import make_trend_indicators, validate_research

A = '0001193125-26-277521'
S = {'accession': A, 'form': '10-K', 'filing_date': '2026-06-22',
     'url': 'https://www.sec.gov/Archives/edgar/data/1341439/000119312526277521/orcl.htm'}

def test_xbrl_indicator_generated_and_accepted():
    trends = {'rows': [{'accession': A, 'period': '2026-05-31',
                        'metrics': {'capital_expenditures': {'value': 55_663_000_000}}}]}
    indicators = make_trend_indicators(trends, {'sources': [S]})
    assert len(indicators) == 1
    assert indicators[0]['value'] == 55_663_000_000
    result = {'question': 'Investigate Oracle capital spending', 'conclusion': 'insufficient_evidence',
              'sources': [S], 'findings': [], 'direct_evidence': [],
              'financial_indicators': indicators, 'limitations': []}
    assert validate_research(result) == []

def test_unlisted_accession_not_generated():
    trends = {'rows': [{'accession': A, 'period': '2026-05-31',
                        'metrics': {'capital_expenditures': {'value': 10}}}]}
    assert make_trend_indicators(trends, {'sources': []}) == []

def test_fake_xbrl_number_rejected():
    result = {'question': 'Investigate Oracle', 'conclusion': 'insufficient_evidence',
              'sources': [S], 'findings': [], 'direct_evidence': [], 'limitations': [],
              'financial_indicators': [{'statement': 'The company reported a financial change during fiscal 2026.',
                                        'accession': A, 'evidence_type': 'xbrl',
                                        'metric': 'capital_expenditures', 'period': '2026-05-31',
                                        'value': 'not a number'}]}
    assert any('numeric XBRL' in x for x in validate_research(result))

def test_debt_review_inventory_is_not_reconciliation():
    from accounting import review_debt_footnote
    filings = [{'accessionNumber': A, 'form': '10-K', 'primaryDocument': 'orcl.htm'}]
    fetch = lambda *args: {'url': S['url'], 'text': 'The company has convertible notes and operating lease liabilities.'}
    review = review_debt_footnote('0001341439', A, filings=filings, fetch=fetch)
    assert review['inventory']['reconciled_to_xbrl'] is False
    assert review['inventory']['operating_leases_discussed'] is True
    assert review['status'] == 'review_required'
