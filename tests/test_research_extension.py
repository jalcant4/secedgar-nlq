import pytest

from filing_text import filing_url, search_filing
from secedgar_agent import format_research_answer, make_contract, validate_research


def test_contract():
    assert make_contract('Investigate whether Oracle financial conditions precipitated the force majeure').kind == 'financial_research'
    assert make_contract('What is net income divided by debt ratio?').kind == 'financial_ratio'

def test_url():
    assert filing_url('0001341439','0001193125-26-277521','orcl-20260531.htm').endswith('/1341439/000119312526277521/orcl-20260531.htm')
    with pytest.raises(ValueError):
        filing_url('1341439','0001193125-26-277521','../passwd')

def test_search():
    hits = search_filing('The data center has lease commitments. No force majeure was reported.', ['lease', 'force majeure'])
    assert len(hits) == 2
    assert all('excerpt' in h for h in hits)

def sample():
    return {'company':'Oracle Corporation','cik':'0001341439','question':'Why?', 'findings':[],
            'financial_indicators':[], 'direct_evidence':[], 'limitations':['Not in filings'],
            'conclusion':'insufficient_evidence', 'sources':[{'accession':'0001193125-26-277521',
            'url':filing_url('0001341439','0001193125-26-277521','orcl-20260531.htm'),
            'form':'10-K','filing_date':'2026-06-22'}]}

def test_research():
    result=sample()
    assert validate_research(result)==[]
    assert 'References' in format_research_answer(result)
    assert '(Oracle Corporation, 2026)' in format_research_answer(result)
    assert 'U.S. Securities and Exchange Commission.' in format_research_answer(result)
    result['conclusion']='documented_causation'
    assert validate_research(result)
    result['conclusion']='insufficient_evidence'
    result['findings']=[{'statement':'Claim','accession':'bad','excerpt':'evidence'}]
    assert validate_research(result)
