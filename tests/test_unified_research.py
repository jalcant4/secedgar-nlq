import pytest

from accounting import review_debt_footnote
from filing_text import prepare_evidence, validate_synthesis

ACC='0001769628-26-000104'
F=[{'form':'10-K','accessionNumber':ACC,'primaryDocument':'cw.htm','filingDate':'2026-03-02'}]
URL='https://www.sec.gov/Archives/edgar/data/1769628/000176962826000104/cw.htm'
def fetch(cik, accn, primary):
    return {'url':URL,'text':'The long-term debt includes senior secured borrowings. Finance lease commitments are also described.'}

def test_footnote_is_review_not_verified():
    r=review_debt_footnote('0001769628',ACC,filings=F,fetch=fetch)
    assert r['status']=='review_required'
    assert r['source']['accession']==ACC
    assert r['excerpts']
    assert 'exhaustive' in r['reason']

def test_missing_footnote_is_not_reviewed():
    assert review_debt_footnote('1',ACC,filings=[],fetch=fetch)['status']=='not_reviewed'

def test_retrieval_error_is_not_verified():
    def bad(*args): raise OSError('offline')
    r=review_debt_footnote('1',ACC,filings=F,fetch=bad)
    assert r['status']=='not_reviewed'

def test_typed_trend_rows():
    evidence={'findings':[{'accession':ACC,'term':'lease','excerpt':'The company disclosed commitments.'}], 'sources':[], 'failures':[]}
    trends={'rows':[{'period':'2025-12-31','metrics':{'net_income':{'value':-100}}}], 'changes':[]}
    r=prepare_evidence(evidence,trends)
    assert r['trends']['rows'][0]['metrics']['net_income']['value']==-100
    assert len(r['snippets'])==1

def test_bad_trend_shape_rejected():
    with pytest.raises(TypeError): prepare_evidence({},['2025-12-31'])

def test_raw_excerpt_rejected():
    r={'findings':[{'statement':'SEC filing contains reference to data center.'}], 'financial_indicators':[], 'direct_evidence':[], 'conclusion':'insufficient_evidence'}
    assert validate_synthesis(r)

def test_causality_inconsistency_rejected():
    r={'findings':[{'statement':'Liquidity constraints caused the event.'}], 'financial_indicators':[], 'direct_evidence':[], 'conclusion':'insufficient_evidence'}
    assert validate_synthesis(r)
