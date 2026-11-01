from filing_text import filing_url, research_filings
from secedgar_agent import format_research_answer, make_contract, research_complete

CIK='0001341439'
ACC='0001193125-26-277521'
FILINGS=[{'form':'10-K','accessionNumber':ACC,'primaryDocument':'orcl.htm','filingDate':'2026-06-22'},
         {'form':'8-K','accessionNumber':'0001193125-26-277522','primaryDocument':'event.htm','filingDate':'2026-06-23'}]

def fetch(cik, acc, primary):
    return {'url':filing_url(cik,acc,primary),'accession':acc,
            'text':'The company disclosed significant lease commitments.'}

def test_search_and_source_provenance():
    evidence=research_filings(CIK,filings=FILINGS,fetch=fetch,terms=['lease commitments'])
    assert evidence['searched']==2 and evidence['attempted']==2
    assert len(evidence['findings'])==2
    assert all(f['excerpt'] in 'The company disclosed significant lease commitments.' for f in evidence['findings'])

def test_retrieval_failure_not_negative_evidence():
    def broken(cik,acc,primary):
        raise OSError('blocked')
    evidence=research_filings(CIK,filings=FILINGS,fetch=broken)
    assert evidence['searched']==0 and len(evidence['failures'])==2
    assert evidence['limitations']

def test_insufficient_evidence_completes_without_causal_match():
    evidence=research_filings(CIK,filings=FILINGS,fetch=fetch,terms=['Project Jupiter'])
    result={'company':'Oracle Corporation','cik':CIK,'question':'Did financial conditions cause the event?',
            'findings':evidence['findings'],'financial_indicators':[], 'direct_evidence':[],
            'limitations':['No direct causal evidence in the reviewed documents.'],
            'conclusion':'insufficient_evidence','sources':evidence['sources']}
    facts={'cik':CIK,'entityName':'Oracle Corporation'}
    assert research_complete(result,make_contract('Investigate force majeure'),facts)==[]
    assert '(Oracle Corporation, 2026)' in format_research_answer(result)

def test_limit_and_form_filter():
    evidence=research_filings(CIK,filings=FILINGS,fetch=fetch,forms=['8-K'],max_filings=1)
    assert evidence['searched']==1 and evidence['sources'][0]['form']=='8-K'
