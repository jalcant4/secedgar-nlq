from financial_trends import financial_trends
from secedgar_agent import format_research_answer, validate_research

A='0001193125-26-277521'; B='0000950170-25-087926'
S=lambda a,d: {'accession':a,'filing_date':d,'form':'10-K','url':'https://www.sec.gov/Archives/edgar/data/1341439/'+a.replace('-','')+'/orcl.htm'}

def sample():
    return {'company':'Oracle Corporation', 'question':'Investigate the event', 'conclusion':'insufficient_evidence',
            'findings':[{'statement':'The company reported significant capital investment associated with infrastructure expansion', 'accession':A,'excerpt':'significant capital investment associated with infrastructure expansion'}],
            'financial_indicators':[], 'direct_evidence':[], 'limitations':[], 'sources':[S(A,'2026-06-22'),S(B,'2025-06-18')]}

def test_generic_keyword_finding_rejected():
    r=sample();r['findings'][0]['statement']='Filing mentions data center'
    assert any('substantive' in e for e in validate_research(r))

def test_selective_apa_references():
    out=format_research_answer(sample())
    assert out.count('https://')==1
    assert '(Oracle Corporation, 2026)' in out
    assert '2025' not in out

def test_same_year_suffixes():
    r=sample();r['sources'][1]['filing_date']='2026-03-18'
    r['findings'].append({'statement':'The company described lease commitments associated with data center expansion', 'accession':B, 'excerpt':'lease commitments associated with data center expansion'})
    out=format_research_answer(r)
    assert '(Oracle Corporation, 2026a)' in out and '(Oracle Corporation, 2026b)' in out
    assert out.count('https://')==2

def test_trends_exact_accession_and_no_fake_fcf():
    facts={'facts':{'us-gaap':{'NetIncomeLoss':{'units':{'USD':[
        {'form':'10-K','accn':A,'start':'2025-06-01','end':'2026-05-31','val':100},
        {'form':'10-K','accn':B,'start':'2024-06-01','end':'2025-05-31','val':80},
        {'form':'10-K','accn':B,'start':'2025-06-01','end':'2026-05-31','val':999},
    ]}}}}}
    filings=[{'form':'10-K','accessionNumber':A,'reportDate':'2026-05-31','filingDate':'2026-06-22'},
             {'form':'10-K','accessionNumber':B,'reportDate':'2025-05-31','filingDate':'2025-06-18'}]
    result=financial_trends('0001341439',facts=facts,filings=filings)
    assert result['rows'][0]['metrics']['net_income']['value']==100
    assert result['rows'][0]['missing']
    assert 'free_cash_flow' not in result['rows'][0]['metrics']
    assert any(c['absolute_change']==20 for c in result['changes'])
