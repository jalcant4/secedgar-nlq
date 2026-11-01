"""Deterministic, accession-grounded comparisons of SEC Company Facts observations."""
import itertools
from math import isfinite

METRICS = {
    'net_income': ('NetIncomeLoss', True),
    'operating_cash_flow': ('NetCashProvidedByUsedInOperatingActivities', True),
    'capital_expenditures': ('PaymentsToAcquirePropertyPlantAndEquipment', True),
    'cash_and_equivalents': ('CashAndCashEquivalentsAtCarryingValue', False),
    'debt_carrying_amount': ('DebtInstrumentCarryingAmount', False),
}

def financial_trends(cik, *, facts=None, filings=None, periods=3):
    """Return only values present in the corresponding 10-K, with exact accession.

    Missing metrics are explicit. CapEx and OCF are *not* inferred from net income;
    free cash flow is computed only when both corresponding flow values exist.
    Debt carrying amount is not labeled total interest-bearing debt.
    """
    if not 1 <= periods <= 10:
        raise ValueError('periods must be 1..10')
    if facts is None:
        from edgar import get_company_facts
        facts = get_company_facts(cik)
    if filings is None:
        from sec_data import get_recent_filings
        filings = get_recent_filings(cik, '10-K')
    annual = sorted((f for f in filings if f.get('form') == '10-K' and f.get('reportDate')
                     and f.get('accessionNumber')), key=lambda f:f['reportDate'], reverse=True)
    selected = []
    seen = set()
    for f in annual:
        if f['reportDate'] not in seen:
            selected.append(f)
            seen.add(f['reportDate'])
        if len(selected) == periods:
            break
    gaap = facts.get('facts', {}).get('us-gaap', {})
    rows = []
    for f in selected:
        end, accession = f['reportDate'], f['accessionNumber']
        row = {'period':end, 'accession':accession, 'filing_date':f.get('filingDate', ''), 'metrics':{}, 'missing':[]}
        for label, (concept, duration) in METRICS.items():
            observations = gaap.get(concept, {}).get('units', {}).get('USD', [])
            matches = [o for o in observations if o.get('end') == end and o.get('accn') == accession
                       and o.get('form') == '10-K' and isinstance(o.get('val'), (int,float))
                       and isfinite(o['val']) and (bool(o.get('start')) == duration)]
            if duration:
                matches = [o for o in matches if 330 <= (__import__('datetime').date.fromisoformat(o['end']) - __import__('datetime').date.fromisoformat(o['start'])).days <= 380]
            if matches:
                row['metrics'][label] = {'value':matches[0]['val'], 'concept':concept}
            else:
                row['missing'].append(label)
        ocf = row['metrics'].get('operating_cash_flow')
        capex = row['metrics'].get('capital_expenditures')
        if ocf and capex:
            row['metrics']['free_cash_flow'] = {'value':ocf['value']-capex['value'], 'concept':'derived:operating_cash_flow_minus_capital_expenditures'}
        rows.append(row)
    changes = []
    for newer, older in itertools.pairwise(rows):
        for label in newer['metrics'].keys() & older['metrics'].keys():
            a, b = newer['metrics'][label]['value'], older['metrics'][label]['value']
            changes.append({'metric':label, 'from_period':older['period'], 'to_period':newer['period'],
                            'from_value':b, 'to_value':a, 'absolute_change':a-b,
                            'percent_change':(a-b)/abs(b)*100 if b else None,
                            'accessions':[older['accession'], newer['accession']]})
    return {'rows':rows, 'changes':changes, 'units':'USD', 'notes':['Missing data are not zero.', 'DebtInstrumentCarryingAmount is not necessarily total interest-bearing debt.', 'Free cash flow is calculated as operating cash flow minus capital expenditures when both are present.']}
