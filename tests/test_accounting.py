from accounting import validate_debt_composition

ACC = '0001769628-26-000104'
def fact(value):
    return {'units': {'USD': [{'form': '10-K', 'end': '2025-12-31', 'accn': ACC, 'val': value}]}}
FACTS = {'facts': {'us-gaap': {
    'LongTermDebtCurrent': fact(6708),
    'LongTermDebtNoncurrent': fact(14665),
    'LongTermDebt': fact(21373),
    'InterestExpenseDebt': fact(1148),
}}}
BASE = {'period': '2025-12-31', 'sources': [ACC], 'debt_method': 'current_plus_noncurrent',
        'total_debt': 21373, 'debt_components': [
            {'concept': 'LongTermDebtCurrent', 'value': 6708},
            {'concept': 'LongTermDebtNoncurrent', 'value': 14665}]}

def test_reconciled():
    assessment = validate_debt_composition(BASE, FACTS)
    assert not assessment.errors
    assert assessment.warnings

def test_reject_double_count():
    candidate = {**BASE, 'total_debt': 42746, 'debt_components': BASE['debt_components'] + [{'concept': 'LongTermDebt', 'value': 21373}]}
    assert validate_debt_composition(candidate, FACTS).errors

def test_reject_expense():
    candidate = {**BASE, 'total_debt': 22521, 'debt_components': BASE['debt_components'] + [{'concept': 'InterestExpenseDebt', 'value': 1148}]}
    assert validate_debt_composition(candidate, FACTS).errors

def test_reject_wrong_reconciliation():
    candidate = {**BASE, 'total_debt': 20000, 'debt_components': [BASE['debt_components'][0], {'concept': 'LongTermDebtNoncurrent', 'value': 13292}]}
    assert validate_debt_composition(candidate, FACTS).errors

def test_qualified_debt_identifiers_accepted():
    candidate = {**BASE, 'debt_components': [
        {'concept': 'us-gaap:' + c['concept'], 'value': c['value']}
        for c in BASE['debt_components']]}
    assert not validate_debt_composition(candidate, FACTS).errors


def test_mixed_qualified_identifiers_rejected_as_duplicate():
    candidate = {**BASE, 'debt_components': BASE['debt_components'] + [
        {'concept': 'us-gaap:LongTermDebtCurrent', 'value': 6708}], 'total_debt': 28081}
    assert validate_debt_composition(candidate, FACTS).errors
