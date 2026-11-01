import pytest

import accounting
import sec_data
import secedgar_agent as agent

ACC = '0001769628-26-000104'
PERIOD = '2025-12-31'


def test_select_annual_and_instant():
    annual = [{'start':'2025-01-01','end':PERIOD,'val':-1167,'form':'10-K','accn':ACC}]
    instant = [{'end':PERIOD,'val':6708,'form':'10-K','accn':ACC}]
    assert sec_data.select_observation(annual,period=PERIOD,accession=ACC,duration=True)['val'] == -1167
    assert sec_data.select_observation(instant,period=PERIOD,accession=ACC)['val'] == 6708
    with pytest.raises(ValueError, match='Missing'):
        sec_data.select_observation(annual,period=PERIOD,accession=ACC)


def test_select_rejects_comparatives_and_conflicts():
    observations = [{'end':PERIOD,'val':1,'form':'10-K','accn':ACC},
                    {'end':PERIOD,'val':2,'form':'10-Q','accn':'other'}]
    assert sec_data.select_observation(observations,period=PERIOD,accession=ACC)['val'] == 1
    with pytest.raises(ValueError, match='Conflicting'):
        sec_data.select_observation(observations + [{'end':PERIOD,'val':3,'form':'10-K','accn':ACC}],period=PERIOD,accession=ACC)


def test_build_debt_requires_both_components(monkeypatch):
    def observations(cik, concept):
        if concept == 'LongTermDebtCurrent':
            return [{'end':PERIOD,'val':6708,'form':'10-K','accn':ACC}]
        return []
    monkeypatch.setattr(sec_data, 'get_annual_observations', observations)
    with pytest.raises(ValueError, match='Missing'):
        accounting.build_debt_composition('x',period=PERIOD,accession=ACC)


def test_build_debt_uses_exact_tags(monkeypatch):
    values = {'LongTermDebtCurrent':6708, 'LongTermDebtNoncurrent':14665}
    monkeypatch.setattr(sec_data,'get_annual_observations',lambda cik, concept: [
        {'end':PERIOD,'val':values[concept],'form':'10-K','accn':ACC}])
    composition = accounting.build_debt_composition('x',period=PERIOD,accession=ACC)
    assert composition['total_debt'] == 21373
    assert [x['concept'] for x in composition['debt_components']] == list(accounting.DEBT_STRATEGIES['current_plus_noncurrent'])


def test_extract_code_json_wrapper():
    assert agent.extract_code('{"code": "x = 1\\nprint(x)"}') == 'x = 1\nprint(x)'
    with pytest.raises(ValueError, match='wrapper'):
        agent.extract_code('{"code": 4}')

def test_extract_code_fenced_json_wrapper():
    assert agent.extract_code('```json\n{"code":"x = 2"}\n```') == 'x = 2'
