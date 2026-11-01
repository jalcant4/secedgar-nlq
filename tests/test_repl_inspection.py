from repl import REPL


def test_inspection_preserves_state_and_keys():
    repl = REPL()
    repl.execute("filing = {'accessionNumber': 'abc', 'reportDate': '2025-01-01'}")
    assert repl.inspect_state('filing', 'keys') == ['accessionNumber', 'reportDate']
    assert repl.inspect_state()['filing'] == 'dict'
    assert repl.namespace['filing']['accessionNumber'] == 'abc'


def test_inspection_after_exception_and_recovery():
    repl = REPL()
    repl.execute("x = 3\nraise ValueError('bad key')")
    assert 'bad key' in repl.inspect_state(mode='last_error')
    assert repl.execute("print(x)").strip() == '3'
    assert repl.inspect_state(mode='last_error') == 'No execution error'


def test_validation_feedback_and_bounded_preview():
    repl = REPL()
    repl.validation = {'status': 'rejected', 'errors': ['wrong accession']}
    repl.execute("data = 'a' * 10000")
    assert repl.inspect_state(mode='validation')['errors'] == ['wrong accession']
    assert len(repl.inspect_state('data', max_chars=64)) <= 64
    assert len(repl.namespace['data']) == 10000


def test_missing_and_invalid_modes():
    repl = REPL()
    assert 'error' in repl.inspect_state('absent')
    assert 'error' in repl.inspect_state('inspect', 'keys')
    assert 'error' in repl.inspect_state(mode='keys')
