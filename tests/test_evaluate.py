"""Offline tests for the golden-set scoring logic (no LLM, no network)."""
from evaluate import score_question, value_variants


def test_value_variants_covers_common_renderings():
    variants = value_variants(-1167000000)
    assert "-1,167,000,000" in variants
    assert "$-1,167,000,000" in variants or "-$1,167,000,000" in variants
    assert "-1.167B" in variants
    assert "-1167.0M" in variants
    assert "-1.167 billion" in variants
    assert "-$1.167 billion" in variants
    assert "$(1,167,000,000)" in variants
    assert "$1.167 billion" in variants  # "net loss of $1.167 billion"


def test_accession_accepted_in_url_form():
    item = {"question": "q",
            "checks": [{"type": "accession_cited", "value": "0001769628-26-000104"}]}
    checks = score_question(item, "See 000176962826000104 for details.")
    assert checks[1]["pass"] is True


def test_accession_missing_fails():
    item = {"question": "q",
            "checks": [{"type": "accession_cited", "value": "0001769628-26-000104"}]}
    checks = score_question(item, "No accession here.")
    assert checks[1]["pass"] is False


def test_score_completed_answer_with_all_checks_passing():
    item = {
        "question": "q",
        "checks": [
            {"type": "accession_cited", "value": "0001769628-26-000104"},
            {"type": "value_in_answer", "name": "ni", "value": -1167000000},
            {"type": "text_in_answer", "values": ["2025-12-31", "December 31, 2025"]},
        ],
    }
    answer = ("Net income was -$1,167,000,000 for the year ended 2025-12-31. "
              "Source accession 0001769628-26-000104.")
    checks = score_question(item, answer)
    assert all(c["pass"] for c in checks), checks


def test_score_fails_on_missing_accession():
    item = {"question": "q",
            "checks": [{"type": "accession_cited", "value": "0001769628-26-000104"}]}
    checks = score_question(item, "An answer without the accession.")
    assert checks[0]["pass"] is True  # completed
    assert checks[1]["pass"] is False


def test_score_none_answer_is_failure():
    checks = score_question({"question": "q", "checks": []}, None)
    assert checks == [{"name": "completed", "pass": False}]


def test_unknown_check_type_fails():
    item = {"question": "q", "checks": [{"type": "bogus", "value": 1}]}
    checks = score_question(item, "answer")
    assert checks[-1]["pass"] is False
    assert "unknown_check_type" in checks[-1]["name"]
