"""Tests for Solution C: DRAFT: → verification → FINAL: protocol."""
from unittest.mock import MagicMock

from secedgar_agent import (
    check_period,
    extract_draft,
    is_draft_response,
    is_final_response,
    research,
    verify_draft,
)


class TestIsDraftResponse:
    def test_draft_marker_detected(self):
        assert is_draft_response("DRAFT: Apple net income is $100B") is True

    def test_draft_with_whitespace(self):
        assert is_draft_response("  \n  DRAFT:  answer\n  ") is True

    def test_final_is_not_draft(self):
        assert is_draft_response("FINAL: answer") is False

    def test_python_code_is_not_draft(self):
        assert is_draft_response("print('hello')") is False

    def test_empty_string_is_not_draft(self):
        assert is_draft_response("") is False


class TestExtractDraft:
    def test_basic_draft(self):
        response = "DRAFT: Apple net income is $100B"
        assert extract_draft(response) == "Apple net income is $100B"

    def test_draft_with_newlines(self):
        response = "DRAFT:\nApple Inc.\nNet income: $100B\n"
        assert extract_draft(response) == "Apple Inc.\nNet income: $100B"

    def test_draft_with_leading_whitespace(self):
        response = "  DRAFT:   hello  "
        assert extract_draft(response) == "hello"


class TestCheckPeriod:
    def test_valid_period(self):
        assert check_period("2025-01-01", "2025-12-31") is True

    def test_equal_dates(self):
        assert check_period("2025-06-30", "2025-06-30") is True

    def test_invalid_period_start_after_end(self):
        assert check_period("2025-10-01", "2025-09-30") is False

    def test_unknown_start(self):
        assert check_period("unknown", "2025-12-31") is True

    def test_nslasha_start(self):
        assert check_period("N/A", "2025-12-31") is True

    def test_none_start(self):
        assert check_period(None, "2025-12-31") is True


class TestVerifyDraft:
    def test_clean_draft_passes(self):
        repl = MagicMock()
        repl.namespace = {"filings": []}
        draft = "Apple's net income was $100B (Apple Inc., 2025)."
        result = verify_draft(draft, repl)
        assert "VERIFICATION PASSED" in result

    def test_impossible_date_range_detected(self):
        repl = MagicMock()
        repl.namespace = {"filings": []}
        draft = "Period: 2025-10-01 → 2025-09-30"
        result = verify_draft(draft, repl)
        assert "VERIFICATION ISSUES" in result
        assert "Impossible date range" in result

    def test_suspicious_invented_period_start(self):
        repl = MagicMock()
        repl.namespace = {"filings": []}
        draft = "fiscal period start is 2025-01-01"
        result = verify_draft(draft, repl)
        assert "VERIFICATION ISSUES" in result
        assert "Invented fiscal-period start" in result

    def test_unmatched_accession_number(self):
        repl = MagicMock()
        repl.namespace = {
            "filings": [{"accessionNumber": "0001769628-26-000104"}]
        }
        draft = "Accession 0001769628-26-000999 was cited."
        result = verify_draft(draft, repl)
        assert "VERIFICATION ISSUES" in result
        assert "not found in retrieved filings" in result

    def test_match_with_retrieved_accession(self):
        repl = MagicMock()
        repl.namespace = {
            "filings": [{"accessionNumber": "0001769628-26-000104"}]
        }
        draft = "Accession 0001769628-26-000104 was cited."
        result = verify_draft(draft, repl)
        assert "VERIFICATION PASSED" in result


class TestResearchDraftFlow:
    def test_draft_then_final(self):
        """Model returns DRAFT:, gets feedback, then FINAL."""
        responses = [
            "DRAFT:\nApple net income = $100B\n",
            "FINAL:\nApple Inc.\n$100B\n",
        ]
        call_count = [0]

        def mock_query(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        import secedgar_agent
        original = secedgar_agent.llm_query
        secedgar_agent.llm_query = mock_query
        try:
            result = research("What is Apple's net income?", max_steps=5)
            assert result is not None
            assert "FINAL:" in result
            assert call_count[0] == 2
        finally:
            secedgar_agent.llm_query = original

    def test_draft_then_code_then_final(self):
        """Model returns DRAFT:, gets feedback, runs more code, then FINAL."""
        responses = [
            "DRAFT:\nApple net income = $100B\n",
            "```python\nprint('verifying')\n```",
            "FINAL:\nApple Inc.\nVerified: $100B\n",
        ]
        call_count = [0]

        def mock_query(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        import secedgar_agent
        original = secedgar_agent.llm_query
        secedgar_agent.llm_query = mock_query
        try:
            result = research("What is Apple's net income?", max_steps=5)
            assert result is not None
            assert "FINAL:" in result
            assert call_count[0] == 3
        finally:
            secedgar_agent.llm_query = original

    def test_final_without_draft(self):
        """Model can skip DRAFT and go straight to FINAL."""
        responses = ["FINAL:\nApple Inc.\n$100B\n"]
        call_count = [0]

        def mock_query(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        import secedgar_agent
        original = secedgar_agent.llm_query
        secedgar_agent.llm_query = mock_query
        try:
            result = research("What is Apple's net income?", max_steps=5)
            assert result is not None
            assert call_count[0] == 1
        finally:
            secedgar_agent.llm_query = original

    def test_draft_consumes_step_budget(self):
        """DRAFT responses consume a step from the budget."""
        responses = ["DRAFT:\nanswer\n"] * 31  # 31 drafts > max_steps=30
        call_count = [0]

        def mock_query(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        import secedgar_agent
        original = secedgar_agent.llm_query
        secedgar_agent.llm_query = mock_query
        try:
            result = research("Test", max_steps=30)
            assert result is None
            assert call_count[0] == 30
        finally:
            secedgar_agent.llm_query = original
