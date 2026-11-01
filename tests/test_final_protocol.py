"""Tests for the simplified FINAL: response protocol."""
import pytest
from unittest.mock import patch, MagicMock

import secedgar_agent as agent


class TestIsFinalResponse:
    def test_final_marker_detected(self):
        assert agent.is_final_response("FINAL: The ratio is 1.24") is True

    def test_final_with_whitespace(self):
        assert agent.is_final_response("  FINAL: Apple's ratio is 1.24\n") is True

    def test_python_code_not_final(self):
        assert agent.is_final_response("```python\nprint('hello')\n```") is False

    def test_result_dict_not_final(self):
        assert agent.is_final_response("result = {'company': 'Apple'}") is False

    def test_empty_string_not_final(self):
        assert agent.is_final_response("") is False

    def test_none_raises(self):
        with pytest.raises(AttributeError):
            agent.is_final_response(None)


class TestExtractCode:
    def test_fenced_python(self):
        code = agent.extract_code("```python\nx = 1\nprint(x)\n```")
        assert code == "x = 1\nprint(x)"

    def test_fenced_python_no_lang(self):
        code = agent.extract_code("```\nx = 2\n```")
        assert code == "x = 2"

    def test_json_wrapper(self):
        code = agent.extract_code('{"code": "y = 3\\nprint(y)"}')
        assert code == "y = 3\nprint(y)"

    def test_fenced_json_wrapper(self):
        code = agent.extract_code("```json\n{\"code\": \"z = 4\"}\n```")
        assert code == "z = 4"

    def test_empty_code_rejected(self):
        with pytest.raises(ValueError, match="no code"):
            agent.extract_code("```python\n```")

    def test_syntax_error_rejected(self):
        with pytest.raises(SyntaxError):
            agent.extract_code("```python\nif True\n```")

    def test_malformed_json_rejected(self):
        with pytest.raises(ValueError, match="wrapper"):
            agent.extract_code('{"code": 42}')

    def test_plain_python_no_fence(self):
        """Plain Python without a fence is also accepted."""
        code = agent.extract_code("x = 5\nprint(x)")
        assert code == "x = 5\nprint(x)"


class TestResearchLoop:
    """Test the simplified research loop with mocked LLM."""

    def test_final_response_accepted_immediately(self):
        """Model returns FINAL: on first turn."""
        with patch('secedgar_agent.llm_query') as mock_llm:
            mock_llm.return_value = "FINAL: The answer is 42."
            result = agent.research("What is 6 * 7?", max_steps=5)
        assert result == "FINAL: The answer is 42."

    def test_code_then_final(self):
        """Model does code first, then returns FINAL:."""
        responses = [
            "```python\nx = 6 * 7\nprint(x)\n```",
            "FINAL: The product is 42.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("What is 6 * 7?", max_steps=5)
        assert result == "FINAL: The product is 42."
        assert mock_llm.call_count == 2

    def test_syntax_error_does_not_consume_turn(self):
        """Syntax errors are retried without burning a step."""
        responses = [
            "```python\nif True\n```",  # syntax error
            "FINAL: The answer is 42.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("What is 6 * 7?", max_steps=5)
        assert result == "FINAL: The answer is 42."
        # Only 1 step consumed (the successful FINAL response is not a step)
        assert mock_llm.call_count == 2

    def test_malformed_code_does_not_consume_turn(self):
        """Malformed JSON wrapper is retried without burning a step."""
        responses = [
            '{"code": 42}',  # malformed JSON
            "FINAL: The answer is 42.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("What is 6 * 7?", max_steps=5)
        assert result == "FINAL: The answer is 42."
        assert mock_llm.call_count == 2

    def test_max_steps_exhausted(self):
        """Model never returns FINAL: — loop exits after max_steps."""
        responses = ["```python\nprint('thinking')\n```"] * 10
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("What is 6 * 7?", max_steps=3)
        assert result is None
        assert mock_llm.call_count == 3

    def test_inference_error_returns_none(self):
        """LLM query failure returns None."""
        with patch('secedgar_agent.llm_query', side_effect=OSError("offline")):
            result = agent.research("What is 6 * 7?", max_steps=5)
        assert result is None

    def test_persistent_repl_state(self):
        """Variables persist across REPL executions."""
        responses = [
            "```python\nx = 10\n```",
            "```python\ny = x * 2\nprint(y)\n```",
            "FINAL: The result is 20.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("Double 10", max_steps=5)
        assert result == "FINAL: The result is 20."

    def test_execution_error_preserves_state(self):
        """REPL errors don't wipe state; model can retry."""
        responses = [
            "```python\nx = 10\n```",
            "```python\nx / 0\n```",  # execution error
            "```python\ny = x * 2\nprint(y)\n```",
            "FINAL: The result is 20.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("Double 10", max_steps=5)
        assert result == "FINAL: The result is 20."

    def test_no_result_dict_needed(self):
        """Model can return FINAL: without assigning result dict."""
        mock_tools = {
            "get_cik": lambda c: "0000320193",
            "get_company_facts": lambda cik: {"entityName": "Apple Inc."},
            "get_observations": lambda cik, concept: [],
            "get_annual_observations": lambda cik, concept: [],
            "get_recent_filings": lambda cik, form=None: [],
            "get_filings": lambda cik: {"accessionNumber": [], "form": [], "reportDate": [], "filingDate": []},
            "get_filing_text": lambda cik, acc, primary: {"text": "", "url": ""},
            "search_filing": lambda text, terms: [],
            "filing_url": lambda cik, acc, doc: f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}",
            "find_concepts": lambda cik, metric: [],
            "financial_trends": lambda cik, periods: {"rows": [], "changes": []},
            "review_debt_footnote": lambda cik, accession: {"status": "not_reviewed"},
            "research_filings": lambda cik, forms, terms, max_filings: {"findings": [], "sources": [], "failures": []},
            "build_debt_composition": lambda cik, period, accession, method: {"total_debt": 0},
            "select_observation": lambda obs, **kw: obs[0] if obs else {},
            "normalize_concept": lambda c: c,
            "prepare_evidence": lambda evidence, trends: {"snippets": [], "trends": {"rows": [], "changes": []}},
            "SUPPORTED_ONTOLOGY_METRICS": (),
            "DEBT_STRATEGIES": {},
            "_retrieved_documents": {},
        }
        responses = [
            "```python\n"
            "cik = get_cik('Apple Inc.')\n"
            "facts = get_company_facts(cik)\n"
            "print(facts.get('entityName', 'unknown'))\n"
            "```",
            "FINAL:\nApple's latest annual net-income-to-debt ratio is 1.24.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            with patch('secedgar_agent.sec_tools', return_value=mock_tools):
                result = agent.research("What is Apple's ratio?", max_steps=10)
        assert result is not None
        assert "Apple" in result
        assert "1.24" in result

    def test_build_prompt_includes_protocol_and_tools(self):
        """build_prompt composes protocol + tools + question + history."""
        prompt = agent.build_prompt("What is Apple's revenue?", [])
        assert "Response protocol" in prompt
        assert "FINAL:" in prompt
        assert "SEC tools available" in prompt
        assert "What is Apple's revenue?" in prompt
        assert "Recent steps" in prompt

    def test_build_prompt_with_history(self):
        """build_prompt includes recent history."""
        history = ["CODE:\nprint(x)\nOBSERVATION:\n10", "CODE:\nprint(x*2)\nOBSERVATION:\n20"]
        prompt = agent.build_prompt("What's next?", history)
        assert "10" in prompt
        assert "20" in prompt


class TestBoundedInvalidResponses:
    """Invalid responses are bounded so the loop cannot run indefinitely."""

    def test_invalid_responses_accumulate(self):
        """Syntax errors increment the invalid counter."""
        responses = ["```python\nif True\n```"] * 5 + ["FINAL: done."]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=20)
        assert result is None  # exceeded MAX_INVALID_RESPONSES

    def test_valid_code_resets_invalid_count(self):
        """A successful execution prevents the invalid counter from triggering."""
        responses = [
            "```python\nif True\n```",  # invalid
            "```python\nx = 1\n```",    # valid
            "```python\nif True\n```",  # invalid again
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=20)
        assert result == "FINAL: done."

    def test_malformed_json_accumulates(self):
        """Malformed JSON wrappers also increment the invalid counter."""
        responses = ['{"code": 42}'] * 5 + ["FINAL: done."]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=20)
        assert result is None

    def test_mixed_invalid_then_final(self):
        """Mix of syntax errors and malformed JSON before FINAL."""
        responses = [
            "```python\nbad\n```",       # syntax
            '{"code": 1}',               # malformed JSON
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=20)
        assert result == "FINAL: done."

    def test_max_steps_still_works(self):
        """Normal max_steps exhaustion still works."""
        responses = ["```python\nprint(1)\n```"] * 35
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=30)
        assert result is None


class TestSecTools:
    def test_sec_tools_exposes_data_accessors(self):
        tools = agent.sec_tools()
        assert "get_cik" in tools
        assert "get_company_facts" in tools
        assert "get_observations" in tools
        assert "get_annual_observations" in tools
        assert "find_concepts" in tools
        assert "get_filings" in tools
        assert "get_recent_filings" in tools
        assert "get_filing_text" in tools
        assert "search_filing" in tools
        assert "filing_url" in tools

    def test_sec_tools_exposes_analysis_helpers(self):
        tools = agent.sec_tools()
        assert "financial_trends" in tools
        assert "review_debt_footnote" in tools
        assert "research_filings" in tools
        assert "build_debt_composition" in tools
        assert "select_observation" in tools
        assert "normalize_concept" in tools
        assert "prepare_evidence" in tools

    def test_sec_tools_tracked_documents(self):
        tools = agent.sec_tools()
        assert tools["_retrieved_documents"] == {}
        with patch('secedgar_agent.get_filing_text', return_value={"text": "hello", "url": "https://example.com"}):
            doc = tools["get_filing_text"]("0000320193", "0001769628-25-000001", "primary_doc.xml")
        assert "0001769628-25-000001" in tools["_retrieved_documents"]


class TestCompletionReminder:
    """Near the step limit, the prompt warns the model to finalize."""

    def test_prompt_contains_finalization_instruction(self):
        """Normal prompt tells the model to decide whether to finalize."""
        prompt = agent.build_prompt("What is Apple's revenue?", [], remaining=10)
        assert "Decide whether the question has been answered" in prompt
        assert "return FINAL:" in prompt
        assert "Only" not in prompt

    def test_prompt_contains_budget_warning_near_limit(self):
        """When 3 or fewer steps remain, a budget warning is added."""
        prompt = agent.build_prompt("What is Apple's revenue?", [], remaining=3)
        assert "Only 3 execution steps remain" in prompt
        assert "produce FINAL:" in prompt

    def test_prompt_contains_budget_warning_at_one_step(self):
        prompt = agent.build_prompt("test", [], remaining=1)
        assert "Only 1 execution steps remain" in prompt

    def test_budget_warning_not_added_with_sufficient_steps(self):
        prompt = agent.build_prompt("test", [], remaining=5)
        assert "Only" not in prompt


class TestRepeatedErrorDetection:
    """Repeated execution failures trigger a corrective observation."""

    def test_repeated_typeerror_triggers_corrective_feedback(self):
        """Three identical TypeErrors produce a corrective observation."""
        responses = [
            "```python\nx / 0\n```",
            "```python\nx / 0\n```",
            "```python\nx / 0\n```",
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("test", max_steps=10)
        # The model should get the corrective feedback before the 4th call.
        assert mock_llm.call_count == 4
        assert result == "FINAL: done."

    def test_different_errors_do_not_trigger_repeat_guard(self):
        """Two different errors followed by success should not trigger the guard."""
        responses = [
            "```python\nx / 0\n```",
            "```python\n{}['missing']\n```",
            "```python\nprint('ok')\n```",
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("test", max_steps=10)
        assert result == "FINAL: done."
        # All 4 calls should succeed (no guard triggered)
        assert mock_llm.call_count == 4

    def test_same_error_resets_after_success(self):
        """A successful execution resets the repeated-error counter."""
        responses = [
            "```python\nx / 0\n```",
            "```python\nx / 0\n```",
            "```python\nprint('recovered')\n```",
            "```python\nx / 0\n```",
            "```python\nx / 0\n```",
            "```python\nx / 0\n```",
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("test", max_steps=20)
        assert result == "FINAL: done."
        # Counter resets after success; then 3 more identical errors trigger the guard.
        # Calls: 2 errors + 1 success + 3 errors (guard fires) + FINAL = 7
        assert mock_llm.call_count == 7

    def test_error_signature_normalization(self):
        """Different traceback line numbers should match the same error type."""
        sig1 = agent._error_signature(
            'Traceback (most recent call last):\n'
            '  File "<string>", line 3, in <module>\n'
            'TypeError: unsupported operand type(s) for /: int and int'
        )
        sig2 = agent._error_signature(
            'Traceback (most recent call last):\n'
            '  File "<string>", line 7, in <module>\n'
            'TypeError: unsupported operand type(s) for /: int and int'
        )
        assert sig1 == sig2

    def test_max_steps_still_triggers_with_repeated_errors(self):
        """Even with the repeated-error guard, max_steps still terminates."""
        responses = ["```python\nx / 0\n```"] * 35
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=30)
        assert result is None
