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
            "make_trend_indicators": lambda trends, evidence: [],
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


class TestHelperPreservation:
    """Verify that helper functions remain available for backward compatibility."""

    def test_make_contract_still_exists(self):
        assert callable(agent.make_contract)

    def test_validate_ratio_still_exists(self):
        assert callable(agent.validate_ratio)

    def test_validate_research_still_exists(self):
        assert callable(agent.validate_research)

    def test_research_complete_still_exists(self):
        assert callable(agent.research_complete)

    def test_validate_result_still_exists(self):
        assert callable(agent.validate_result)

    def test_format_answer_still_exists(self):
        assert callable(agent.format_answer)

    def test_format_ratio_answer_still_exists(self):
        assert callable(agent.format_ratio_answer)

    def test_format_research_answer_still_exists(self):
        assert callable(agent.format_research_answer)

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
        assert "make_trend_indicators" in tools
        assert "prepare_evidence" in tools
