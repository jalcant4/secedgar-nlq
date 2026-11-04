"""Tests for the simplified FINAL: response protocol."""
from unittest.mock import patch

import pytest

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
            ("```python\n"
             "cik = get_cik('Apple Inc.')\n"
             "facts = get_company_facts(cik)\n"
             "print(facts.get('entityName', 'unknown'))\n"
             "```"),
            "FINAL:\nApple's latest annual net-income-to-debt ratio is 1.24.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses), \
             patch('secedgar_agent.sec_tools', return_value=mock_tools):
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
            tools["get_filing_text"]("0000320193", "0001769628-25-000001", "primary_doc.xml")
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

    def test_repeated_errors_still_respect_max_steps(self):
        """Even with the repeated-error guard, max_steps still terminates."""
        responses = ["```python\nx / 0\n```"] * 35
        with patch('secedgar_agent.llm_query', side_effect=responses):
            result = agent.research("test", max_steps=30)
        assert result is None

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


class TestStructuredFinalAnswer:
    """The prompt guides the model toward a structured final answer."""

    def test_prompt_contains_structured_format_guidance(self):
        """Normal prompt tells the model about RESEARCH SUMMARY and REFERENCES."""
        prompt = agent.build_prompt("test", [], remaining=10)
        assert "RESEARCH SUMMARY" in prompt
        assert "REFERENCES" in prompt
        assert "CSV TABLE" in prompt or "structured data" in prompt

    def test_prompt_contains_citation_rules(self):
        """Normal prompt includes citation rules."""
        prompt = agent.build_prompt("test", [], remaining=10)
        assert "Do not fabricate" in prompt
        assert "Do not cite a source that was not retrieved" in prompt

    def test_structured_final_answers_accepted(self):
        """FINAL: answers with CSV tables and RESEARCH SUMMARY/REFERENCES are accepted."""
        with_summary = (
            "FINAL:\n"
            "CoreWeave, Inc.\n"
            "\n"
            "RESEARCH SUMMARY\n"
            "CoreWeave had four filings (CoreWeave, Inc., 2025, 2026a, 2026b, 2026c).\n"
            "\n"
            "REFERENCES\n"
            "[1] CoreWeave, Inc. (2025). Form 10-Q.\n"
        )
        with_csv = (
            "FINAL:\n"
            "fiscal_period_start,fiscal_period_end,filing_form\n"
            "2025-01-01,2025-12-31,10-K\n"
            "\n"
            "RESEARCH SUMMARY\n"
            "One 10-K filing was identified.\n"
            "\n"
            "REFERENCES\n"
            "[1] CoreWeave, Inc. (2026). Form 10-K.\n"
        )
        for answer in (with_summary, with_csv):
            assert agent.is_final_response(answer) is True
            assert answer.strip().startswith("FINAL:")

    def test_prompt_contains_evidence_handling_rules(self):
        """Normal prompt includes evidence handling guidance."""
        prompt = agent.build_prompt("test", [], remaining=10)
        assert "Never replace missing observations with guessed" in prompt
        assert "N/A and explain why" in prompt
        assert "review your conclusions against the" in prompt
        assert "already obtained" in prompt

    def test_prompt_contains_apa_disambiguation_rule(self):
        """Normal prompt includes APA year-suffix disambiguation guidance."""
        prompt = agent.build_prompt("test", [], remaining=10)
        assert "2026a, 2026b, 2026c" in prompt or "suffixes" in prompt.lower()
        assert "filing dates" in prompt

    def test_prompt_contains_standard_library_guidance(self):
        """Normal prompt instructs to use standard library, not pandas."""
        prompt = agent.build_prompt("test", [], remaining=10)
        assert "standard library" in prompt.lower()
        assert "pandas" in prompt.lower()


class TestEvidenceProvenance:
    """Evidence dataclass and verify_claim helper."""

    def test_evidence_dataclass_defaults(self):
        e = agent.Evidence(claim="x", value=1, source="url")
        assert e.claim == "x"
        assert e.value == 1
        assert e.source == "url"
        assert e.method == "retrieved"

    def test_evidence_dataclass_immutable(self):
        e = agent.Evidence(claim="x", value=1, source="url")
        with pytest.raises(AttributeError):
            e.claim = "y"

    def test_evidence_dataclass_hashable(self):
        e1 = agent.Evidence(claim="x", value=1, source="url")
        e2 = agent.Evidence(claim="x", value=1, source="url")
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_verify_claim_all_match(self):
        sources = [
            {"accessionNumber": "0001-26-001", "reportDate": "2026-06-30"},
            {"accessionNumber": "0001-26-002", "reportDate": "2026-03-31"},
        ]
        claim = {"accessionNumber": "0001-26-001", "reportDate": "2026-06-30"}
        results = agent.verify_claim(claim, sources)
        assert results["accessionNumber"]["match"] is True
        assert results["reportDate"]["match"] is True

    def test_verify_claim_mismatch(self):
        sources = [
            {"accessionNumber": "0001-26-001", "reportDate": "2026-06-30"},
        ]
        claim = {"accessionNumber": "0001-26-001", "reportDate": "2025-12-31"}
        results = agent.verify_claim(claim, sources)
        assert results["reportDate"]["match"] is False
        assert results["reportDate"]["expected"] == "2025-12-31"
        assert results["reportDate"]["actual"] == "2026-06-30"

    def test_verify_claim_missing_source(self):
        sources = [
            {"accessionNumber": "0001-26-001", "reportDate": "2026-06-30"},
        ]
        claim = {"accessionNumber": "0001-26-999", "reportDate": "2026-06-30"}
        results = agent.verify_claim(claim, sources)
        assert results["accessionNumber"]["match"] is False
        assert results["accessionNumber"]["actual"] is None

    def test_verify_claim_custom_key_field(self):
        sources = [
            {"cik": "0001769628", "form": "10-Q"},
        ]
        claim = {"cik": "0001769628", "form": "10-K"}
        results = agent.verify_claim(claim, sources, key_field="cik")
        assert results["form"]["match"] is False

    def test_sec_tools_exposes_evidence_helpers(self):
        tools = agent.sec_tools()
        assert tools["Evidence"] is agent.Evidence
        assert tools["verify_claim"] is agent.verify_claim


class TestGroundingGuidance:
    """Evidence-grounding guidance is part of the prompt."""

    def test_grounding_guidance_in_prompt(self):
        prompt = agent.build_prompt("Analyze CoreWeave filings", [])
        assert agent.GROUNDING_GUIDANCE in prompt
        assert "Do not invent values" in prompt
        assert "report it as unknown" in prompt
        for term in ("Retrieved:", "Derived:", "Inferred:", "Unsupported:"):
            assert term in prompt

    def test_verification_remains_optional(self):
        prompt = agent.build_prompt("Analyze SEC filings", [])
        assert "The model decides" in prompt
        assert "No specific research workflow is required" in prompt


class TestInvalidCountBugFix:
    """Invalid responses are already in history; they should not double-count."""

    def test_invalid_responses_dont_double_count_with_history(self):
        """After 3 invalid responses, history has 3 items; loop should not add 3 more."""
        # 3 invalid responses + FINAL. With the bug, history would have 3 + invalid_count=3 = 6.
        # With the fix, only len(history)=3 < max_steps=5, so FINAL is reached.
        responses = [
            "```python\nif True\n```",  # invalid, appended to history (len=1)
            "```python\nif True\n```",  # invalid, appended to history (len=2)
            "```python\nif True\n```",  # invalid, appended to history (len=3)
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("test", max_steps=5)
        # Should reach FINAL without exhausting steps
        assert result == "FINAL: done."
        assert mock_llm.call_count == 4

    def test_invalid_count_resets_on_valid_code(self):
        """Valid code extraction resets invalid_count so the counter tracks consecutives."""
        responses = [
            "```python\nif True\n```",     # invalid (count=1)
            "```python\nx = 1\n```",      # valid (count resets to 0)
            "```python\nif True\n```",     # invalid (count=1 again)
            "```python\nif True\n```",     # invalid (count=2)
            "```python\nif True\n```",     # invalid (count=3)
            "```python\nif True\n```",     # invalid (count=4)
            "```python\nif True\n```",     # invalid (count=5, triggers guard)
            "FINAL: done.",
        ]
        with patch('secedgar_agent.llm_query', side_effect=responses) as mock_llm:
            result = agent.research("test", max_steps=10)
        # 5 consecutive invalid responses trigger the guard
        assert result is None
        # FINAL is never reached
        assert mock_llm.call_count == 7
