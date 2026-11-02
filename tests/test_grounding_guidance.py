"""Tests for Experiment B: evidence-grounded prompting."""
from secedgar_agent import (
    GROUNDING_GUIDANCE,
    build_prompt,
    is_final_response,
)


def test_grounding_guidance_is_in_prompt():
    prompt = build_prompt("Analyze CoreWeave filings", [])
    assert GROUNDING_GUIDANCE in prompt


def test_missing_evidence_guidance():
    prompt = build_prompt("Find fiscal period starts", [])
    assert "Do not invent values" in prompt
    assert "report it as unknown" in prompt


def test_provenance_distinctions():
    prompt = build_prompt("Analyze SEC filings", [])
    for term in ("Retrieved:", "Derived:", "Inferred:", "Unsupported:"):
        assert term in prompt


def test_verification_remains_optional():
    prompt = build_prompt("Analyze SEC filings", [])
    assert "The model decides" in prompt
    assert "No specific research workflow is required" in prompt


def test_grounding_guidance_does_not_break_finalization():
    answer = (
        "FINAL:\n"
        "CoreWeave, Inc.\n"
        "\n"
        "fiscal_period_start,fiscal_period_end,filing_form,filing_date,accession_number,sec_filing_url\n"
        "unknown,2025-12-31,10-K,2026-03-02,0001769628-26-000104,https://www.sec.gov/...\n"
        "\n"
        "RESEARCH SUMMARY\n"
        "One 10-K filing retrieved (CoreWeave, Inc., 2026a).\n"
        "Fiscal period start is unknown from available metadata.\n"
        "\n"
        "REFERENCES\n"
        "[1] CoreWeave, Inc. (2026a). Form 10-K.\n"
    )
    assert is_final_response(answer) is True
