"""Tests for the FINAL: evidence gate.

The gate must reject final answers whose cited accessions or material
numbers cannot be traced to evidence retrieved in the session — even when
the model submits them confidently.
"""
import secedgar_agent
from repl import REPL

ACCN = "0001769628-26-000104"


def make_repl() -> REPL:
    return REPL(namespace={
        "filings": [{"accessionNumber": ACCN, "reportDate": "2025-12-31",
                     "filingDate": "2026-03-02", "form": "10-K"}],
        "ni_obs": {"start": "2025-01-01", "end": "2025-12-31",
                   "val": -1167000000, "accn": ACCN, "form": "10-K"},
        "debt_current": {"end": "2025-12-31", "val": 6708000000, "accn": ACCN,
                         "form": "10-K"},
        "debt_noncurrent": {"end": "2025-12-31", "val": 14665000000, "accn": ACCN,
                            "form": "10-K"},
    })


def test_grounded_answer_passes():
    answer = (f"CoreWeave ratio for FY2025: -0.0546. Net income $(1,167,000,000), "
              f"total debt $21,373,000,000 (current $6.708 billion, noncurrent "
              f"$14.665 billion). Accession {ACCN}.")
    assert secedgar_agent.verify_final(answer, make_repl()) == []


def test_fabricated_accession_rejected():
    answer = f"Ratio -0.0546 per accession 0009999999-26-000999 (see also {ACCN})."
    issues = secedgar_agent.verify_final(answer, make_repl())
    assert any("0009999999-26-000999" in issue for issue in issues)


def test_fabricated_number_rejected():
    answer = f"Net income was $5,000,000,000. Accession {ACCN}."
    issues = secedgar_agent.verify_final(answer, make_repl())
    assert any("5,000,000,000" in issue for issue in issues)


def test_accession_accepted_in_url_form():
    answer = ("Ratio -0.0546. Source: "
              "https://www.sec.gov/Archives/edgar/data/1769628/000176962826000104/x.htm")
    assert secedgar_agent.verify_final(answer, make_repl()) == []


def test_ratio_grounded_via_two_level_derivation():
    # -0.0546 = -1167000000 / (6708000000 + 14665000000): sum then division.
    answer = f"Ratio: -0.0546. Accession {ACCN}."
    assert secedgar_agent.verify_final(answer, make_repl()) == []


def test_number_printed_in_output_is_grounded():
    """Values printed in observation output count as session evidence, even when
    the model did not persist them as variables (live-run regression)."""
    repl = REPL(namespace={})
    obs = ("CODE:\nprint(ni)\nOUTPUT:\n=== NET INCOME ===\n"
           f"  val: -1167000000\n  accn: {ACCN}\n"
           "=== DEBT ===\n  total_debt: 21373000000\n"
           "=== RATIO ===\n  ratio: -0.05460160014972161\n")
    answer = ("FINAL:\nCoreWeave, Inc.\nNet income to debt ratio: -0.0546.\n"
              "Net income $(1,167,000,000), total debt $21,373,000,000.\n"
              f"Accession {ACCN}.")
    assert secedgar_agent.verify_final(answer, repl, obs) == []


def test_research_rejects_confident_wrong_final(monkeypatch):
    """Milestone: a confidently submitted but unsupported FINAL is rejected
    and the research stops without releasing an answer."""
    wrong = (f"FINAL:\nCoreWeave, Inc.\nNet income to debt ratio: -0.0546.\n"
             f"Net income was $5,000,000,000 and total debt was $21,373,000,000.\n"
             f"Accession {ACCN}.\n\nRESEARCH SUMMARY\nConfident answer.\n\n"
             f"REFERENCES\nCoreWeave, Inc. (2026). Form 10-K. "
             f"U.S. Securities and Exchange Commission.")
    monkeypatch.setattr(secedgar_agent, "llm_query", lambda prompt: wrong)
    assert secedgar_agent.research("What is CoreWeave's net income to debt ratio?") is None


def test_research_releases_grounded_final(monkeypatch, capsys):
    repl_code = (
        f"filings = [dict(accessionNumber='{ACCN}', reportDate='2025-12-31',"
        f" filingDate='2026-03-02', form='10-K')]\n"
        f"ni_obs = dict(start='2025-01-01', end='2025-12-31',"
        f" val=-1167000000, accn='{ACCN}', form='10-K')\n"
        f"debt_current = dict(end='2025-12-31', val=6708000000, accn='{ACCN}')\n"
        f"debt_noncurrent = dict(end='2025-12-31', val=14665000000, accn='{ACCN}')"
    )
    good = (f"FINAL:\nCoreWeave, Inc.\nNet income to debt ratio: -0.0546.\n"
            f"Net income $(1,167,000,000), total debt $21,373,000,000.\n"
            f"Accession {ACCN}.\n\nRESEARCH SUMMARY\nGrounded.\n\n"
            f"REFERENCES\nCoreWeave, Inc. (2026). Form 10-K.")
    responses = iter([repl_code, good, good])
    monkeypatch.setattr(secedgar_agent, "llm_query", lambda prompt: next(responses))
    answer = secedgar_agent.research("What is CoreWeave's net income to debt ratio?")
    assert answer is not None and "-0.0546" in answer
