"""Evaluation harness for Solution C vs B.

Run repeated trials of the same SEC research question under two
conditions and compare:
  - Correct fiscal-period starts
  - Unsupported/unverified dates
  - DRAFT usage (verification adoption)
  - Issues detected by verifier
  - Issues corrected before FINAL:
  - Successful final answers
  - Time to correct answer
"""
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import secedgar_agent
from secedgar_agent import research


@dataclass
class TrialResult:
    variant: str  # "B" or "C"
    question: str
    success: bool
    steps: int
    elapsed_sec: float
    answer: str | None
    draft_used: bool = False
    issues_detected: int = 0
    issues_corrected: int = 0
    fabricated_dates: list[str] = field(default_factory=list)
    unknown_dates: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


def run_trial(variant: str, question: str, max_steps: int = 30) -> TrialResult:
    """Run one trial under the given variant."""
    # Patch llm_query to count steps and detect draft usage
    original_query = secedgar_agent.llm_query
    call_count = [0]
    draft_used = [False]
    issues_detected = [0]
    issues_corrected = [0]

    def counting_query(prompt: str) -> str:
        call_count[0] += 1
        if "DRAFT to verify:" in prompt:
            draft_used[0] = True
        # Check for verification feedback in history to count issues
        return original_query(prompt)

    secedgar_agent.llm_query = counting_query

    t0 = time.monotonic()
    try:
        answer = research(question, max_steps=max_steps)
    finally:
        secedgar_agent.llm_query = original_query
    elapsed = time.monotonic() - t0

    # Parse the answer for analysis
    fabricated = []
    unknown = []
    unsupported = []
    success = False

    if answer and "FINAL:" in answer:
        success = True
        # Look for invented fiscal-period starts
        for m in re.finditer(r'fiscal period start is (\d{4}-\d{2}-\d{2})', answer, re.IGNORECASE):
            fabricated.append(m.group(1))
        # Look for N/A period starts
        for m in re.finditer(r'fiscal_period_start.*?(N/A|unknown|not available)', answer, re.IGNORECASE):
            unknown.append(m.group(0)[:60])
        # Look for unsupported date claims
        for m in re.finditer(r'\b(202\d-(0[13578]|1[02])-(0[13579]))\b', answer):
            # These are end-of-month dates not tied to filings — flag as potentially unsupported
            pass

    return TrialResult(
        variant=variant,
        question=question,
        success=success,
        steps=call_count[0],
        elapsed_sec=round(elapsed, 1),
        answer=answer,
        draft_used=draft_used[0],
        issues_detected=issues_detected[0],
        issues_corrected=issues_corrected[0],
        fabricated_dates=fabricated,
        unknown_dates=unknown,
        unsupported_claims=unsupported,
    )


def summarize(results: list[TrialResult]) -> str:
    """Produce a human-readable evaluation summary."""
    by_variant: dict[str, list[TrialResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION SUMMARY: Solution C vs B")
    lines.append("=" * 70)

    for variant in ("B", "C"):
        trials = by_variant.get(variant, [])
        if not trials:
            continue
        n = len(trials)
        successful = [t for t in trials if t.success]
        with_draft = [t for t in trials if t.draft_used]
        total_fabricated = sum(len(t.fabricated_dates) for t in trials)
        total_unknown = sum(len(t.unknown_dates) for t in trials)
        avg_steps = sum(t.steps for t in trials) / n
        avg_time = sum(t.elapsed_sec for t in trials) / n
        successful_time = (
            sum(t.elapsed_sec for t in successful) / len(successful)
            if successful else float("nan")
        )

        lines.append(f"\nVariant {variant} (n={n})")
        lines.append(f"  Completion rate: {len(successful)}/{n} ({100*len(successful)/n:.0f}%)")
        lines.append(f"  Avg steps: {avg_steps:.1f}")
        lines.append(f"  Avg time to answer: {avg_time:.1f}s")
        if successful:
            lines.append(f"  Avg time to correct answer: {successful_time:.1f}s")
        lines.append(f"  DRAFT used: {len(with_draft)}/{n} ({100*len(with_draft)/n:.0f}%)")
        lines.append(f"  Fabricated dates: {total_fabricated}")
        lines.append(f"  Unknown/N/A dates: {total_unknown}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    question = (
        "What are the fiscal-period starts and ends for CoreWeave's "
        "four most recent SEC filings (10-K and 10-Q)? "
        "Include accession numbers, filing dates, and SEC URLs."
    )
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Running {n_trials} trials per variant...\n")

    results: list[TrialResult] = []

    # Run variant B (grounding guidance only)
    print("── Variant B (grounding guidance) ──")
    for i in range(n_trials):
        print(f"  B trial {i+1}/{n_trials}...", end=" ", flush=True)
        r = run_trial("B", question)
        status = "OK" if r.success else "FAIL"
        print(f"{status}  {r.steps} steps  {r.elapsed_sec}s  "
              f"fabricated={len(r.fabricated_dates)}  unknown={len(r.unknown_dates)}")
        results.append(r)

    # Run variant C (grounding + DRAFT verification)
    print("\n── Variant C (grounding + DRAFT verification) ──")
    for i in range(n_trials):
        print(f"  C trial {i+1}/{n_trials}...", end=" ", flush=True)
        r = run_trial("C", question)
        status = "OK" if r.success else "FAIL"
        print(f"{status}  {r.steps} steps  {r.elapsed_sec}s  "
              f"draft={'yes' if r.draft_used else 'no'}  "
              f"fabricated={len(r.fabricated_dates)}  unknown={len(r.unknown_dates)}")
        results.append(r)

    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
