"""Evaluation harness for Solution C vs B.

Measures evidence-grounded accuracy, not date avoidance.
For each trial:
  - Completes with FINAL:?
  - Uses DRAFT: verification?
  - Reports correct/incorrect/unknown/unverified period starts
  - How many verification issues were detected and corrected?
  - Steps and wall-clock time

Preserves full traces for post-hoc inspection.
"""
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import secedgar_agent
from secedgar_agent import research

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "coreweave_filings.json"
TRACES_DIR = Path(__file__).parent / "traces"


@dataclass
class TrialResult:
    variant: str          # "B" or "C"
    trial_index: int
    question: str
    success: bool
    steps: int
    elapsed_sec: float
    answer: str | None
    draft_used: bool
    issues_detected: int
    issues_corrected: int
    period_scores: dict[str, str] = field(default_factory=dict)
    # Maps accession_number -> "correct" | "incorrect" | "unknown" | "unverified"


def load_fixture() -> list[dict]:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def expected_period_starts(filings: list[dict]) -> dict[str, str | None]:
    """Independently established expected period starts.

    Derived from SEC filing text and XBRL duration observations for
    CoreWeave, not from prior model outputs.

    CoreWeave reports calendar-quarter fiscal periods:
      Q3 2025 (10-Q, reportDate 2025-09-30)  → start 2025-07-01
      FY2025  (10-K, reportDate 2025-12-31)  → start 2025-01-01
      Q1 2026 (10-Q, reportDate 2026-03-31)  → start 2026-01-01
      Q2 2026 (10-Q, reportDate 2026-06-30)  → start 2026-04-01
    """
    mapping: dict[str, str | None] = {
        "0001769628-26-000366": "2026-04-01",   # Q2 2026 10-Q
        "0001769628-26-000201": "2026-01-01",   # Q1 2026 10-Q
        "0001769628-25-000452": "2025-07-01",   # Q3 2025 10-Q
        "0001769628-26-000104": "2025-01-01",   # FY2025 10-K
    }
    return {f["accessionNumber"]: mapping.get(f["accessionNumber"]) for f in filings}


def score_period_start(reported: str, expected: str | None) -> str:
    """Classify a reported period start against independent ground truth."""
    if reported in ("N/A", "unknown", "", None):
        return "unknown"
    if expected is None:
        return "unverified"
    if reported.strip() == expected:
        return "correct"
    return "incorrect"


def parse_period_starts(answer: str | None) -> dict[str, str]:
    """Extract accession_number → fiscal_period_start pairs from a FINAL: answer."""
    starts: dict[str, str] = {}
    if not answer:
        return starts
    # Match CSV-like rows: accession_number, fiscal_period_end, ... fiscal_period_start
    # Try multiple column orderings
    accn_pattern = r'(000\d{10}-\d{2}-\d{6})'
    date_pattern = r'(\d{4}-\d{2}-\d{2})'

    for line in answer.split("\n"):
        accns = re.findall(accn_pattern, line)
        dates = re.findall(date_pattern, line)
        if accns and dates:
            # Heuristic: last date in line is likely period_end, first is start
            # But we need to match by column position from header
            starts[accns[0]] = dates[0] if dates else ""
    return starts


def score_answer(answer: str | None, expected_starts: dict[str, str | None]) -> dict[str, str]:
    """Score each reported period start against ground truth."""
    reported = parse_period_starts(answer)
    scores: dict[str, str] = {}
    for accn, exp_start in expected_starts.items():
        rep_start = reported.get(accn, "")
        scores[accn] = score_period_start(rep_start, exp_start)
    return scores


def run_trial(variant: str, trial_index: int, question: str, max_steps: int = 30) -> TrialResult:
    """Run one trial under the given variant and save a trace."""
    original_query = secedgar_agent.llm_query
    call_count = [0]
    draft_used = [False]
    issues_detected = [0]
    issues_corrected = [0]
    trace_lines: list[str] = []

    def counting_query(prompt: str) -> str:
        call_count[0] += 1
        trace_lines.append(f"--- LLM call {call_count[0]} (variant {variant}) ---")
        trace_lines.append(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
        if "DRAFT to verify:" in prompt:
            draft_used[0] = True
        result = original_query(prompt)
        trace_lines.append(f"--- Response {call_count[0]} ---")
        trace_lines.append(result[:3000] + ("..." if len(result) > 3000 else ""))
        # Detect verification feedback in the prompt (indicates issues were detected)
        if "VERIFICATION ISSUES:" in prompt:
            issues_detected[0] += 1
        elif "VERIFICATION PASSED" in prompt:
            pass  # passed, no issues to count
        return result

    secedgar_agent.llm_query = counting_query

    t0 = time.monotonic()
    try:
        answer = research(question, max_steps=max_steps)
    finally:
        secedgar_agent.llm_query = original_query
    elapsed = time.monotonic() - t0

    # Save trace
    TRACES_DIR.mkdir(exist_ok=True)
    trace_path = TRACES_DIR / f"{variant.lower()}_trial{trial_index+1:02d}.txt"
    with open(trace_path, "w") as f:
        f.write(f"Variant: {variant}\n")
        f.write(f"Trial: {trial_index+1}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Success: {answer is not None and 'FINAL:' in (answer or '')}\n")
        f.write(f"Steps: {call_count[0]}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
        f.write(f"DRAFT used: {draft_used[0]}\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n\n".join(trace_lines))
        if answer:
            f.write("\n\n=== FINAL ANSWER ===\n")
            f.write(answer)

    steps = call_count[0]
    success = bool(answer and "FINAL:" in answer)
    scores = score_answer(answer, expected_starts)
    issues_d = issues_detected[0]
    # Count corrections: issues detected minus issues still present in final answer
    issues_c = max(0, issues_d - sum(1 for s in scores.values() if s == "incorrect"))

    return TrialResult(
        variant=variant,
        trial_index=trial_index,
        question=question,
        success=success,
        steps=steps,
        elapsed_sec=round(elapsed, 1),
        answer=answer,
        draft_used=draft_used[0],
        issues_detected=issues_d,
        issues_corrected=issues_c,
        period_scores=scores,
    )


def summarize(results: list[TrialResult]) -> str:
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

        # Score aggregation
        all_scores: dict[str, list[str]] = {}
        for t in trials:
            for accn, score in t.period_scores.items():
                all_scores.setdefault(accn, []).append(score)

        correct_count = sum(1 for scores in all_scores.values() for s in scores if s == "correct")
        incorrect_count = sum(1 for scores in all_scores.values() for s in scores if s == "incorrect")
        unknown_count = sum(1 for scores in all_scores.values() for s in scores if s == "unknown")
        unverified_count = sum(1 for scores in all_scores.values() for s in scores if s == "unverified")
        total_scored = correct_count + incorrect_count + unknown_count + unverified_count

        avg_steps = sum(t.steps for t in trials) / n
        avg_time = sum(t.elapsed_sec for t in trials) / n
        successful_time = (
            sum(t.elapsed_sec for t in successful) / len(successful)
            if successful else float("nan")
        )
        total_issues_detected = sum(t.issues_detected for t in trials)
        total_issues_corrected = sum(t.issues_corrected for t in trials)

        lines.append(f"\nVariant {variant} (n={n})")
        lines.append(f"  Completion rate: {len(successful)}/{n} ({100*len(successful)/n:.0f}%)")
        lines.append(f"  DRAFT adoption:  {len(with_draft)}/{n} ({100*len(with_draft)/n:.0f}%)")
        lines.append(f"  Avg steps:       {avg_steps:.1f}")
        lines.append(f"  Avg time:        {avg_time:.1f}s")
        if successful:
            lines.append(f"  Median time to correct answer: {sorted(t.elapsed_sec for t in successful)[len(successful)//2]:.1f}s")
        lines.append(f"  Issues detected:   {total_issues_detected}")
        lines.append(f"  Issues corrected:  {total_issues_corrected}")
        lines.append(f"  Period-start accuracy ({total_scored} observations):")
        lines.append(f"    correct:   {correct_count} ({100*correct_count/total_scored:.0f}%)")
        lines.append(f"    incorrect: {incorrect_count} ({100*incorrect_count/total_scored:.0f}%)")
        lines.append(f"    unknown:   {unknown_count} ({100*unknown_count/total_scored:.0f}%)")
        lines.append(f"    unverified:{unverified_count} ({100*unverified_count/total_scored:.0f}%)")

    lines.append(f"\nTraces saved to: {TRACES_DIR}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    question = (
        "What are the fiscal-period starts and ends for CoreWeave's "
        "four most recent SEC filings (10-K and 10-Q)? "
        "Include accession numbers, filing dates, and SEC URLs."
    )
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    expected_starts = expected_period_starts(load_fixture())
    print(f"Expected period starts:\n")
    for accn, start in sorted(expected_starts.items()):
        print(f"  {accn}  →  {start or '(no independent start established)'}")
    print()

    print(f"Running {n_trials} trials per variant (alternating B/C)...\n")

    results: list[TrialResult] = []

    # Alternate B and C trials to reduce order effects
    variants = []
    for i in range(n_trials):
        variants.append(("B", i))
    for i in range(n_trials):
        variants.append(("C", i))
    # Interleave
    interleaved = []
    for i in range(n_trials):
        interleaved.append(variants[i * 2])     # B trial i
        interleaved.append(variants[i * 2 + 1]) # C trial i

    for variant, idx in interleaved:
        label = f"{variant} trial {idx+1}/{n_trials}"
        print(f"── {label} ──", flush=True)
        r = run_trial(variant, idx, question)
        status = "OK" if r.success else "FAIL"
        scores_summary = "/".join(
            f"{s}:{v}" for s, v in sorted(r.period_scores.items())
        )
        print(f"  {status}  {r.steps} steps  {r.elapsed_sec}s  "
              f"draft={'y' if r.draft_used else 'n'}  "
              f"issues_det={r.issues_detected} corr={r.issues_corrected}")
        print(f"  scores: {scores_summary}", flush=True)
        results.append(r)

    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
