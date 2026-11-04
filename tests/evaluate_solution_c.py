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
    period_scores: dict[str, dict[str, str]] = field(default_factory=dict)
    # Maps accession_number -> {period_start: ..., filing_date: ...}
    # each value is one of correct/incorrect/unknown/unverified


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
        "0001769628-26-000222": "2026-01-01",   # Q1 2026 10-Q
        "0001769628-25-000062": "2025-07-01",   # Q3 2025 10-Q
        "0001769628-26-000104": "2025-01-01",   # FY2025 10-K
    }
    return {f["accessionNumber"]: mapping.get(f["accessionNumber"]) for f in filings}


def normalise_date(date_str: str) -> str:
    """Normalise a date string to YYYY-MM-DD, handling human-readable formats."""
    if not date_str or date_str.strip() in ("N/A", "unknown", "", None):
        return ""
    s = date_str.strip()
    # Already YYYY-MM-DD?
    m = re.match(r'^(\d{4}-\d{2}-\d{2})', s)
    if m:
        return m.group(1)
    # Human-readable: "April 1, 2026" or "January 01, 2025"
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # return as-is for scoring


def score_period_start(reported: str, expected: str | None) -> str:
    """Classify a reported period start against independent ground truth."""
    norm_reported = normalise_date(reported)
    if not norm_reported or norm_reported in ("N/A", "unknown"):
        return "unknown"
    if expected is None:
        return "unverified"
    if norm_reported == expected:
        return "correct"
    return "incorrect"


def parse_period_starts(answer: str | None) -> dict[str, str]:
    """Extract accession_number → fiscal_period_start pairs from a FINAL: answer.

    Handles CSV/Markdown-table rows that contain an accession number and at
    least one date (YYYY-MM-DD or human-readable like "April 1, 2026").
    Tries several heuristics:
      1. If a header line identifies a "Period Start" / "Fiscal Period Start"
         column, use its position to pull the start date.
      2. Otherwise, on rows with ≥2 dates the first is taken as the start
         and the last as the end.
      3. Falls back to the first date on rows with exactly one date.
    """
    starts: dict[str, str] = {}
    if not answer:
        return starts

    accn_pattern = r'(000\d{7}-\d{2}-\d{6})'
    ymd_pattern = r'(\d{4}-\d{2}-\d{2})'
    # Also match human-readable dates like "April 1, 2026"
    hr_date_pattern = r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})'
    header_match: dict[str, int] | None = None  # col_name → zero-based index

    for line in answer.split("\n"):
        # Look for a header row that names the period-start column
        if header_match is None:
            lower = line.lower()
            for token, name in [
                ("period start", "period_start"),
                ("fiscal period start", "period_start"),
                ("fiscal_period_start", "period_start"),
            ]:
                if token in lower:
                    cols = re.split(r'[|,\t]+', line.strip().strip("|"))
                    header_match = {}
                    for i, c in enumerate(cols):
                        header_match[c.strip().lower()] = i
                    break

        accns = re.findall(accn_pattern, line)
        ymd_dates = re.findall(ymd_pattern, line)
        hr_dates = re.findall(hr_date_pattern, line)
        all_dates = ymd_dates + hr_dates
        if not accns or not all_dates:
            continue
        accn = accns[0]

        if header_match is not None:
            # Try to find the start-date column by header name
            for col_name, idx in header_match.items():
                if "start" in col_name:
                    cols = re.split(r'[|,\t]+', line.strip().strip("|"))
                    if idx < len(cols):
                        val = cols[idx].strip().replace('"', '')
                        # Try YYYY-MM-DD first
                        m = re.search(ymd_pattern, val)
                        if m:
                            starts[accn] = m.group(1)
                        else:
                            # Try human-readable
                            hm = re.search(hr_date_pattern, val)
                            if hm:
                                starts[accn] = hm.group(1)
                            else:
                                starts[accn] = val
                        break
            else:
                # Header found but no start column – use first date
                starts[accn] = all_dates[0]
        elif len(all_dates) >= 2:
            starts[accn] = all_dates[0]
        else:
            starts[accn] = all_dates[0]

    return starts


def parse_filing_dates(answer: str | None) -> dict[str, str]:
    """Extract accession_number → filing_date pairs from a FINAL: answer."""
    dates: dict[str, str] = {}
    if not answer:
        return dates
    accn_pattern = r'(000\d{7}-\d{2}-\d{6})'
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    header_match: dict[str, int] | None = None

    for line in answer.split("\n"):
        if header_match is None:
            lower = line.lower()
            for token, name in [
                ("filing date", "filing_date"),
                ("filing_date", "filing_date"),
            ]:
                if token in lower:
                    cols = re.split(r'[|,\t]+', line.strip().strip("|"))
                    header_match = {}
                    for i, c in enumerate(cols):
                        header_match[c.strip().lower()] = i
                    break

        accns = re.findall(accn_pattern, line)
        all_dates = re.findall(date_pattern, line)
        if not accns or not all_dates:
            continue
        accn = accns[0]

        if header_match is not None:
            for col_name, idx in header_match.items():
                if "filing" in col_name or "date" in col_name:
                    cols = re.split(r'[|,\t]+', line.strip().strip("|"))
                    if idx < len(cols):
                        val = cols[idx].strip().replace('"', '')
                        m = re.search(date_pattern, val)
                        if m:
                            dates[accn] = m.group(1)
                            break
            else:
                dates[accn] = all_dates[0]
        elif len(all_dates) >= 2:
            dates[accn] = all_dates[0]
        else:
            dates[accn] = all_dates[0]

    return dates


def parse_accessions_in_answer(answer: str | None) -> set[str]:
    """Extract all accession numbers mentioned in the answer."""
    if not answer:
        return set()
    accn_pattern = r'(000\d{7}-\d{2}-\d{6})'
    return set(re.findall(accn_pattern, answer))


def score_answer(answer: str | None, expected_starts: dict[str, str | None],
                 expected_filing_dates: dict[str, str] | None = None,
                 trace_content: str | None = None) -> dict[str, dict[str, str]]:
    """Score period-start and (optionally) filing-date accuracy.

    Looks for CSV/table data in the FINAL: answer first, then falls back to
    the full trace content (where the model may have output the table before
    the FINAL: marker).

    Returns a dict mapping accession_number → {period_start: ..., filing_date: ...}
    where each value is one of correct/incorrect/unknown/unverified.
    """
    reported_starts = parse_period_starts(answer)
    if not reported_starts and trace_content:
        reported_starts = parse_period_starts(trace_content)
    reported_dates = parse_filing_dates(answer) if expected_filing_dates else {}
    if not reported_dates and trace_content and expected_filing_dates:
        reported_dates = parse_filing_dates(trace_content)
    scores: dict[str, dict[str, str]] = {}
    for accn, exp_start in expected_starts.items():
        entry: dict[str, str] = {}
        rep_start = reported_starts.get(accn, "")
        entry["period_start"] = score_period_start(rep_start, exp_start)
        if expected_filing_dates:
            exp_fd = expected_filing_dates.get(accn)
            rep_fd = reported_dates.get(accn, "")
            if rep_fd:
                entry["filing_date"] = "correct" if rep_fd == exp_fd else "incorrect"
            else:
                entry["filing_date"] = "unknown"
        scores[accn] = entry
    return scores


def run_trial(variant: str, trial_index: int, question: str, expected_starts: dict[str, str | None],
              expected_filing_dates: dict[str, str] | None = None, max_steps: int = 30) -> TrialResult:
    """Run one trial under the given variant and save a trace."""
    original_query = secedgar_agent.llm_query
    call_count = [0]
    draft_used = [False]
    issues_detected = [0]
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
    scores = score_answer(answer, expected_starts, expected_filing_dates=expected_filing_dates,
                          trace_content="\n\n".join(trace_lines))
    issues_d = issues_detected[0]
    # Count corrections: issues detected minus issues still present in final answer
    incorrect_count = sum(
        1 for accn_scores in scores.values()
        for s in accn_scores.values() if s == "incorrect"
    )
    issues_c = max(0, issues_d - incorrect_count)

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

        # Flatten nested scores (period_start + optional filing_date)
        all_values = [s for accn_scores in all_scores.values() for s in accn_scores.values()]
        correct_count = sum(1 for s in all_values if s == "correct")
        incorrect_count = sum(1 for s in all_values if s == "incorrect")
        unknown_count = sum(1 for s in all_values if s == "unknown")
        unverified_count = sum(1 for s in all_values if s == "unverified")
        total_scored = correct_count + incorrect_count + unknown_count + unverified_count

        avg_steps = sum(t.steps for t in trials) / n
        avg_time = sum(t.elapsed_sec for t in trials) / n
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

    fixture = load_fixture()
    expected_starts = expected_period_starts(fixture)
    expected_filing_dates = {f["accessionNumber"]: f["filingDate"] for f in fixture}
    print("Expected period starts:\n")
    for accn, start in sorted(expected_starts.items()):
        print(f"  {accn}  →  {start or '(no independent start established)'}")
    print()

    print(f"Running {n_trials} trials per variant (alternating B/C)...\n")

    results: list[TrialResult] = []

    # Alternate B and C trials to reduce order effects
    interleaved = []
    for i in range(n_trials):
        interleaved.append(("B", i))
        interleaved.append(("C", i))

    for variant, idx in interleaved:
        label = f"{variant} trial {idx+1}/{n_trials}"
        print(f"── {label} ──", flush=True)
        r = run_trial(variant, idx, question, expected_starts, expected_filing_dates=expected_filing_dates)
        status = "OK" if r.success else "FAIL"
        scores_summary = "/".join(
            f"{s}:{','.join(f'{k}={v}' for k, v in sorted(v.items()))}"
            for s, v in sorted(r.period_scores.items())
        )
        print(f"  {status}  {r.steps} steps  {r.elapsed_sec}s  "
              f"draft={'y' if r.draft_used else 'n'}  "
              f"issues_det={r.issues_detected} corr={r.issues_corrected}")
        print(f"  scores: {scores_summary}", flush=True)
        results.append(r)

    print("\n" + summarize(results))


if __name__ == "__main__":
    main()
