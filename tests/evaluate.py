"""Live evaluation harness: run golden questions through research() and score answers.

Unlike the offline pytest suite, this requires a configured local LLM endpoint
(src/config.toml) and network access to SEC EDGAR. Each golden item lists
deterministic checks against the final answer text.

Usage:
    python tests/evaluate.py                # run all golden questions
    python tests/evaluate.py --only 0       # run a single item by index
    python tests/evaluate.py --json report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from secedgar_agent import research

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def value_variants(value: float) -> set[str]:
    """Renderings of a number that may plausibly appear in an answer."""
    v = float(value)
    iv = int(v)
    variants = {
        repr(value),
        f"{iv:,}",
        f"${iv:,}",
        f"{v:.4f}",
        f"{v:.6f}",
    }
    if v < 0:
        # Negative amounts appear as -1,167,000,000, -$1,167,000,000, or the
        # accounting convention $(1,167,000,000); prose often uses the absolute
        # value with "net loss" wording.
        variants.add(f"-${abs(iv):,}")
        variants.add(f"$({abs(iv):,})")
        variants.add(f"({abs(iv):,})")
        variants.add(f"{abs(iv):,}")
    for divisor, suffix, word in ((1e9, "B", "billion"), (1e6, "M", "million"),
                                  (1e3, "K", "thousand")):
        if abs(v) >= divisor:
            scaled = v / divisor
            for precision in (3, 2, 1):
                rendered = f"{scaled:.{precision}f}"
                variants.add(f"{rendered}{suffix}")
                variants.add(f"{rendered} {word}")
                variants.add(f"${rendered} {word}")
                if scaled < 0:
                    variants.add(f"-${abs(scaled):.{precision}f} {word}")
                    variants.add(f"{abs(scaled):.{precision}f} {word}")
                    variants.add(f"${abs(scaled):.{precision}f} {word}")
    return {variant for variant in variants if variant}


def score_question(item: dict, answer: str | None) -> list[dict]:
    """Score a final answer against a golden item's deterministic checks."""
    if answer is None:
        return [{"name": "completed", "pass": False}]
    checks = [{"name": "completed", "pass": True}]
    for check in item.get("checks", []):
        kind = check["type"]
        if kind == "accession_cited":
            accession = check["value"]
            # Models cite accessions dashed (0001769628-26-000104) or in the
            # SEC URL form (000176962826000104); accept either.
            present = accession in answer or accession.replace("-", "") in answer
            checks.append({"name": f"accession_cited:{accession}", "pass": present})
        elif kind == "value_in_answer":
            present = any(variant in answer for variant in value_variants(check["value"]))
            checks.append({"name": f"value:{check.get('name', check['value'])}={check['value']}",
                           "pass": present})
        elif kind == "text_in_answer":
            needles = check["values"] if isinstance(check["values"], list) else [check["values"]]
            checks.append({"name": f"text:{'/'.join(needles)[:40]}",
                           "pass": any(needle in answer for needle in needles)})
        else:
            checks.append({"name": f"unknown_check_type:{kind}", "pass": False})
    return checks


def run_golden(golden_path: Path = GOLDEN_PATH, only: int | None = None,
               max_steps: int = 30) -> list[dict]:
    items = json.loads(golden_path.read_text())
    results = []
    for index, item in enumerate(items):
        if only is not None and index != only:
            continue
        started = time.time()
        answer = research(item["question"], max_steps=max_steps)
        elapsed = time.time() - started
        checks = score_question(item, answer)
        passed = all(check["pass"] for check in checks)
        results.append({
            "index": index,
            "question": item["question"],
            "passed": passed,
            "elapsed_sec": round(elapsed, 1),
            "checks": checks,
            "answer": answer,
        })
        print(f"[{index}] {'PASS' if passed else 'FAIL'} ({elapsed:.0f}s) "
              f"{item['question'][:70]}")
        for check in checks:
            print(f"    {'ok  ' if check['pass'] else 'FAIL'} {check['name']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, default=None,
                        help="Run only the golden item with this index")
    parser.add_argument("--json", default=None,
                        help="Write the full report (including answers) to this path")
    args = parser.parse_args()
    results = run_golden(only=args.only)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"Report written to {args.json}")
    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} golden questions passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
