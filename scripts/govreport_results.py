#!/usr/bin/env python3
"""
Compute ROUGE-L, tokens/sec, and tokens/NFE for GovReport-LLaDA experiments.

Reads rank_0.jsonl files from results/GovReport-LLaDA/len{N}/{tag}/
and prints a comparison table. Works with partial results.

Usage:
    python scripts/govreport_results.py
    python scripts/govreport_results.py --results_dir results/GovReport-LLaDA/len1024
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR / "lm_eval_tasks" / "longbench"))
from rouge import Rouge  # noqa: E402 (available in venv)

_rouge = Rouge()


def rouge_l(prediction: str, references: list[str]) -> float:
    prediction = prediction.strip()
    if not prediction:
        return 0.0
    best = 0.0
    for ref in references:
        try:
            score = _rouge.get_scores([prediction], [ref], avg=True)["rouge-l"]["f"]
        except Exception:
            score = 0.0
        best = max(best, score)
    return best


def load_results(path: Path) -> dict:
    """Return dict with keys: n, rouge, tok_per_sec, tok_per_nfe, complete."""
    entries = []
    summary = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("summary"):
                summary = e
            elif "answer" in e:
                entries.append(e)

    if not entries:
        return None

    # ROUGE — requires target field; skip entries that lack it
    scored = [e for e in entries if e.get("target")]
    rouge_mean = (
        sum(
            rouge_l(e["answer"], e["target"])
            for e in tqdm(scored, desc=f"  rouge", unit="sample", leave=False)
        ) / len(scored)
        if scored else float("nan")
    )

    # Speed stats — prefer the summary entry (covers the full run); fall back to
    # the last answer entry which has cumulative totals.
    if summary:
        total_tokens = summary["total_tokens"]
        total_nfe    = summary["total_nfe"]
        total_time   = summary["total_time"]
    else:
        last = entries[-1]
        total_tokens = last["tokens"]
        total_nfe    = last["nfe"]
        total_time   = last["elapsed"]

    tok_per_sec = total_tokens / total_time if total_time > 0 else float("nan")
    tok_per_nfe = total_tokens / total_nfe  if total_nfe  > 0 else float("nan")

    return {
        "n":           len(entries),
        "rouge_l":     rouge_mean,
        "tok_per_sec": tok_per_sec,
        "tok_per_nfe": tok_per_nfe,
        "complete":    summary is not None,
    }


def find_runs(results_dir: Path) -> list[tuple[str, Path]]:
    """Return (tag, path) pairs sorted by tag name."""
    runs = []
    for jsonl in sorted(results_dir.rglob("rank_0.jsonl")):
        tag = jsonl.parent.relative_to(results_dir).as_posix()
        runs.append((tag, jsonl))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT_DIR / "results" / "GovReport-LLaDA" / "len512",
        help="directory containing per-tag subdirs with rank_0.jsonl files",
    )
    args = parser.parse_args()

    runs = find_runs(args.results_dir)
    if not runs:
        print(f"No rank_0.jsonl files found under {args.results_dir}")
        sys.exit(1)

    rows = []
    for tag, path in tqdm(runs, desc="scoring", unit="config"):
        r = load_results(path)
        if r is None:
            continue
        rows.append((tag, r))

    if not rows:
        print("No results to display.")
        sys.exit(1)

    # ── table ──────────────────────────────────────────────────────────────────
    col_tag  = max(len(tag) for tag, _ in rows)
    header = (
        f"{'config':<{col_tag}}  {'N':>5}  {'ROUGE-L':>8}  "
        f"{'tok/sec':>9}  {'tok/NFE':>8}  {'done':>5}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for tag, r in rows:
        done_str = "✓" if r["complete"] else f"{r['n']}"
        rouge_str = f"{r['rouge_l']:.4f}" if r["rouge_l"] == r["rouge_l"] else "  n/a "
        print(
            f"{tag:<{col_tag}}  {r['n']:>5}  {rouge_str:>8}  "
            f"{r['tok_per_sec']:>9.1f}  {r['tok_per_nfe']:>8.2f}  {done_str:>5}"
        )
    print(sep)
    print()


if __name__ == "__main__":
    main()
