#!/usr/bin/env python3
"""
Generate a LaTeX table of GovReport-LLaDA results.

Usage:
    python scripts/govreport_latex.py
    python scripts/govreport_latex.py --results_dir results/GovReport-LLaDA/len1024
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from rouge import Rouge

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR / "lm_eval_tasks" / "longbench"))

_rouge = Rouge()

TAG_LABELS = {
    "baseline":            "Baseline",
    "dual-cache":          "Dual cache",
    "parallel":            "Parallel decoding",
    "dual-cache-parallel": "Dual cache + parallel",
}


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


def load_results(path: Path, desc: str) -> dict | None:
    entries, summary = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("summary"):
                summary = e
            elif "answer" in e:
                entries.append(e)

    if not entries:
        return None

    scored = [e for e in entries if e.get("target")]
    rouge_mean = (
        sum(
            rouge_l(e["answer"], e["target"])
            for e in tqdm(scored, desc=f"  {desc}", unit="sample", leave=False)
        ) / len(scored)
        if scored else float("nan")
    )

    if summary:
        total_tokens = summary["total_tokens"]
        total_nfe    = summary["total_nfe"]
        total_time   = summary["total_time"]
    else:
        last = entries[-1]
        total_tokens = last["tokens"]
        total_nfe    = last["nfe"]
        total_time   = last["elapsed"]

    return {
        "n":           len(entries),
        "rouge_l":     rouge_mean,
        "tok_per_sec": total_tokens / total_time if total_time > 0 else float("nan"),
        "tok_per_nfe": total_tokens / total_nfe  if total_nfe  > 0 else float("nan"),
        "complete":    summary is not None,
    }


def find_runs(results_dir: Path) -> list[tuple[str, Path]]:
    runs = []
    for jsonl in sorted(results_dir.rglob("rank_0.jsonl")):
        tag = jsonl.parent.relative_to(results_dir).as_posix()
        runs.append((tag, jsonl))
    return runs


def fmt_multiplier(value: float, baseline: float) -> str:
    if baseline == 0 or baseline != baseline:
        return ""
    return f"{value / baseline:.1f}\\times"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT_DIR / "results" / "GovReport-LLaDA" / "len512",
    )
    args = parser.parse_args()

    runs = find_runs(args.results_dir)
    if not runs:
        print(f"No rank_0.jsonl files found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for tag, path in tqdm(runs, desc="scoring", unit="config"):
        r = load_results(path, desc=tag)
        if r:
            rows.append((tag, r))

    if not rows:
        print("No results.", file=sys.stderr)
        sys.exit(1)

    baseline = next((r for tag, r in rows if tag == "baseline"), None)
    if baseline is None:
        baseline = rows[0][1]

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Config & ROUGE-L & Tok/sec & Tok/NFE \\",
        r"\midrule",
    ]

    for tag, r in rows:
        label = TAG_LABELS.get(tag, tag.replace("_", r"\_"))
        rouge_str = f"{r['rouge_l']:.4f}"

        if tag == "baseline":
            tps_str  = f"{r['tok_per_sec']:.1f}"
            tnfe_str = f"{r['tok_per_nfe']:.2f}"
        else:
            tps_mult  = fmt_multiplier(r["tok_per_sec"],  baseline["tok_per_sec"])
            tnfe_mult = fmt_multiplier(r["tok_per_nfe"],  baseline["tok_per_nfe"])
            tps_str   = f"{r['tok_per_sec']:.1f} (${tps_mult}$)"
            tnfe_str  = f"{r['tok_per_nfe']:.2f} (${tnfe_mult}$)"

        partial = "" if r["complete"] else r" \textsuperscript{*}"
        lines.append(f"{label}{partial} & {rouge_str} & {tps_str} & {tnfe_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{GovReport summarization results (LLaDA-8B-Instruct). "
        r"Tok/sec and Tok/NFE multipliers are relative to the baseline. "
        r"\textsuperscript{*}Partial results.}",
        r"\label{tab:govreport}",
        r"\end{table}",
    ]

    print("\n".join(lines))


if __name__ == "__main__":
    main()
