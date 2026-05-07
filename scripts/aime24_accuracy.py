#!/usr/bin/env python3
"""
Compute AIME 2024 accuracy from lm-eval samples JSONL files.

Scans evals_results/AIME24-LLaDA for all samples_aime24_*.jsonl files,
prints per-problem results, and summarises accuracy for each run.

Usage:
    python scripts/aime24_accuracy.py
    python scripts/aime24_accuracy.py --results_dir evals_results/AIME24-LLaDA
"""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{} in text, or None."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    if i >= len(text):
        return None
    if text[i] == " ":
        # \boxed <content>$
        rest = text[i + 1:]
        return rest.split("$")[0].strip()
    if text[i] != "{":
        return None
    depth, start = 0, i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return None


def extract_last_int(text: str) -> str | None:
    """Fallback: return the last integer found in text."""
    matches = re.findall(r"-?[0-9]+", text)
    return matches[-1] if matches else None


def predicted_answer(generation: str) -> str:
    """Extract the model's answer from a generation string."""
    boxed = extract_boxed(generation)
    if boxed is not None:
        # strip surrounding whitespace / dollar signs
        return boxed.strip().strip("$").strip()
    return extract_last_int(generation) or ""


def process_file(path: Path) -> dict:
    samples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    results = []
    for s in samples:
        doc = s["doc"]
        gen = s["filtered_resps"][0] if s.get("filtered_resps") else (
              s["resps"][0][0] if s.get("resps") else "")
        pred = predicted_answer(gen)
        correct = bool(s.get("exact_match", 0))
        results.append({
            "id":      doc.get("ID", s["doc_id"]),
            "target":  s["target"],
            "pred":    pred,
            "correct": correct,
        })
    return results


def print_run(label: str, results: list[dict]) -> None:
    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Accuracy: {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"{'=' * 60}")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark}  {r['id']:15s}  target={r['target']:4s}  pred={r['pred']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path,
                        default=ROOT / "evals_results" / "AIME24-LLaDA")
    args = parser.parse_args()

    sample_files = sorted(args.results_dir.rglob("samples_aime24_*.jsonl"))
    if not sample_files:
        print(f"No samples_aime24_*.jsonl files found under {args.results_dir}")
        return

    for path in sample_files:
        # Build a short label from the path relative to results_dir
        rel = path.relative_to(args.results_dir)
        label = str(rel.parent)
        results = process_file(path)
        print_run(label, results)


if __name__ == "__main__":
    main()
