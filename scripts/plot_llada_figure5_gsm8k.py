#!/usr/bin/env python3
"""Plot Figure 5 GSM8K results for LLaDA-Instruct.

This script expects the outputs produced by:
  scripts/reproduce_llada_figure5_gsm8k.sh

It reads the per-run lm-eval result JSON files and the cumulative rank_0.jsonl
logs, then produces the three Figure 5 panels:
  (a) accuracy vs threshold
  (b) inference steps vs threshold
  (c) accuracy vs average tokens per step

It also writes a CSV summary of the extracted points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class RunStats:
    label: str
    kind: str
    threshold: Optional[float]
    tokens_per_step: Optional[float]
    accuracy: Optional[float]
    avg_nfe: Optional[float]
    avg_tokens: Optional[float]
    throughput: Optional[float]
    samples: int
    result_path: Optional[Path]
    log_path: Optional[Path]


def find_latest_json(path: Path) -> Optional[Path]:
    candidates = sorted(path.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def load_accuracy(result_root: Path) -> Optional[float]:
    result_path = find_latest_json(result_root)
    if result_path is None:
        return None

    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results", {}).get("gsm8k", {})
    for key in (
        "exact_match,flexible-extract",
        "exact_match",
        "acc,none",
    ):
        if key in results:
            return float(results[key]) * 100.0

    numeric_values = [value for value in results.values() if isinstance(value, (int, float))]
    if numeric_values:
        return float(numeric_values[0]) * 100.0
    return None


def load_log_stats(log_root: Path) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    log_path = log_root / "rank_0.jsonl"
    if not log_path.exists():
        return None, None, None, 0

    last_entry = None
    sample_count = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(entry, dict) and entry.get("summary"):
                continue
            if isinstance(entry, dict) and "answer" in entry:
                sample_count += 1
                last_entry = entry
            elif isinstance(entry, str):
                sample_count += 1

    if not isinstance(last_entry, dict):
        return None, None, None, sample_count

    total_tokens = float(last_entry.get("tokens", 0.0))
    total_nfe = float(last_entry.get("nfe", 0.0))
    total_elapsed = float(last_entry.get("elapsed", 0.0))
    if total_nfe <= 0:
        return None, None, None, sample_count

    avg_nfe = total_nfe / max(sample_count, 1)
    avg_tokens = total_tokens / total_nfe
    throughput = total_tokens / total_elapsed if total_elapsed > 0 else None
    return avg_nfe, avg_tokens, throughput, sample_count


def load_run(label: str, kind: str, root: Path, gen_length: int, tag: str, threshold: Optional[float], tokens_per_step: Optional[float]) -> RunStats:
    result_root = root / "evals_results" / "Figure5" / "GSM8K" / f"len{gen_length}" / tag
    log_root = root / "results" / "Figure5" / "GSM8K" / f"len{gen_length}" / tag

    accuracy = load_accuracy(result_root)
    avg_nfe, avg_tokens, throughput, samples = load_log_stats(log_root)

    return RunStats(
        label=label,
        kind=kind,
        threshold=threshold,
        tokens_per_step=tokens_per_step,
        accuracy=accuracy,
        avg_nfe=avg_nfe,
        avg_tokens=avg_tokens,
        throughput=throughput,
        samples=samples,
        result_path=find_latest_json(result_root),
        log_path=log_root / "rank_0.jsonl",
    )


def pretty_threshold(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def build_runs(root: Path, gen_length: int, thresholds: Iterable[float], fixed_tokens: Iterable[int]) -> list[RunStats]:
    runs: list[RunStats] = []

    for threshold in thresholds:
        tag = f"threshold/t{str(threshold).replace('.', 'p')}"
        runs.append(
            load_run(
                label=f"t={pretty_threshold(threshold)}",
                kind="threshold",
                root=root,
                gen_length=gen_length,
                tag=tag,
                threshold=threshold,
                tokens_per_step=None,
            )
        )

    for tokens_per_step in fixed_tokens:
        tag = f"fixed/{tokens_per_step}tok-per-step"
        runs.append(
            load_run(
                label=f"{tokens_per_step} tok/step",
                kind="fixed",
                root=root,
                gen_length=gen_length,
                tag=tag,
                threshold=None,
                tokens_per_step=float(tokens_per_step),
            )
        )

    return runs


def style_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_selected(ax, x, y, text_offset=(10, 10)):
    ax.scatter([x], [y], s=88, facecolors="white", edgecolors="#c62828", linewidths=2.0, zorder=6)
    ax.annotate(
        "Selected",
        xy=(x, y),
        xytext=text_offset,
        textcoords="offset points",
        fontsize=9,
        weight="bold",
        color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Figure 5 GSM8K results.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root directory.")
    parser.add_argument("--gen-length", type=int, default=512, help="Generation length used for the runs.")
    parser.add_argument("--output", type=Path, default=Path("figures/figure5_gsm8k.png"), help="Output image path.")
    parser.add_argument("--csv", type=Path, default=Path("figures/figure5_gsm8k_summary.csv"), help="Output CSV summary path.")
    args_local = parser.parse_args()

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fixed_tokens = [1, 2, 4, 8]
    runs = build_runs(args.root, args.gen_length, thresholds, fixed_tokens)

    rows = []
    missing = []
    for run in runs:
        row = {
            "label": run.label,
            "kind": run.kind,
            "threshold": run.threshold,
            "tokens_per_step_target": run.tokens_per_step,
            "accuracy": run.accuracy,
            "avg_nfe": run.avg_nfe,
            "avg_tokens_per_step": run.avg_tokens,
            "throughput_tokens_per_sec": run.throughput,
            "samples": run.samples,
        }
        rows.append(row)

        if run.accuracy is None or run.avg_nfe is None or run.avg_tokens is None:
            missing.append(run.label)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    threshold_df = df[df["kind"] == "threshold"].sort_values("threshold")
    fixed_df = df[df["kind"] == "fixed"].sort_values("tokens_per_step_target")

    # Panel (a): accuracy vs threshold
    ax = axes[0]
    style_axes(ax, "(a) Accuracy vs Threshold", "Confidence Threshold", "GSM8K (5-shot) Accuracy (%)")
    ax.plot(
        threshold_df["threshold"],
        threshold_df["accuracy"],
        color="#c62828",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label="Ours",
    )
    for _, row in threshold_df.iterrows():
        ax.text(row["threshold"], row["accuracy"] + 0.25, f"{row['avg_tokens_per_step']:.1f}", fontsize=9, ha="center", color="#8b0000")
    for baseline_tokens in [2, 4, 8]:
        baseline_row = fixed_df[fixed_df["tokens_per_step_target"] == baseline_tokens].iloc[0]
        ax.axhline(
            baseline_row["accuracy"],
            color="#455a64",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label="Fixed-step baseline" if baseline_tokens == 2 else None,
        )
    selected_row = threshold_df[threshold_df["threshold"] == 0.9].iloc[0]
    annotate_selected(ax, selected_row["threshold"], selected_row["accuracy"], text_offset=(8, 12))
    ax.set_xlim(0.48, 1.02)

    # Panel (b): inference steps vs threshold
    ax = axes[1]
    style_axes(ax, "(b) Inference Steps vs Threshold", "Confidence Threshold", "Average Inference Steps")
    ax.plot(
        threshold_df["threshold"],
        threshold_df["avg_nfe"],
        color="#c62828",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label="Ours",
    )
    for baseline_tokens in [2, 4, 8]:
        baseline_row = fixed_df[fixed_df["tokens_per_step_target"] == baseline_tokens].iloc[0]
        ax.axhline(
            baseline_row["avg_nfe"],
            color="#455a64",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label="Fixed-step baseline" if baseline_tokens == 2 else None,
        )
    annotate_selected(ax, selected_row["threshold"], selected_row["avg_nfe"], text_offset=(8, 12))
    ax.set_xlim(0.48, 1.02)

    # Panel (c): accuracy vs average tokens per step
    ax = axes[2]
    style_axes(ax, "(c) Accuracy vs Average Tokens/Step", "Average Tokens per Step", "GSM8K (5-shot) Accuracy (%)")
    ax.plot(
        threshold_df["avg_tokens_per_step"],
        threshold_df["accuracy"],
        color="#c62828",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label="Ours",
    )
    ax.scatter(
        fixed_df["avg_tokens_per_step"],
        fixed_df["accuracy"],
        color="#1565c0",
        marker="^",
        s=72,
        label="Fixed-step baselines",
        zorder=5,
    )

    baseline_row = fixed_df[fixed_df["tokens_per_step_target"] == 1].iloc[0]
    ax.axvline(
        baseline_row["avg_tokens_per_step"],
        color="#616161",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="Non-parallel baseline",
    )
    for _, row in fixed_df.iterrows():
        ax.annotate(
            f"{int(row['tokens_per_step_target'])}",
            xy=(row["avg_tokens_per_step"], row["accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#0d47a1",
        )
    annotate_selected(ax, selected_row["avg_tokens_per_step"], selected_row["accuracy"], text_offset=(8, 12))

    handles, labels = axes[2].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)

    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), dpi=220, bbox_inches="tight")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.csv}")

    if missing:
        print("Missing metrics for:")
        for label in missing:
            print(f"  - {label}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())