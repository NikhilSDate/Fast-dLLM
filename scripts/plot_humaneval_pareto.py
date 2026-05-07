#!/usr/bin/env python3
"""
Scatter plots with Pareto frontiers for HumanEval batch-1 experiments.

Plots:
  1) Accuracy vs tokens/NFE   (factor and threshold strategies)
  2) Accuracy vs tokens/sec   (same)

Run from the repo root:
  python scripts/plot_humaneval_pareto.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Experiment definitions ──────────────────────────────────────────────────
#   rank0   : path to rank_0.jsonl (speed / NFE data)
#   evals_dir: directory under evals_results that contains the model subdir
#              with samples_humaneval_*.jsonl.cleaned  (latest is used)

EXPERIMENTS = [
    # factor sweeps
    dict(strategy="factor", param=0.4,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/f0p4-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/f0p4-cache-block32-batch1"),
    dict(strategy="factor", param=0.7,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/f0p7-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/f0p7-cache-block32-batch1"),
    dict(strategy="factor", param=1.0,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/cache-block32-batch1"),
    dict(strategy="factor", param=1.3,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/f1p3-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/f1p3-cache-block32-batch1"),
    dict(strategy="factor", param=1.6,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/f1p6-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/f1p6-cache-block32-batch1"),
    dict(strategy="factor", param=1.9,
         rank0="results/v2/HumanEval-FastDLLMv2/factor/f1p9-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/factor/f1p9-cache-block32-batch1"),
    # threshold sweeps
    dict(strategy="threshold", param=0.5,
         rank0="results/v2/HumanEval-FastDLLMv2/threshold/t0p5-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/threshold/t0p5-cache-block32-batch1"),
    dict(strategy="threshold", param=0.6,
         rank0="results/v2/HumanEval-FastDLLMv2/threshold/t0p6-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/threshold/t0p6-cache-block32-batch1"),
    dict(strategy="threshold", param=0.7,
         rank0="results/v2/HumanEval-FastDLLMv2/threshold/t0p7-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/threshold/t0p7-cache-block32-batch1"),
    dict(strategy="threshold", param=0.8,
         rank0="results/v2/HumanEval-FastDLLMv2/threshold/t0p8-cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/threshold/t0p8-cache-block32-batch1"),
    dict(strategy="threshold", param=0.9,
         rank0="results/v2/HumanEval-FastDLLMv2/threshold/cache-block32-batch1/rank_0.jsonl",
         evals_dir="evals_results/v2/HumanEval-FastDLLMv2/threshold/cache-block32-batch1"),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rank0_summary(path: Path) -> dict:
    summary = None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("summary"):
                summary = d
    if summary is None:
        raise ValueError(f"No summary line in {path}")
    return summary


def load_accuracy(evals_dir: Path) -> float:
    """Return pass@1 from the latest .cleaned file under evals_dir."""
    cleaned_files = sorted(evals_dir.rglob("*.jsonl.cleaned"))
    if not cleaned_files:
        raise FileNotFoundError(f"No .cleaned file under {evals_dir}")
    # latest by filename (ISO timestamp sorts lexicographically)
    cleaned = cleaned_files[-1]
    records = [json.loads(l) for l in cleaned.read_text().splitlines() if l.strip()]
    return sum(r["pass_at_1"] for r in records) / len(records)


def load_all() -> list[dict]:
    points = []
    for exp in EXPERIMENTS:
        rank0 = ROOT / exp["rank0"]
        evals_dir = ROOT / exp["evals_dir"]
        try:
            summary = load_rank0_summary(rank0)
            accuracy = load_accuracy(evals_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {exp['strategy']} {exp['param']}: {e}")
            continue
        tok_per_sec = summary["total_tokens"] / summary["total_time"]
        tok_per_nfe = summary["avg_tokens_per_nfe"]
        points.append({
            "strategy": exp["strategy"],
            "param":    exp["param"],
            "accuracy": accuracy,
            "tok_per_sec": tok_per_sec,
            "tok_per_nfe": tok_per_nfe,
        })
        print(f"  {exp['strategy']:9s} {exp['param']:.1f}  "
              f"acc={accuracy:.3f}  tok/s={tok_per_sec:.1f}  tok/NFE={tok_per_nfe:.2f}")
    return points


# ── Pareto frontier ───────────────────────────────────────────────────────────

def pareto_frontier(points: list[dict], x_key: str) -> list[dict]:
    """Return all points sorted by x (connects every operating point)."""
    return sorted(points, key=lambda p: p[x_key])


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"factor": "#1f77b4", "threshold": "#ff7f0e"}
MARKERS = {"factor": "o", "threshold": "s"}
LABELS  = {"factor": "Factor", "threshold": "Threshold"}


def plot_panel(ax, points: list[dict], x_key: str, xlabel: str):
    for strategy in ("factor", "threshold"):
        pts = [p for p in points if p["strategy"] == strategy]
        if not pts:
            continue
        xs = [p[x_key] for p in pts]
        ys = [p["accuracy"] * 100 for p in pts]
        ax.scatter(xs, ys,
                   color=COLORS[strategy], marker=MARKERS[strategy],
                   s=60, zorder=3, label=LABELS[strategy])
        # label each point with its parameter value
        for p in pts:
            ax.annotate(
                f"{p['param']}",
                (p[x_key], p["accuracy"] * 100),
                textcoords="offset points", xytext=(5, 4),
                fontsize=7.5, color=COLORS[strategy],
            )
        # Pareto frontier
        frontier = pareto_frontier(pts, x_key)
        if len(frontier) >= 2:
            fx = [p[x_key] for p in frontier]
            fy = [p["accuracy"] * 100 for p in frontier]
            ax.plot(fx, fy, color=COLORS[strategy],
                    linewidth=1.5, linestyle="--", zorder=2)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)


def main():
    print("Loading data...")
    points = load_all()
    if not points:
        print("No data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("HumanEval — Factor vs Threshold (batch size 1)", fontsize=13)

    plot_panel(axes[0], points, "tok_per_nfe", "Tokens / NFE")
    plot_panel(axes[1], points, "tok_per_sec", "Tokens / second")

    plt.tight_layout()
    out = ROOT / "figures" / "humaneval_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved → {out}")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=150)
    print(f"Saved → {png_out}")


if __name__ == "__main__":
    main()
