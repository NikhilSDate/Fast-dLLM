"""Plot GSM8K accuracy and throughput vs prefix cache block size for LLaDA gen_length=256."""

import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")

def load_throughput(results_jsonl):
    with open(results_jsonl) as f:
        lines = f.readlines()
    summary = json.loads(lines[-1])
    assert summary.get("summary"), f"Last line is not summary in {results_jsonl}"
    return summary["total_tokens"] / summary["total_time"]

def load_accuracy(evals_dir):
    pattern = os.path.join(evals_dir, "GSAI-ML__LLaDA-8B-Instruct", "results_*.json")
    files = glob.glob(pattern)
    assert files, f"No results JSON found in {evals_dir}"
    with open(files[0]) as f:
        d = json.load(f)
    return d["results"]["gsm8k"]["exact_match,flexible-extract"] * 100

# --- collect per-block-size data ---
block_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
accuracies = []
throughputs = []

for bs in block_sizes:
    if bs == 32:
        # block32 lives in the top-level prefix-cache directory
        results_path = os.path.join(BASE, "results/GSM8K-LLaDA/len256/prefix-cache/rank_0.jsonl")
        evals_path   = os.path.join(BASE, "evals_results/GSM8K-LLaDA/len256/prefix-cache")
    else:
        results_path = os.path.join(BASE, f"results/GSM8K-LLaDA/len256/prefix-cache/block{bs}/rank_0.jsonl")
        evals_path   = os.path.join(BASE, f"evals_results/GSM8K-LLaDA/len256/prefix-cache/block{bs}")
    accuracies.append(load_accuracy(evals_path))
    throughputs.append(load_throughput(results_path))

# --- baseline (no cache) ---
baseline_tp  = load_throughput(os.path.join(BASE, "results/GSM8K-LLaDA/len256/baseline/rank_0.jsonl"))
baseline_acc = load_accuracy(os.path.join(BASE, "evals_results/GSM8K-LLaDA/len256/baseline"))

selected_idx = block_sizes.index(32)
selected_acc = accuracies[selected_idx]
selected_tp  = throughputs[selected_idx]
speedup = selected_tp / baseline_tp

# --- plot ---
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

x = np.arange(len(block_sizes))
labels = [str(bs) for bs in block_sizes]

blue   = "#4472C4"
orange = "#ED7D31"

l1, = ax1.plot(x, accuracies,  color=blue,   marker="o", linewidth=2, zorder=3)
l2, = ax2.plot(x, throughputs, color=orange,  marker="s", linewidth=2, zorder=3)

ax1.axhline(baseline_acc, color=blue,   linestyle="--", linewidth=1.5, alpha=0.7)
ax2.axhline(baseline_tp,  color=orange, linestyle="--", linewidth=1.5, alpha=0.7)

ax1.text(x[0] - 0.15, baseline_acc + 0.3, "No cache", color=blue,   fontsize=9, va="bottom")
ax2.text(x[0] - 0.15, baseline_tp  - 0.5, "No cache", color=orange, fontsize=9, va="top")

# highlight selected point (block32)
ax1.scatter([x[selected_idx]], [selected_acc], color=blue,   s=120, zorder=5)
ax2.scatter([x[selected_idx]], [selected_tp],  color=orange, s=120, zorder=5, marker="*")

ax1.annotate("Selected", xy=(x[selected_idx], selected_acc),
             xytext=(x[selected_idx] + 0.3, selected_acc + 0.8),
             fontsize=9, color="red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="red"))

# speedup annotation
mid_y_acc = (baseline_tp + selected_tp) / 2
ax2.annotate(
    f"{speedup:.1f}x Speedup",
    xy=(x[selected_idx] - 0.5, baseline_tp),
    xytext=(x[selected_idx] - 1.6, mid_y_acc),
    fontsize=11, color="red", fontweight="bold",
    arrowprops=None,
)
ax2.annotate("", xy=(x[selected_idx] - 0.5, baseline_tp),
             xytext=(x[selected_idx] - 0.5, selected_tp),
             arrowprops=dict(arrowstyle="<->", color="red", lw=2))

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Cache Block Size", fontsize=11)
ax1.set_ylabel("GSM8K (5-shot) Accuracy", color=blue, fontsize=11)
ax2.set_ylabel("Throughput (tokens/s)", color=orange, fontsize=11)
ax1.tick_params(axis="y", labelcolor=blue)
ax2.tick_params(axis="y", labelcolor=orange)

ax1.set_ylim(bottom=50)
ax2.set_ylim(bottom=4)

plt.title("LLaDA GSM8K: Accuracy & Throughput vs Cache Block Size (gen=256)", fontsize=10)
plt.tight_layout()

out = os.path.join(BASE, "figures", "cache_block_size.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
