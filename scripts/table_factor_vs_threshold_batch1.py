#!/usr/bin/env python3
"""
Auto-generate figures/table_factor_vs_threshold_batch1.tex from experiment results.

Data sources:
  GSM8K accuracy  : evals_results/v2/GSM8K-FastDLLMv2/{strategy}/*/results_*.json
                    metric: exact_match,flexible-extract
  HumanEval acc.  : evals_results/v2/HumanEval-FastDLLMv2/{strategy}/*/*.jsonl[.cleaned]
                    metric: pass@1 via llada/postprocess_code logic
  Speed stats     : results/v2/{Dataset}-FastDLLMv2/{strategy}/rank_0.jsonl (summary line)

Usage:
    python scripts/table_factor_vs_threshold_batch1.py
    python scripts/table_factor_vs_threshold_batch1.py --no-write   # print only
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── HumanEval scoring (mirrors llada/postprocess_code.py) ─────────────────────

sys.path.insert(0, str(ROOT_DIR / "llada"))
import evaluate as hf_evaluate
from sanitize import sanitize  # noqa: E402

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
_pass_at_k = hf_evaluate.load("code_eval")


def _humaneval_pass_at_1_from_samples(samples_path: Path) -> float:
    """Compute pass@1 from a lm-eval samples JSONL, caching result in .cleaned."""
    cleaned_path = Path(str(samples_path) + ".cleaned")

    if cleaned_path.exists():
        with open(cleaned_path) as f:
            data = [json.loads(l) for l in f]
        return sum(d["pass_at_1"] for d in data) / len(data)

    with open(samples_path) as f:
        data = [json.loads(l) for l in f]

    references = [s["target"] for s in data]
    predictions = [
        [
            sanitize(
                s["doc"]["prompt"] + "\n" + s["resps"][0][0]
                    .split("```python\n", 1)[-1]
                    .split("```")[0],
                s["doc"]["entry_point"],
            )
        ]
        for s in data
    ]

    pass_at_1s = [
        _pass_at_k.compute(references=[ref], predictions=[pred], k=[1])[0]["pass@1"]
        for ref, pred in zip(references, predictions)
    ]

    # cache
    with open(cleaned_path, "w") as f:
        for s, pred, p in zip(data, predictions, pass_at_1s):
            f.write(json.dumps({"task_id": s["doc"]["task_id"],
                                "completion": pred, "pass_at_1": p}) + "\n")

    return sum(pass_at_1s) / len(pass_at_1s)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(ROOT_DIR / pattern)))
    return Path(files[-1]) if files else None


def load_gsm8k_accuracy(strategy: str) -> float | None:
    path = _latest(f"evals_results/v2/GSM8K-FastDLLMv2/{strategy}/*/results_*.json")
    if path is None:
        return None
    with open(path) as f:
        d = json.load(f)
    return d["results"]["gsm8k"].get("exact_match,flexible-extract")


def load_humaneval_accuracy(strategy: str) -> float | None:
    path = _latest(f"evals_results/v2/HumanEval-FastDLLMv2/{strategy}/*/*.jsonl")
    if path is None:
        return None
    return _humaneval_pass_at_1_from_samples(path)


def load_speed(dataset: str, strategy: str) -> dict | None:
    """Return dict with tok_per_sec and tok_per_nfe from rank_0.jsonl summary."""
    path = ROOT_DIR / "results" / "v2" / f"{dataset}-FastDLLMv2" / strategy / "rank_0.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("summary"):
                t = e["total_time"]
                tok = e["total_tokens"]
                nfe = e["total_nfe"]
                return {
                    "tok_per_sec": tok / t   if t   > 0 else float("nan"),
                    "tok_per_nfe": tok / nfe if nfe > 0 else float("nan"),
                }
    return None


# ── LaTeX generation ──────────────────────────────────────────────────────────

FACTOR_VALUE    = "1.0"
THRESHOLD_VALUE = "0.9"

# Each entry: (accuracy_strategy, speed_strategy)
# Accuracy and speed can come from different runs (cache only affects speed).
CELL_PATHS: dict[tuple[str, str], tuple[str, str]] = {
    ("GSM8K",     "factor"):    ("factor/cache-block32-batch1",  "factor/cache-block32-batch1"),
    ("GSM8K",     "threshold"): ("threshold/no-cache-0.9",       "threshold/cache-block32-batch1"),
    ("HumanEval", "factor"):    ("factor/cache-block32-batch1",   "factor/cache-block32-batch1"),
    ("HumanEval", "threshold"): ("threshold/cache-block32-batch1","threshold/cache-block32-batch1"),
}


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}\\%" if v is not None else "---"


def _spd(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "---"


def _nfe(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "---"


def build_table() -> str:
    # Collect all data first
    datasets   = ["GSM8K", "HumanEval"]
    strategies = ["factor", "threshold"]
    ds_names   = {"GSM8K": "GSM8K", "HumanEval": "HumanEval"}

    data: dict[str, dict[str, dict]] = {}
    for ds_key in datasets:
        data[ds_key] = {}
        for strat_key in strategies:
            acc_path, spd_path = CELL_PATHS[(ds_key, strat_key)]
            acc_fn = load_gsm8k_accuracy if ds_key == "GSM8K" else load_humaneval_accuracy
            data[ds_key][strat_key] = {
                "acc":   acc_fn(acc_path),
                "speed": load_speed(ds_names[ds_key], spd_path),
            }

    lines = [
        r"% Table comparing Factor vs Threshold parallel decoding strategies (batch size 1)",
        r"% Rows: dataset; Columns: strategy",
        r"% Each cell: Accuracy / Throughput (tok/s) / Tokens per NFE",
        r"%",
        r"% Data sources:",
        r"%   Accuracy : evals_results/v2/{dataset}-FastDLLMv2/{strategy}/cache-block32-batch1/",
        r"%   Speed    : results/v2/{dataset}-FastDLLMv2/{strategy}/cache-block32-batch1/rank_0.jsonl (last line)",
        r"%",
        r"% Model: Efficient-Large-Model/Fast_dLLM_v2_1.5B",
        f"% Factor value: {FACTOR_VALUE}   |   Threshold: {THRESHOLD_VALUE}",
        r"% GSM8K accuracy: flexible-extract exact match",
        r"% HumanEval accuracy: pass@1 via llada/postprocess_code.py (lm-eval pass@1 is unreliable for this setup)",
        r"",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{%",
        r"  Comparison of factor-based and threshold-based parallel decoding strategies",
        r"  at batch size~1 using the Fast-dLLM v2 1.5B model.",
        r"}",
        r"\label{tab:factor-vs-threshold-batch1}",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{l cc}",
        r"\toprule",
        rf"& \textbf{{Factor ($\alpha={FACTOR_VALUE}$)}} & \textbf{{Threshold ($\tau={THRESHOLD_VALUE}$)}} \\",
        r"\midrule",
    ]

    for i, ds_key in enumerate(datasets):
        f = data[ds_key]["factor"]
        t = data[ds_key]["threshold"]

        f_spd = f["speed"] or {}
        t_spd = t["speed"] or {}

        lines += [
            rf"\multirow{{3}}{{*}}{{\textbf{{{ds_key}}}}}",
            rf"  & Acc.: {_pct(f['acc'])}    & Acc.: {_pct(t['acc'])} \\",
            rf"  & {_spd(f_spd.get('tok_per_sec'))} tok/s     & {_spd(t_spd.get('tok_per_sec'))} tok/s \\",
            rf"  & {_nfe(f_spd.get('tok_per_nfe'))} tok/NFE    & {_nfe(t_spd.get('tok_per_nfe'))} tok/NFE \\",
        ]
        if i < len(datasets) - 1:
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true",
                        help="print to stdout only, do not write the .tex file")
    args = parser.parse_args()

    table = build_table()

    print(table)

    if not args.no_write:
        out = ROOT_DIR / "figures" / "table_factor_vs_threshold_batch1.tex"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
        print(f"Written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
