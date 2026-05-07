import json
import glob
from collections import defaultdict

print("=== AIMO Throughput ===")
for f in sorted(glob.glob("results/*AIMO*/**/rank_0.jsonl", recursive=True)):
    try:
        lines = open(f).readlines()
        if not lines: continue
        total_time = 0
        total_tokens = 0
        for line in lines:
            d = json.loads(line)
            if "elapsed" in d and "tokens" in d:
                total_time += d["elapsed"]
                total_tokens += d["tokens"]
        if total_time > 0:
            print(f"{f:70} | {total_tokens/total_time:.2f} tok/s")
    except Exception as e: 
        print(f"Error reading {f}: {e}")

print("\n=== AIMO Accuracy ===")
found_evals = glob.glob("evals_results/*AIMO*/**/results_*.json", recursive=True)
if not found_evals:
    print("No complete JSON evaluations found.")
for f in found_evals:
    try:
        with open(f) as fin:
            data = json.load(fin)
            acc = data.get("results", {}).get("gsm8k", {}).get("exact_match,flexible-extract", "N/A")
            print(f"{f:70} | exact_match: {acc}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
