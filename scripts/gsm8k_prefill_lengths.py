#!/usr/bin/env python3
"""
Print the tokenized prefill length for each AIME 2024 sample at the
5-shot configuration, using Maxwell-Jia/AIME_2024.

Prompt format:
  - 5 examples from the top of the dataset formatted as "Problem: ...\nAnswer: ...\n\n"
  - Test question formatted as "Problem: ...\nAnswer:"
  - The whole thing wrapped in the LLaDA instruct chat template.

Usage:
    python scripts/gsm8k_prefill_lengths.py
    python scripts/gsm8k_prefill_lengths.py --num_fewshot 0
"""

import argparse
import random
import statistics
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
DATASET = "Maxwell-Jia/AIME_2024"


def format_shot(doc: dict) -> str:
    """Format one AIME doc as a few-shot example (problem + worked solution + answer)."""
    return f"Problem: {doc['Problem']}\nSolution: {doc['Solution']}\nAnswer: {doc['Answer']}\n\n"


def format_query(doc: dict) -> str:
    """Format the test query (problem only, no answer)."""
    return f"Problem: {doc['Problem']}\nAnswer:"


def build_prompt(test_doc: dict, fewshot_docs: list[dict]) -> str:
    """Concatenate few-shot context and test query."""
    context = "".join(format_shot(d) for d in fewshot_docs)
    return context + format_query(test_doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on test samples (None = all)")
    parser.add_argument("--random_fewshot", action="store_true",
                        help="Sample few-shot examples randomly per test doc (excluding the test doc itself)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    all_docs = list(ds)

    if args.random_fewshot:
        ds_test = ds
    else:
        # Fixed: first num_fewshot docs are the few-shot pool, rest are test
        fewshot_docs = all_docs[:args.num_fewshot]
        ds_test = ds.select(range(args.num_fewshot, len(ds)))

    if args.max_samples is not None:
        ds_test = ds_test.select(range(min(args.max_samples, len(ds_test))))

    fewshot_mode = "random (per-sample)" if args.random_fewshot else "fixed (first N)"
    print(f"\nnum_fewshot={args.num_fewshot}, test_samples={len(ds_test)}, fewshot={fewshot_mode}\n")

    # Solution length stats across the full dataset (all 30 problems)
    sol_char_lens = [len(doc["Solution"]) for doc in ds]
    sol_tok_lens  = [len(tokenizer(doc["Solution"])["input_ids"]) for doc in ds]
    print(f"--- Solution lengths (all {len(ds)} problems) ---")
    print(f"  chars  — mean: {statistics.mean(sol_char_lens):.1f}  "
          f"min: {min(sol_char_lens)}  max: {max(sol_char_lens)}")
    print(f"  tokens — mean: {statistics.mean(sol_tok_lens):.1f}  "
          f"min: {min(sol_tok_lens)}  max: {max(sol_tok_lens)}\n")

    def sample_fewshot(test_doc: dict) -> list[dict]:
        if args.random_fewshot:
            pool = [d for d in all_docs if d["ID"] != test_doc["ID"]]
            return random.sample(pool, args.num_fewshot)
        return fewshot_docs

    # Print an example prompt for the first test doc
    if len(ds_test) > 0:
        example_fewshot = sample_fewshot(ds_test[0])
        example_prompt = build_prompt(ds_test[0], example_fewshot)
        m = [{"role": "user", "content": example_prompt}]
        example_chat = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        print("=" * 70)
        print(f"EXAMPLE PROMPT (first test sample, {args.num_fewshot}-shot prefill):")
        print("=" * 70)
        print(example_chat)
        print("=" * 70 + "\n")

    lengths = []
    for i, doc in enumerate(ds_test):
        prompt_text = build_prompt(doc, sample_fewshot(doc))

        m = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(chat_text)["input_ids"]
        length = len(input_ids)
        lengths.append(length)
        print(f"[{i:4d}] id={doc.get('ID', i)!s:25s}  prefill_len={length}")

    print(f"\n--- Summary ({len(lengths)} samples, {args.num_fewshot}-shot) ---")
    print(f"  min    : {min(lengths)}")
    print(f"  max    : {max(lengths)}")
    print(f"  mean   : {statistics.mean(lengths):.1f}")
    print(f"  median : {statistics.median(lengths):.1f}")
    print(f"  stdev  : {statistics.stdev(lengths):.1f}")


if __name__ == "__main__":
    main()
