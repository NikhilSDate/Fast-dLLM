#!/usr/bin/env python3
"""
Print the tokenized prefill length for each GSM8K test sample at the
5-shot configuration used in eval_llada.py / run_experiments.py.

The prompt format mirrors lm-eval-harness's GSM8K task:
  - 5 training examples formatted as "Question: ...\nAnswer: ...\n\n"
  - Test question formatted as "Question: ...\nAnswer:"
  - The whole thing wrapped in the LLaDA instruct chat template.

Usage:
    python scripts/gsm8k_prefill_lengths.py
    python scripts/gsm8k_prefill_lengths.py --num_fewshot 0 --max_samples 50
"""

import argparse
import statistics
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336


def format_shot(doc: dict) -> str:
    """Format one GSM8K doc as a few-shot example (question + answer)."""
    # print(doc['solution'])
    return f"\nAnswer: {doc['solution']}\n\n"


def format_query(doc: dict) -> str:
    """Format the test query (question only, no answer)."""
    return f"Question: {doc['problem']}\nAnswer:"


def build_prompt(test_doc: dict, fewshot_docs: list[dict]) -> str:
    """Concatenate few-shot context and test query."""
    context = "".join(format_shot(d) for d in fewshot_docs)
    return context + format_query(test_doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on test samples to process (None = all 1319)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading GSM8K dataset ...")
    # ds_train = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    # ds_test  = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    ds_train = load_dataset("L4NLP/LEval", "financial_qa", split="test")
    ds_test  = load_dataset("L4NLP/LEval", "financial_qa", split="test")

    # lm-eval draws few-shot examples from the top of the training split
    fewshot_docs = list(ds_train.select(range(args.num_fewshot)))

    if args.max_samples is not None:
        ds_test = ds_test.select(range(min(args.max_samples, len(ds_test))))

    print(f"\nnum_fewshot={args.num_fewshot}, test_samples={len(ds_test)}\n")

    lengths = []
    for i, doc in enumerate(ds_test):
        prompt_text = build_prompt(doc, fewshot_docs)

        # eval_llada.py wraps the prompt in the chat template for instruct models
        m = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(chat_text)["input_ids"]
        length = len(input_ids)
        lengths.append(length)
        print(f"[{i:4d}] prefill_len={length}")

    print(f"\n--- Summary ({len(lengths)} samples) ---")
    print(f"  min    : {min(lengths)}")
    print(f"  max    : {max(lengths)}")
    print(f"  mean   : {statistics.mean(lengths):.1f}")
    print(f"  median : {statistics.median(lengths):.1f}")
    print(f"  stdev  : {statistics.stdev(lengths):.1f}")


if __name__ == "__main__":
    main()
