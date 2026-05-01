#!/usr/bin/env python3
"""
Needle-in-a-Haystack (NIAH) evaluation for LLaDA-8B-Instruct (Direct Extension),
reproducing Figure 2 of LongLLaDA (arxiv:2506.14429).

Experimental setup matches the paper:
  - steps=32, block_length=32, gen_length=32
  - Context lengths: 2k, 4k, 8k, 16k, 24k, 32k tokens (GPT-4 tokenisation)
  - Depths: 0–100% in 10-point increments
  - 10 needle/haystack pairs per (context_length, depth) cell
  - English haystack (en_un_asr.jsonl) + needles.jsonl from opencompass/needlebench
  - Prompt format: guide=True, position=End (matches LongLLaDA origin.py)
  - Scoring: NeedleBenchOriginEvaluator (keyword hit → 100, else 0.2 × Lev-sim)

Supported cache modes (our addition, not in the original paper):
  none   – plain generate()               (baseline, expensive at long contexts)
  prefix – generate_with_prefix_cache()
  dual   – generate_with_dual_cache()

Usage (run from the llada/ directory):
  python eval_niah.py --cache_mode prefix
  python eval_niah.py --cache_mode dual --context_lengths 2000 4000 8000
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM

try:
    import tiktoken
    _GPT4_ENC = tiktoken.encoding_for_model('gpt-4')
except ImportError:
    _GPT4_ENC = None
    print("WARNING: tiktoken not installed; context lengths will be approximate. "
          "Install with: pip install tiktoken")

MASK_ID = 126336
DEFAULT_CONTEXT_LENGTHS = [2000, 4000, 8000, 16000, 24000, 32000]
DEFAULT_DEPTHS           = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LENGTH_BUFFER            = 600   # mirrors LongLLaDA: actual context = length - buffer


# ── GPT-4 token helpers ───────────────────────────────────────────────────────

def _enc(text: str) -> list:
    if _GPT4_ENC is not None:
        return _GPT4_ENC.encode(text)
    # Approximate: 1 UTF-8 byte ≈ 1 token (over-counts; safe for buffered targets)
    return list(text.encode('utf-8'))


def _dec(tokens: list) -> str:
    if _GPT4_ENC is not None:
        return _GPT4_ENC.decode(tokens)
    return bytes(tokens).decode('utf-8', errors='replace')


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_needlebench(cache_dir=None) -> str:
    """Return local path to the opencompass/needlebench HuggingFace dataset."""
    return snapshot_download(
        repo_id='opencompass/needlebench',
        repo_type='dataset',
        cache_dir=cache_dir,
    )


def _pick_needle(counter: int, needle_file: str, language: str = 'English'):
    """
    Deterministically pick a needle for repeat `counter`.
    Replicates the random.seed(counter) + random.choice() call in LongLLaDA.
    Returns (needle_text, retrieval_question, keyword).
    """
    with open(needle_file, 'r', encoding='utf-8') as f:
        items = [json.loads(l) for l in f if json.loads(l).get('language') == language]
    random.seed(counter)
    item = random.choice(items)
    return item['needle'], item['retrieval_question'], item['arg2']


def _build_context(haystack_tokens: list, depth_pct: int, needle_text: str) -> str:
    """Insert needle_text at depth_pct% into the haystack token sequence."""
    needle_tokens = _enc(needle_text)
    pos = int(len(haystack_tokens) * depth_pct / 100)
    merged = haystack_tokens[:pos] + needle_tokens + haystack_tokens[pos:]
    return _dec(merged)


def _build_prompt(context: str, retrieval_question: str) -> str:
    """
    Matches LongLLaDA origin.py with guide=True, position='End', language='English'.

    The two-step transformation:
      1. Insert a 'think about relevance' instruction before 'Please answer in the format'.
      2. Strip the format-string prefix and trailing format example.
    """
    # Step 1: insert guide instruction
    parts = retrieval_question.split('Please answer in the format')
    if len(parts) == 2:
        retrieval_question = (
            parts[0]
            + 'Before answering, please consider what in the document is most '
              'relevant to this question. Please answer in the format'
            + parts[1]
        )

    # Step 2: strip format label and trailing example (matches origin.py exactly)
    question_clean = retrieval_question.replace("Please answer in the format '", '')
    if len(question_clean) >= 10:
        question_clean = question_clean[:-10]

    return (
        'You are an intelligent AI assistant skilled in answering user questions.\n'
        'Please keep your answers concise and clear. Do not talk about irrelevant '
        'topics or repeat your answers.\n'
        f'The document given to you by the user is {context}\n\n'
        f'Now, the question is: {question_clean}'
    )


# ── Scoring (NeedleBenchOriginEvaluator) ─────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + int(c1 != c2)))
        prev = curr
    return prev[-1]


def score_prediction(prediction: str, needle_text: str, keyword: str) -> float:
    """
    Returns 0–100.
      - 100  if keyword appears verbatim in the raw prediction.
      - 0.2 × Levenshtein-similarity(prediction, needle_text) otherwise.
    Whitespace is stripped before similarity comparison (matches the evaluator).
    """
    if keyword in prediction:
        return 100.0
    ref  = re.sub(r'\s+', '', needle_text)
    pred = re.sub(r'\s+', '', prediction)
    max_len = max(len(ref), len(pred))
    if max_len == 0:
        return 100.0
    lev_sim = 1.0 - _levenshtein(pred, ref) / max_len
    return 0.2 * 100.0 * lev_sim


# ── Main evaluation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_niah(args) -> dict:
    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config,
    ).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Fetching opencompass/needlebench …")
    data_path   = fetch_needlebench(args.data_cache_dir)
    haystack_file = os.path.join(data_path, 'en_un_asr.jsonl')
    needle_file   = os.path.join(data_path, 'needles.jsonl')

    with open(haystack_file, 'r', encoding='utf-8') as f:
        all_docs = [json.loads(l) for l in f]

    # ── Generation config ─────────────────────────────────────────────────────
    # Round gen_length up to a multiple of block_length (required by all generators)
    gen_length = args.gen_length
    if gen_length % args.block_length != 0:
        gen_length = (gen_length // args.block_length + 1) * args.block_length

    print(f"cache_mode={args.cache_mode}  steps={args.steps}  "
          f"block_length={args.block_length}  gen_length={gen_length}\n")

    results: dict = {}   # {ctx_len: {depth: [per-repeat scores]}}

    for ctx_len in args.context_lengths:
        results[ctx_len] = {}
        for depth in args.depths:
            cell_scores = []
            docs = all_docs.copy()

            for repeat in range(args.num_repeats):
                # Shuffle haystack — matches LongLLaDA outer random.seed(counter)
                random.seed(repeat)
                random.shuffle(docs)

                # Pick needle — _pick_needle also calls random.seed(repeat) internally,
                # matching LongLLaDA's get_random_line_by_language(counter, ...)
                needle_text, retrieval_q, keyword = _pick_needle(repeat, needle_file)
                needle_with_newlines = '\n' + needle_text + '\n'

                # Accumulate haystack tokens up to target length
                needle_tok_count = len(_enc(needle_with_newlines))
                target = max(0, ctx_len - LENGTH_BUFFER - needle_tok_count)
                haystack_tokens: list = []
                for doc in docs:
                    haystack_tokens.extend(_enc(doc['text']))
                    if len(haystack_tokens) >= target:
                        break
                haystack_tokens = haystack_tokens[:target]

                context     = _build_context(haystack_tokens, depth, needle_with_newlines)
                prompt_text = _build_prompt(context, retrieval_q)

                # Tokenise with instruct chat template
                chat_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt_text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                input_ids = torch.tensor(
                    tokenizer(chat_text)['input_ids'], dtype=torch.long
                ).unsqueeze(0).to('cuda')

                print(f"  [ctx={ctx_len:5d} depth={depth:3d}% rep={repeat}] "
                      f"prefill={input_ids.shape[1]:6d}", end='  ', flush=True)

                gen_kwargs = dict(
                    steps=args.steps,
                    gen_length=gen_length,
                    block_length=args.block_length,
                    temperature=0.,
                    remasking='low_confidence',
                    mask_id=MASK_ID,
                )
                if args.cache_mode == 'prefix':
                    out, nfe = generate_with_prefix_cache(model, input_ids, **gen_kwargs)
                elif args.cache_mode == 'dual':
                    out, nfe = generate_with_dual_cache(model, input_ids, **gen_kwargs)
                else:
                    out, nfe = generate(model, input_ids, **gen_kwargs)

                generated = tokenizer.decode(
                    out[0, input_ids.shape[1]:], skip_special_tokens=True
                )
                s = score_prediction(generated, needle_with_newlines, keyword)
                cell_scores.append(s)
                print(f"nfe={nfe:3d}  score={s:5.1f}  pred={generated[:50]!r}")

            avg = sum(cell_scores) / len(cell_scores)
            results[ctx_len][depth] = cell_scores
            print(f"  → avg score @ ctx={ctx_len} depth={depth}%: {avg:.1f}\n")

    return results


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(results: dict, out_path: str, title: str):
    import matplotlib.pyplot as plt

    ctx_lens = sorted(results.keys())
    depths   = sorted(next(iter(results.values())).keys())

    # rows=depths (0% at top), cols=context lengths
    matrix = np.array([
        [np.mean(results[cl][d]) for cl in ctx_lens]
        for d in depths
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(ctx_lens) * 1.4), max(5, len(depths) * 0.8)))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto', origin='upper')

    ax.set_xticks(range(len(ctx_lens)))
    ax.set_xticklabels([f'{cl // 1000}k' for cl in ctx_lens], fontsize=10)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f'{d}%' for d in depths], fontsize=10)
    ax.set_xlabel('Context Length (GPT-4 tokens)', fontsize=11)
    ax.set_ylabel('Needle Depth', fontsize=11)
    ax.set_title(title, fontsize=12)

    for i in range(len(depths)):
        for j in range(len(ctx_lens)):
            v = matrix[i, j]
            txt_color = 'white' if v < 30 or v > 85 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Score (0–100)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NIAH evaluation for LLaDA-8B-Instruct — Direct Extension (LongLLaDA Fig. 2)'
    )
    parser.add_argument('--model_path', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument(
        '--cache_mode', choices=['none', 'prefix', 'dual'], default='prefix',
        help='none = baseline generate(); prefix = prefix KV-cache; dual = dual KV-cache',
    )
    parser.add_argument('--context_lengths', nargs='+', type=int,
                        default=DEFAULT_CONTEXT_LENGTHS,
                        metavar='N',
                        help='Context lengths in GPT-4 tokens to evaluate')
    parser.add_argument('--depths', nargs='+', type=int,
                        default=DEFAULT_DEPTHS,
                        metavar='D',
                        help='Needle insertion depths as percentages (0–100)')
    parser.add_argument('--num_repeats', type=int, default=10,
                        help='Needle/haystack samples per (context_length, depth) cell')
    parser.add_argument('--steps', type=int, default=32,
                        help='Diffusion steps (= steps_per_block when gen_length==block_length)')
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--gen_length', type=int, default=32,
                        help='Max tokens to generate; rounded up to a multiple of block_length')
    parser.add_argument('--data_cache_dir', default=None,
                        help='HuggingFace cache dir for the needlebench dataset download')
    _default_output_dir = str(Path(__file__).resolve().parent.parent / 'results' / 'niah')
    parser.add_argument('--output_dir', default=_default_output_dir)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_niah(args)

    # Save raw results (JSON keys must be strings)
    json_path = os.path.join(args.output_dir, f'niah_{args.cache_mode}.json')
    with open(json_path, 'w') as f:
        json.dump(
            {str(cl): {str(d): v for d, v in depth_map.items()}
             for cl, depth_map in results.items()},
            f, indent=2,
        )
    print(f"Results saved → {json_path}")

    png_path = os.path.join(args.output_dir, f'niah_{args.cache_mode}.png')
    plot_heatmap(
        results, png_path,
        title=f'LLaDA-8B-Instruct · NIAH Direct Extension · {args.cache_mode} cache',
    )
