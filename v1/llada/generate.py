# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    print(f'batch size = {prompt.shape[0]}')

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@torch.no_grad()
def generate_prefix_cache_variable(model, prompt, steps=128, max_gen_length=256, block_length=128,
                                   temperature=0., remasking='low_confidence', mask_id=126336,
                                   eos_id=None, threshold=None, factor=None):
    '''
    Variable-length generation using prefix KV-cache. Generates block-by-block up to
    max_gen_length, stopping early at the first block that contains eos_id.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Total sampling steps (divided evenly across blocks).
        max_gen_length: Maximum tokens to generate; must be divisible by block_length.
        block_length: Semi-autoregressive block size.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] (default 126336).
        eos_id: Token id used as end-of-sequence. If None, runs to max_gen_length.
        threshold: Confidence threshold for unmasking (alternative to step-based quota).
        factor: Dynamic transfer factor.

    Returns:
        x: (B, Lp + generated_length) — truncated at first EOS if found, else max_gen_length.
        nfe: Number of function evaluations used.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + max_gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert max_gen_length % block_length == 0
    num_blocks = max_gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Step 0: full forward pass to warm KV cache for the prefix
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        nfe += 1

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        # Truncate KV cache to prompt prefix only (strip block and beyond)
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        past_key_values = new_past_key_values

        # Refinement steps: only process current block with cached prefix
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                        x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index,
                                                                 x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            i += 1

        # Check for EOS in the completed block
        if eos_id is not None:
            eos_mask = (x[:, current_block_start:current_block_end] == eos_id)  # (B, block_length)
            if eos_mask.any():
                # Find the first EOS position per batch item; take the earliest across items that have one
                has_eos = eos_mask.any(dim=1)                               # (B,)
                first_eos_per_item = eos_mask.int().argmax(dim=1)           # (B,) — valid only where has_eos
                first_eos_col = first_eos_per_item[has_eos].min().item()    # earliest EOS column in block
                cutoff = current_block_start + first_eos_col
                return x[:, :cutoff + 1], nfe                               # include the EOS token

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe



@torch.no_grad()
def generate_streaming_blocks(
    model, prompt, steps=128, block_length=32, max_gen_length=2048,
    temperature=0., remasking='low_confidence', mask_id=126336,
    eos_id=None, threshold=None, use_cache=False,
):
    '''
    Streaming block generation: allocate and decode one block of mask tokens at a
    time. After each block is fully decoded, check for EOS; if not found, append
    the next block of mask tokens. No suffix masks beyond the current block are
    ever present in the sequence, so the model never attends to future positions.

    Args:
        model: Mask predictor.
        prompt: (B, Lp) long tensor.
        steps: Diffusion steps *per block* (not total).
        block_length: Number of tokens per streaming block.
        max_gen_length: Hard cap on generated tokens; must be a multiple of block_length.
        temperature: Gumbel noise temperature.
        remasking: 'low_confidence' or 'random'.
        mask_id: Token id for [MASK] (default 126336).
        eos_id: If given, stop at the first block that contains this token.
        threshold: Confidence threshold (alternative to step-based quota).
        use_cache: When True, cache the KV for the prefix before the current block.

    Returns:
        x: (B, Lp + generated) — truncated at first EOS block if eos_id is set.
        nfe: Total number of model forward passes.
    '''
    assert max_gen_length % block_length == 0, \
        f"max_gen_length={max_gen_length} must be divisible by block_length={block_length}"

    B, Lp = prompt.shape
    device = model.device

    # Pre-allocate the full output buffer once.  The generation region is already
    # all mask_id, so future blocks are implicitly present; we just slide the
    # active window x[:, :e] forward each iteration without any reallocation.
    x = torch.full((B, Lp + max_gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt.clone().to(device)

    nfe = 0

    for block_idx in range(max_gen_length // block_length):
        s = Lp + block_idx * block_length   # block start (absolute)
        e = s + block_length                 # block end   (absolute)

        # x[:, :e] is the active window: prompt + decoded blocks + current mask
        # block.  Nothing beyond position e is ever passed to the model.
        block_mask_index    = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        if use_cache:
            # ── Prefix-cache path ────────────────────────────────────────────
            output = model(x[:, :e], use_cache=True)
            past_key_values = output.past_key_values
            nfe += 1

            # Step 0: unmask from full-pass logits, active window only
            mask_index = (x[:, :e] == mask_id)
            x0, transfer_index = get_transfer_index(
                output.logits, temperature, remasking, mask_index, x[:, :e],
                num_transfer_tokens[:, 0] if threshold is None else None, threshold,
            )
            x[:, :e][transfer_index] = x0[transfer_index]

            # Truncate KV cache to the decoded prefix (positions 0..s-1)
            past_key_values = [
                tuple(kv[:, :, :s] for kv in layer_kv)
                for layer_kv in past_key_values
            ]

            # Refinement: current block only, on top of the cached prefix
            step_i = 1
            while True:
                if (x[:, s:e] == mask_id).sum() == 0:
                    break
                nfe += 1
                mask_index_blk = (x[:, s:e] == mask_id)
                logits = model(
                    x[:, s:e],
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index_blk, x[:, s:e],
                    num_transfer_tokens[:, step_i] if threshold is None else None, threshold,
                )
                x[:, s:e][transfer_index] = x0[transfer_index]
                step_i += 1
        else:
            # ── No-cache path ────────────────────────────────────────────────
            for step_i in range(steps):
                if (x[:, s:e] == mask_id).sum() == 0:
                    break
                nfe += 1
                mask_index = (x[:, :e] == mask_id)
                logits = model(x[:, :e]).logits
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x[:, :e],
                    num_transfer_tokens[:, step_i] if threshold is None else None, threshold,
                )
                x[:, :e][transfer_index] = x0[transfer_index]

        # Check for EOS in the completed block
        if eos_id is not None:
            eos_in_block = (x[:, s:e] == eos_id)
            if eos_in_block.any():
                has_eos       = eos_in_block.any(dim=1)
                first_eos_col = eos_in_block.int().argmax(dim=1)
                earliest_col  = first_eos_col[has_eos].min().item()
                return x[:, :s + earliest_col + 1], nfe

    return x, nfe


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')

        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])


def main_variable():
    import time

    device = 'cuda'
    MAX_GEN_LENGTH = 2048  # generous ceiling; early stopping may cut this short
    BLOCK_LENGTH   = 32
    STEPS          = 128  # must satisfy: STEPS % (MAX_GEN_LENGTH // BLOCK_LENGTH) == 0

    assert STEPS % (MAX_GEN_LENGTH // BLOCK_LENGTH) == 0, \
        f"steps={STEPS} must be divisible by num_blocks={MAX_GEN_LENGTH // BLOCK_LENGTH}"

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "What is 1 + 1?"
    m = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to(device)
    prompt_len = input_ids.shape[1]

    print(f"Prompt tokens : {prompt_len}")
    print(f"Max gen length: {MAX_GEN_LENGTH}")
    print(f"EOS token id  : {tokenizer.eos_token_id}")
    print()

    # --- variable-length run ---
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_var, nfe_var = generate_prefix_cache_variable(
            model, input_ids,
            steps=STEPS,
            max_gen_length=MAX_GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            temperature=0.,
            remasking='low_confidence',
            eos_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        t_var = time.perf_counter() - t0

    gen_len_var = out_var.shape[1] - prompt_len
    text_var = tokenizer.batch_decode(out_var[:, prompt_len:], skip_special_tokens=True)[0]

    print(f"[variable] generated tokens : {gen_len_var} / {MAX_GEN_LENGTH}  (EOS {'hit' if gen_len_var < MAX_GEN_LENGTH else 'not hit'})")
    print(f"[variable] NFE              : {nfe_var}")
    print(f"[variable] wall time        : {t_var:.2f}s")
    print(f"[variable] output           :\n{text_var}")
    print()

    # --- fixed-length baseline for comparison ---
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_fix, nfe_fix = generate_with_prefix_cache(
            model, input_ids,
            steps=STEPS,
            gen_length=MAX_GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            temperature=0.,
            remasking='low_confidence',
        )
        torch.cuda.synchronize()
        t_fix = time.perf_counter() - t0

    text_fix = tokenizer.batch_decode(out_fix[:, prompt_len:], skip_special_tokens=True)[0]

    print(f"[fixed]    generated tokens : {MAX_GEN_LENGTH}")
    print(f"[fixed]    NFE              : {nfe_fix}")
    print(f"[fixed]    wall time        : {t_fix:.2f}s")
    print(f"[fixed]    output           :\n{text_fix}")


if __name__ == '__main__':
    main_variable()
