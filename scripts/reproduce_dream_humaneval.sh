#!/usr/bin/env bash
set -euo pipefail

# Reproduce Dream HumanEval experiments for 5 settings:
#   baseline, prefix-cache, parallel, prefix-cache-parallel, dual-cache-parallel
#
# Usage:
#   bash scripts/reproduce_dream_humaneval.sh [all|baseline|prefix-cache|parallel|prefix-cache-parallel|dual-cache-parallel]
#
# Optional environment overrides:
#   MODEL_PATH, GEN_LENGTH, BLOCK_LENGTH, OUTPUT_PATH, SAVE_DIR

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export LD_LIBRARY_PATH="/sw/rh9.4/python/miniforge3/lib:$LD_LIBRARY_PATH"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/dream"

task="humaneval"
length="${GEN_LENGTH:-256}"
block_length="${BLOCK_LENGTH:-32}"
steps_parallel="$((length / block_length))"
model_path="${MODEL_PATH:-Dream-org/Dream-v0-Base-7B}"
output_path="${OUTPUT_PATH:-${ROOT_DIR}/evals_results/HumanEval-Dream/len${length}}"
save_dir="${SAVE_DIR:-${ROOT_DIR}/results/HumanEval-Dream/len${length}}"
mode="${1:-all}"

run_eval() {
    local tag="$1"
    local extra_model_args="$2"

    local current_save_dir="${save_dir}/${tag}"

    echo "[humaneval] Running ${tag}, saving to ${current_save_dir}"

    accelerate launch eval.py \
        --model dream \
        --model_args "pretrained=${model_path},max_new_tokens=${length},add_bos_token=true,show_speed=True,escape_until=true,save_dir=${current_save_dir},${extra_model_args}" \
        --tasks "${task}" \
        --batch_size 1 \
        --confirm_run_unsafe_code \
        --output_path "${output_path}/${tag}" \
        --log_samples
}

run_baseline() {
    run_eval "baseline" "diffusion_steps=${length},alg=entropy"
}

run_prefix_cache() {
    run_eval "prefix-cache" "diffusion_steps=${length},alg=entropy,use_cache=true"
}

run_parallel() {
    run_eval "parallel" "diffusion_steps=${steps_parallel},alg=confidence_threshold,threshold=0.9"
}

run_prefix_cache_parallel() {
    run_eval "prefix-cache-parallel" "diffusion_steps=${steps_parallel},alg=confidence_threshold,threshold=0.9,use_cache=true"
}

run_dual_cache_parallel() {
    run_eval "dual-cache-parallel" "diffusion_steps=${steps_parallel},alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true"
}

case "${mode}" in
all)
    run_baseline
    run_prefix_cache
    run_parallel
    run_prefix_cache_parallel
    run_dual_cache_parallel
    ;;
baseline)
    run_baseline
    ;;
prefix-cache)
    run_prefix_cache
    ;;
parallel)
    run_parallel
    ;;
prefix-cache-parallel)
    run_prefix_cache_parallel
    ;;
dual-cache-parallel)
    run_dual_cache_parallel
    ;;
*)
    echo "Unknown mode: ${mode}"
    echo "Expected one of: all, baseline, prefix-cache, parallel, prefix-cache-parallel, dual-cache-parallel"
    exit 1
    ;;
esac
