#!/usr/bin/env bash
set -euo pipefail

# Reproduce LLaDA GSM8K experiments for 5 settings:
#   baseline, prefix-cache, parallel, prefix-cache-parallel, dual-cache-parallel
#
# Usage:
#   bash scripts/reproduce_llada_gsm8k.sh [all|baseline|prefix-cache|parallel|prefix-cache-parallel|dual-cache-parallel]
#
# Optional environment overrides:
#   MODEL_PATH, GEN_LENGTH, BLOCK_LENGTH, NUM_FEWSHOT, OUTPUT_PATH, SAVE_DIR

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export LD_LIBRARY_PATH="/sw/rh9.4/python/miniforge3/lib:$LD_LIBRARY_PATH"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

task="gsm8k"
length="${GEN_LENGTH:-512}"
block_length="${BLOCK_LENGTH:-32}"
num_fewshot="${NUM_FEWSHOT:-5}"
steps_parallel="$((length / block_length))"
model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
output_path="${OUTPUT_PATH:-evals_results/GSM8K-LLaDA/len${length}}"
save_dir="${SAVE_DIR:-./results/GSM8K-LLaDA/len${length}}"
mode="${1:-all}"

run_eval() {
    local tag="$1"
    local extra_model_args="$2"
    
    local current_save_dir="${save_dir}/${tag}"
    
    echo "[gsm8k] Running ${tag}, saving to ${current_save_dir}"

    accelerate launch llada/eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},${extra_model_args},show_speed=True,save_dir=${current_save_dir}" \
        --output_path "${output_path}/${tag}" \
        --log_samples
}

run_baseline() {
    run_eval "baseline" "steps=${length},block_length=${block_length}"
}

run_prefix_cache() {
    run_eval "prefix-cache" "steps=${length},block_length=${block_length},use_cache=True"
}

run_parallel() {
    run_eval "parallel" "steps=${steps_parallel},block_length=${block_length},threshold=0.9"
}

run_prefix_cache_parallel() {
    run_eval "prefix-cache-parallel" "steps=${steps_parallel},block_length=${block_length},use_cache=True,threshold=0.9"
}

run_dual_cache_parallel() {
    run_eval "dual-cache-parallel" "steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9"
}

run_prefix_cache_variable() {
    run_eval "prefix-cache-variable" "steps=${length},block_length=${block_length},variable_cache=True"
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
prefix-cache-variable)
    run_prefix_cache_variable
    ;;
*)
    echo "Unknown mode: ${mode}"
    echo "Expected one of: all, baseline, prefix-cache, parallel, prefix-cache-parallel, dual-cache-parallel, prefix-cache-variable"
    exit 1
    ;;
esac
