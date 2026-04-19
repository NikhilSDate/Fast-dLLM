#!/usr/bin/env bash
set -euo pipefail

# Reproduce the GSM8K runs needed for Figure 5 in the Fast-dLLM paper.
#
# Figure 5 uses LLaDA-Instruct on GSM8K with 5-shot prompting and compares:
#   - Confidence-aware parallel decoding across thresholds 0.5..1.0
#   - Fixed-step baselines that decode exactly 1, 2, 4, or 8 tokens per step
#
# Usage:
#   bash scripts/reproduce_llada_figure5_gsm8k.sh [all|thresholds|fixed]
#
# Optional overrides:
#   MODEL_PATH, GEN_LENGTH, BLOCK_LENGTH, NUM_FEWSHOT, THRESHOLDS,
#   FIXED_TOKENS_PER_STEP, OUTPUT_ROOT, SAVE_ROOT
#
# Defaults match the paper setup unless overridden.

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export LD_LIBRARY_PATH="/sw/rh9.4/python/miniforge3/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

task="gsm8k"
length="${GEN_LENGTH:-512}"
block_length="${BLOCK_LENGTH:-32}"
num_fewshot="${NUM_FEWSHOT:-5}"
model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
output_root="${OUTPUT_ROOT:-evals_results/Figure5/GSM8K/len${length}}"
save_root="${SAVE_ROOT:-./results/Figure5/GSM8K/len${length}}"
mode="${1:-all}"

thresholds=( ${THRESHOLDS:-0.5 0.6 0.7 0.8 0.9 1.0} )
fixed_tokens_per_step=( ${FIXED_TOKENS_PER_STEP:-1 2 4 8} )

run_eval() {
    local tag="$1"
    local extra_model_args="$2"

    local current_save_dir="${save_root}/${tag}"
    local current_output_path="${output_root}/${tag}"

    mkdir -p "${current_save_dir}" "${current_output_path}"

    echo "[figure5/gsm8k] Running ${tag}"
    echo "  save_dir    = ${current_save_dir}"
    echo "  output_path = ${current_output_path}"

    accelerate launch llada/eval_llada.py \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        --confirm_run_unsafe_code \
        --model llada_dist \
        --model_args "model_path=${model_path},gen_length=${length},${extra_model_args},show_speed=True,save_dir=${current_save_dir}" \
        --output_path "${current_output_path}" \
        --log_samples
}

run_threshold_sweep() {
    for threshold in "${thresholds[@]}"; do
        local tag="threshold/t${threshold//./p}"
        run_eval "${tag}" "steps=${length},block_length=${block_length},threshold=${threshold}"
    done
}

run_fixed_step_baselines() {
    for tokens_per_step in "${fixed_tokens_per_step[@]}"; do
        if (( length % tokens_per_step != 0 )); then
            echo "[ERROR] gen_length=${length} must be divisible by tokens_per_step=${tokens_per_step}" >&2
            exit 1
        fi

        local steps=$((length / tokens_per_step))
        local tag="fixed/${tokens_per_step}tok-per-step"
        run_eval "${tag}" "steps=${steps},block_length=${block_length}"
    done
}

case "${mode}" in
all)
    run_threshold_sweep
    run_fixed_step_baselines
    ;;
thresholds|threshold-sweep)
    run_threshold_sweep
    ;;
fixed|fixed-baselines)
    run_fixed_step_baselines
    ;;
*)
    echo "Unknown mode: ${mode}"
    echo "Expected one of: all, thresholds, threshold-sweep, fixed, fixed-baselines"
    exit 1
    ;;
esac

echo ""
echo "[figure5/gsm8k] Done."
echo "  Results: ${save_root}"
echo "  Eval outputs: ${output_root}"