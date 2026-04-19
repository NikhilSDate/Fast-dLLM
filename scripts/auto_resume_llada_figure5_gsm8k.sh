#!/usr/bin/env bash
# auto_resume_llada_figure5_gsm8k.sh — Keep re-requesting srun resources until each Figure 5 run is done.
#
# Usage (inside a tmux session):
#   bash scripts/auto_resume_llada_figure5_gsm8k.sh [all|thresholds|fixed]
#
# Optional overrides:
#   MODEL_PATH, GEN_LENGTH, BLOCK_LENGTH, NUM_FEWSHOT, THRESHOLDS,
#   FIXED_TOKENS_PER_STEP, OUTPUT_ROOT, SAVE_ROOT, VENV
#
# The underlying eval script resumes from rank_0.jsonl automatically, so each
# srun call picks up exactly where the last one left off.

set -uo pipefail

MODE="${1:-all}"
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/u/jvancosampedro/fastdllm_venv/bin/activate}"

TASK="gsm8k"
TOTAL=1319
OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/Figure5/GSM8K/len${GEN_LENGTH}}"
SAVE_ROOT="${SAVE_ROOT:-./results/Figure5/GSM8K/len${GEN_LENGTH}}"

THRESHOLDS=( ${THRESHOLDS:-0.5 0.6 0.7 0.8 0.9 1.0} )
FIXED_TOKENS_PER_STEP=( ${FIXED_TOKENS_PER_STEP:-1 2 4 8} )

# srun resource parameters (same shape as scripts/auto_resume.sh)
SRUN_ACCOUNT="bdes-delta-gpu"
SRUN_TIME="02:30:00"
SRUN_PARTITION="gpuA100x4"
SRUN_NODES=1
SRUN_NTASKS=1
SRUN_GPUS=1
SRUN_MEM="32g"

mkdir -p "${OUTPUT_ROOT}" "${SAVE_ROOT}"

get_done() {
    local result_file="$1"
    [[ -f "${result_file}" ]] && grep -c '"answer"' "${result_file}" || echo 0
}

run_eval_until_done() {
    local tag="$1"
    local extra_model_args="$2"

    local current_save_dir="${SAVE_ROOT}/${tag}"
    local current_output_path="${OUTPUT_ROOT}/${tag}"
    local result_file="${current_save_dir}/rank_0.jsonl"

    mkdir -p "${current_save_dir}" "${current_output_path}"

    echo ""
    echo "=============================================================="
    echo " Starting: tag=${tag}"
    echo " Result  : ${result_file}"
    echo " Save    : ${current_save_dir}"
    echo " Output  : ${current_output_path}"
    echo "=============================================================="

    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local done
        done=$(get_done "${result_file}")

        echo "[$(date)]  ${tag}  Attempt #${attempt}  |  ${done}/${TOTAL}"

        if [[ ${done} -ge ${TOTAL} ]]; then
            echo "[$(date)]  Complete (${done}/${TOTAL})."
            break
        fi

        echo "[$(date)]  Requesting srun resources..."

        srun \
            -A "${SRUN_ACCOUNT}" \
            --time="${SRUN_TIME}" \
            --nodes="${SRUN_NODES}" \
            --ntasks="${SRUN_NTASKS}" \
            --partition="${SRUN_PARTITION}" \
            --gpus="${SRUN_GPUS}" \
            --mem="${SRUN_MEM}" \
            bash -c "source ${VENV} && cd ${ROOT_DIR} && \
                accelerate launch llada/eval_llada.py \
                    --tasks ${TASK} \
                    --num_fewshot ${NUM_FEWSHOT} \
                    --confirm_run_unsafe_code \
                    --model llada_dist \
                    --model_args \"model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},${extra_model_args},show_speed=True,save_dir=${current_save_dir}\" \
                    --output_path ${current_output_path} \
                    --log_samples" \
            && SRUN_EXIT=0 || SRUN_EXIT=$?

        done=$(get_done "${result_file}")
        echo "[$(date)]  srun exited (code=${SRUN_EXIT})  |  Progress now: ${done}/${TOTAL}"

        if [[ ${done} -ge ${TOTAL} ]]; then
            echo "[$(date)]  Complete!"
            break
        fi

        echo "[$(date)]  Waiting 15s before re-queuing..."
        sleep 15
    done
}

run_threshold_sweep() {
    for threshold in "${THRESHOLDS[@]}"; do
        local tag="threshold/t${threshold//./p}"
        run_eval_until_done "${tag}" "steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${threshold}"
    done
}

run_fixed_step_baselines() {
    for tokens_per_step in "${FIXED_TOKENS_PER_STEP[@]}"; do
        if (( GEN_LENGTH % tokens_per_step != 0 )); then
            echo "[ERROR] GEN_LENGTH=${GEN_LENGTH} must be divisible by tokens_per_step=${tokens_per_step}" >&2
            exit 1
        fi

        local steps=$((GEN_LENGTH / tokens_per_step))
        local tag="fixed/${tokens_per_step}tok-per-step"
        run_eval_until_done "${tag}" "steps=${steps},block_length=${BLOCK_LENGTH}"
    done
}

echo "================================================================"
echo " auto_resume_llada_figure5_gsm8k.sh  mode=${MODE}  started $(date)"
echo " Results: ${SAVE_ROOT}"
echo " Outputs: ${OUTPUT_ROOT}"
echo "==============================================================="

case "${MODE}" in
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
    echo "Unknown mode: ${MODE}"
    echo "Expected one of: all, thresholds, threshold-sweep, fixed, fixed-baselines"
    exit 1
    ;;
esac

echo ""
echo "================================================================"
echo " auto_resume_llada_figure5_gsm8k.sh finished $(date)"
echo " Final outputs: ${OUTPUT_ROOT}"
echo " Final results: ${SAVE_ROOT}"
echo "================================================================"