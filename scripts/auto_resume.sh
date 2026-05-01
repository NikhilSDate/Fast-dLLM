#!/usr/bin/env bash
# auto_resume.sh — Keep re-requesting srun resources until the eval is done.
#
# Usage (inside a tmux session):
#   bash scripts/auto_resume.sh [MODEL] [MODE]
#
#   MODEL : llada (default) | dream
#   MODE  : baseline | prefix-cache | parallel | prefix-cache-parallel
#           (dream also supports: dual-cache-parallel)
#
# Examples:
#   bash scripts/auto_resume.sh llada baseline
#   bash scripts/auto_resume.sh dream parallel
#   GEN_LENGTH=512 bash scripts/auto_resume.sh dream prefix-cache-parallel
#
# The underlying eval script already resumes from rank_0.jsonl automatically,
# so each srun call picks up exactly where the last one left off.

set -uo pipefail

MODEL="${1:-llada}"
MODE="${2:-baseline}"
GEN_LENGTH="${GEN_LENGTH:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/u/xzhou12/fastdllm_venv/bin/activate"

# GSM8K test set size
TOTAL=1319

# Resolve per-model result path and eval script
case "${MODEL}" in
llada)
    EVAL_SCRIPT="scripts/reproduce_llada_gsm8k.sh"
    DEFAULT_GEN_LENGTH=512
    SAVE_DIR_ROOT="${ROOT_DIR}/results"
    OUTPUT_PATH_ROOT="${ROOT_DIR}/evals_results"
    ;;
dream)
    EVAL_SCRIPT="scripts/reproduce_dream_gsm8k.sh"
    DEFAULT_GEN_LENGTH=256
    SAVE_DIR_ROOT="${ROOT_DIR}/results/GSM8K-Dream"
    OUTPUT_PATH_ROOT="${ROOT_DIR}/evals_results/GSM8K-Dream"
    ;;
*)
    echo "Unknown MODEL: ${MODEL}. Expected: llada | dream"
    exit 1
    ;;
esac

GEN_LENGTH="${GEN_LENGTH:-${DEFAULT_GEN_LENGTH}}"
SAVE_DIR="${SAVE_DIR_ROOT}/len${GEN_LENGTH}"
OUTPUT_PATH="${OUTPUT_PATH_ROOT}/len${GEN_LENGTH}"
RESULT_FILE="${SAVE_DIR}/${MODE}/rank_0.jsonl"

LOG_DIR="${ROOT_DIR}/evals_results"
LOG_FILE="${LOG_DIR}/gsm8k_${MODEL}_${MODE}_len${GEN_LENGTH}_autoresume_$(date +%F_%H-%M).log"

# ---- srun resource parameters (same as scripts/srun.sh) ----
SRUN_ACCOUNT="bdes-delta-gpu"
SRUN_TIME="02:30:00"
SRUN_PARTITION="gpuA100x4"
SRUN_NODES=1
SRUN_NTASKS=1
SRUN_GPUS=1
SRUN_MEM="32g"

get_done() {
    [[ -f "${RESULT_FILE}" ]] && grep -c '"answer"' "${RESULT_FILE}" || echo 0
}

mkdir -p "${LOG_DIR}"

# Tee all output to the log file from here on
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo " auto_resume.sh  model=${MODEL}  mode=${MODE}  started $(date)"
echo " Result file : ${RESULT_FILE}"
echo " Log         : ${LOG_FILE}"
echo "================================================================"

ATTEMPT=0

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    DONE=$(get_done)

    echo ""
    echo "[$(date)]  Attempt #${ATTEMPT}  |  Progress: ${DONE}/${TOTAL}"

    if [[ ${DONE} -ge ${TOTAL} ]]; then
        echo "[$(date)]  All ${TOTAL} samples complete. Exiting."
        break
    fi

    REMAINING=$((TOTAL - DONE))
    echo "[$(date)]  ${REMAINING} samples remaining. Requesting srun resources..."

    srun \
        -A "${SRUN_ACCOUNT}" \
        --time="${SRUN_TIME}" \
        --nodes="${SRUN_NODES}" \
        --ntasks="${SRUN_NTASKS}" \
        --partition="${SRUN_PARTITION}" \
        --gpus="${SRUN_GPUS}" \
        --mem="${SRUN_MEM}" \
        bash -c "source ${VENV} && cd ${ROOT_DIR} && GEN_LENGTH=${GEN_LENGTH} SAVE_DIR=${SAVE_DIR} OUTPUT_PATH=${OUTPUT_PATH} bash ${EVAL_SCRIPT} ${MODE}" \
    && SRUN_EXIT=0 || SRUN_EXIT=$?

    DONE=$(get_done)
    echo "[$(date)]  srun exited (code=${SRUN_EXIT})  |  Progress now: ${DONE}/${TOTAL}"

    if [[ ${DONE} -ge ${TOTAL} ]]; then
        echo "[$(date)]  Complete!"
        break
    fi

    echo "[$(date)]  Waiting 15s before re-queuing..."
    sleep 15
done

echo ""
echo "================================================================"
echo " auto_resume.sh finished $(date)  |  Final: $(get_done)/${TOTAL}"
echo " model=${MODEL}  mode=${MODE}  len=${GEN_LENGTH}"
echo "================================================================"
