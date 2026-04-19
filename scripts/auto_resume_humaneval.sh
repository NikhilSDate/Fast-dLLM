#!/usr/bin/env bash
set -uo pipefail

# auto_resume_humaneval.sh — Keep re-requesting srun resources until evaluation is done.
#
# Usage (inside a tmux session):
#   bash scripts/auto_resume_humaneval.sh [MODEL] [MODE]
#
#   MODEL : llada (default) | dream
#   MODE  : baseline | prefix-cache | parallel | prefix-cache-parallel
#           (dream also supports: dual-cache-parallel)
#
# Examples:
#   bash scripts/auto_resume_humaneval.sh llada baseline
#   bash scripts/auto_resume_humaneval.sh dream parallel
#   GEN_LENGTH=512 bash scripts/auto_resume_humaneval.sh dream dual-cache-parallel

MODEL="${1:-llada}"
MODE="${2:-baseline}"
GEN_LENGTH="${GEN_LENGTH:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/u/jvancosampedro/fastdllm_venv/bin/activate"

# HumanEval test set size
TOTAL=164

# Resolve per-model result path and eval script
case "${MODEL}" in
llada)
    EVAL_SCRIPT="scripts/reproduce_llada_humaneval.sh"
    DEFAULT_GEN_LENGTH=256
    RESULT_ROOT="${ROOT_DIR}/results/humaneval"
    case "${MODE}" in
    baseline|prefix-cache|parallel|prefix-cache-parallel)
        ;;
    *)
        echo "Unknown MODE for llada: ${MODE}"
        echo "Expected one of: baseline, prefix-cache, parallel, prefix-cache-parallel"
        exit 1
        ;;
    esac
    ;;
dream)
    EVAL_SCRIPT="scripts/reproduce_dream_humaneval.sh"
    DEFAULT_GEN_LENGTH=256
    RESULT_ROOT="${ROOT_DIR}/results/HumanEval-Dream"
    case "${MODE}" in
    baseline|prefix-cache|parallel|prefix-cache-parallel|dual-cache-parallel)
        ;;
    *)
        echo "Unknown MODE for dream: ${MODE}"
        echo "Expected one of: baseline, prefix-cache, parallel, prefix-cache-parallel, dual-cache-parallel"
        exit 1
        ;;
    esac
    ;;
*)
    echo "Unknown MODEL: ${MODEL}. Expected: llada | dream"
    exit 1
    ;;
esac

GEN_LENGTH="${GEN_LENGTH:-${DEFAULT_GEN_LENGTH}}"
RESULT_FILE="${RESULT_ROOT}/len${GEN_LENGTH}/${MODE}/rank_0.jsonl"

LOG_DIR="${ROOT_DIR}/evals_results"
LOG_FILE="${LOG_DIR}/humaneval_${MODEL}_${MODE}_len${GEN_LENGTH}_autoresume_$(date +%F_%H-%M).log"

# ---- srun resource parameters ----
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
echo " auto_resume_humaneval.sh  model=${MODEL}  mode=${MODE}  started $(date)"
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
        bash -c "source ${VENV} && cd ${ROOT_DIR} && GEN_LENGTH=${GEN_LENGTH} bash ${EVAL_SCRIPT} ${MODE}" \
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
echo " auto_resume_humaneval.sh finished $(date)  |  Final: $(get_done)/${TOTAL}"
echo " model=${MODEL}  mode=${MODE}  len=${GEN_LENGTH}"
echo "================================================================"