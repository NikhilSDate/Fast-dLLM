#!/usr/bin/env bash
# auto_resume.sh — Keep re-requesting srun resources until the eval is done.
#
# Usage (inside a tmux session):
#   bash scripts/auto_resume.sh [baseline|prefix-cache|parallel|prefix-cache-parallel]
#
# The underlying eval script already resumes from rank_0.jsonl automatically,
# so each srun call picks up exactly where the last one left off.

set -uo pipefail

MODE="${1:-baseline}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/u/xzhou12/fastdllm_venv/bin/activate"

# GSM8K test set size
TOTAL=1319

RESULT_FILE="${ROOT_DIR}/results/${MODE}/rank_0.jsonl"
LOG_DIR="${ROOT_DIR}/evals_results"
LOG_FILE="${LOG_DIR}/gsm8k_${MODE}_autoresume_$(date +%F_%H-%M).log"

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
echo " auto_resume.sh  mode=${MODE}  started $(date)"
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
        bash -c "source ${VENV} && cd ${ROOT_DIR} && bash scripts/reproduce_llada_gsm8k.sh ${MODE}" \
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
echo "================================================================"
