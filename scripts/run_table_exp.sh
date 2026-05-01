#!/usr/bin/env bash
# run_table_exp.sh — Run one GSM8K-table experiment with auto-resume.
#
# Usage (inside a tmux pane):
#   bash scripts/run_table_exp.sh <MODE> <NUM_FEWSHOT> <GEN_LENGTH>
#
#   MODE        : baseline | parallel | prefix-cache-parallel | dual-cache-parallel
#   NUM_FEWSHOT : 5 | 8
#   GEN_LENGTH  : 256 | 512 | 1024
#
# Examples:
#   bash scripts/run_table_exp.sh parallel 8 1024
#   bash scripts/run_table_exp.sh dual-cache-parallel 8 256
#   bash scripts/run_table_exp.sh baseline 8 1024
#
# Results land in:
#   results/GSM8K-table/llada/len<N>/<fs>shot/<mode>/rank_0.jsonl
#
# Each srun slot is 2.5 h; the script keeps re-queuing until all 1319
# samples are complete.

set -uo pipefail

MODE="${1:?Usage: $0 <MODE> <NUM_FEWSHOT> <GEN_LENGTH>}"
NUM_FEWSHOT="${2:?Usage: $0 <MODE> <NUM_FEWSHOT> <GEN_LENGTH>}"
GEN_LENGTH="${3:?Usage: $0 <MODE> <NUM_FEWSHOT> <GEN_LENGTH>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/u/xzhou12/fastdllm_venv/bin/activate"
EVAL_SCRIPT="scripts/reproduce_llada_gsm8k.sh"
TOTAL=1319

SAVE_DIR="${ROOT_DIR}/results/GSM8K-table/llada/len${GEN_LENGTH}/${NUM_FEWSHOT}shot"
OUTPUT_PATH="${ROOT_DIR}/evals_results/GSM8K-table/llada/len${GEN_LENGTH}/${NUM_FEWSHOT}shot"
RESULT_FILE="${SAVE_DIR}/${MODE}/rank_0.jsonl"

LOG_DIR="${ROOT_DIR}/evals_results/GSM8K-table-logs"
LOG_FILE="${LOG_DIR}/llada_${MODE}_${NUM_FEWSHOT}shot_len${GEN_LENGTH}_$(date +%F_%H-%M).log"

# srun resource parameters
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

mkdir -p "${SAVE_DIR}" "${OUTPUT_PATH}" "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo " run_table_exp.sh  started $(date)"
echo " mode=${MODE}  fewshot=${NUM_FEWSHOT}  gen_length=${GEN_LENGTH}"
echo " result : ${RESULT_FILE}"
echo " log    : ${LOG_FILE}"
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

    echo "[$(date)]  $((TOTAL - DONE)) remaining — requesting srun..."

    srun \
        -A "${SRUN_ACCOUNT}" \
        --time="${SRUN_TIME}" \
        --nodes="${SRUN_NODES}" \
        --ntasks="${SRUN_NTASKS}" \
        --partition="${SRUN_PARTITION}" \
        --gpus="${SRUN_GPUS}" \
        --mem="${SRUN_MEM}" \
        bash -c "source ${VENV} && cd ${ROOT_DIR} \
            && GEN_LENGTH=${GEN_LENGTH} NUM_FEWSHOT=${NUM_FEWSHOT} \
               SAVE_DIR=${SAVE_DIR} OUTPUT_PATH=${OUTPUT_PATH} \
               bash ${EVAL_SCRIPT} ${MODE}" \
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
echo " run_table_exp.sh finished $(date)"
echo " mode=${MODE}  fewshot=${NUM_FEWSHOT}  gen_length=${GEN_LENGTH}"
echo " Final: $(get_done)/${TOTAL}"
echo "================================================================"
