#!/usr/bin/env bash
# reproduce_gsm8k_table.sh — Reproduce the GSM8K accuracy + speed table.
#
# The table has 4 columns × 2 fewshot settings = 8 runs total:
#
#   Setting | LLaDA (baseline) | No Cache (parallel) | PrefixCache | DualCache
#   --------|------------------|---------------------|-------------|----------
#   5-shot  |  llada/baseline  |  llada/parallel     | llada/prefix-cache-parallel | llada/dual-cache-parallel
#   8-shot  |       "          |       "             |      "      |       "
#
# Usage (inside a tmux session):
#   bash scripts/reproduce_gsm8k_table.sh
#
# Optional overrides:
#   GEN_LENGTH=512  (default 512 for LLaDA, 256 for Dream)
#   LLADA_GEN_LENGTH=512
#   DREAM_GEN_LENGTH=256
#
# Each srun slot is 2.5 h; the script auto-requeues until every sample is done.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/u/xzhou12/fastdllm_venv/bin/activate"

TOTAL=1319   # GSM8K test-set size

# srun resource parameters
SRUN_ACCOUNT="bdes-delta-gpu"
SRUN_TIME="02:30:00"
SRUN_PARTITION="gpuA100x4"
SRUN_NODES=1
SRUN_NTASKS=1
SRUN_GPUS=1
SRUN_MEM="32g"

LLADA_GEN_LENGTH="${LLADA_GEN_LENGTH:-${GEN_LENGTH:-1024}}"
DREAM_GEN_LENGTH="${DREAM_GEN_LENGTH:-${GEN_LENGTH:-1024}}"

LOG_DIR="${ROOT_DIR}/evals_results/GSM8K-table-logs"
mkdir -p "${LOG_DIR}"

# ── Helpers ─────────────────────────────────────────────────────────────────

get_done() {
    local result_file="$1"
    [[ -f "${result_file}" ]] && grep -c '"answer"' "${result_file}" || echo 0
}

# run_until_done MODEL MODE NUM_FEWSHOT
#   Loops srun until all TOTAL samples are completed.
run_until_done() {
    local model="$1"        # llada | dream
    local mode="$2"         # baseline | parallel | prefix-cache-parallel | dual-cache-parallel
    local num_fewshot="$3"  # 5 | 8

    local gen_length eval_script result_file save_dir output_path

    case "${model}" in
    llada)
        gen_length="${LLADA_GEN_LENGTH}"
        eval_script="scripts/reproduce_llada_gsm8k.sh"
        # Note: the underlying script appends /${mode} to save_dir internally,
        # so we must NOT include mode here.
        save_dir="${ROOT_DIR}/results/GSM8K-table/llada/len${gen_length}/${num_fewshot}shot"
        output_path="${ROOT_DIR}/evals_results/GSM8K-table/llada/len${gen_length}/${num_fewshot}shot"
        result_file="${save_dir}/${mode}/rank_0.jsonl"
        ;;
    dream)
        gen_length="${DREAM_GEN_LENGTH}"
        eval_script="scripts/reproduce_dream_gsm8k.sh"
        save_dir="${ROOT_DIR}/results/GSM8K-table/dream/len${gen_length}/${num_fewshot}shot"
        output_path="${ROOT_DIR}/evals_results/GSM8K-table/dream/len${gen_length}/${num_fewshot}shot"
        result_file="${save_dir}/${mode}/rank_0.jsonl"
        ;;
    *)
        echo "[ERROR] Unknown model: ${model}"
        return 1
        ;;
    esac

    mkdir -p "${save_dir}" "${output_path}"

    local log_file="${LOG_DIR}/${model}_${mode}_${num_fewshot}shot_len${gen_length}.log"
    local attempt=0

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " Starting: model=${model}  mode=${mode}  fewshot=${num_fewshot}  len=${gen_length}"
    echo " Result  : ${result_file}"
    echo " Log     : ${log_file}"
    echo "════════════════════════════════════════════════════════════════"

    while true; do
        attempt=$((attempt + 1))
        local done
        done=$(get_done "${result_file}")

        echo "[$(date)]  ${model}/${mode}/${num_fewshot}shot  Attempt #${attempt}  |  ${done}/${TOTAL}"

        if [[ ${done} -ge ${TOTAL} ]]; then
            echo "[$(date)]  Complete (${done}/${TOTAL})."
            break
        fi

        echo "[$(date)]  Requesting srun..."

        srun \
            -A "${SRUN_ACCOUNT}" \
            --time="${SRUN_TIME}" \
            --nodes="${SRUN_NODES}" \
            --ntasks="${SRUN_NTASKS}" \
            --partition="${SRUN_PARTITION}" \
            --gpus="${SRUN_GPUS}" \
            --mem="${SRUN_MEM}" \
            bash -c "source ${VENV} && cd ${ROOT_DIR} \
                && GEN_LENGTH=${gen_length} NUM_FEWSHOT=${num_fewshot} \
                   SAVE_DIR=${save_dir} OUTPUT_PATH=${output_path} \
                   bash ${eval_script} ${mode}" \
            >> "${log_file}" 2>&1 \
        && SRUN_EXIT=0 || SRUN_EXIT=$?

        done=$(get_done "${result_file}")
        echo "[$(date)]  srun exited (code=${SRUN_EXIT})  |  Progress now: ${done}/${TOTAL}"

        if [[ ${done} -ge ${TOTAL} ]]; then
            echo "[$(date)]  Complete!"
            break
        fi

        echo "[$(date)]  Waiting 15 s before re-queuing..."
        sleep 15
    done
}

# ── Table of runs ────────────────────────────────────────────────────────────
#
# Each entry: "MODEL MODE NUM_FEWSHOT"
# Column mapping:
#   LLaDA (baseline)        → llada  baseline              5 / 8
#   No Cache (parallel)     → llada  parallel              5 / 8
#   PrefixCache             → llada  prefix-cache-parallel  5 / 8
#   DualCache               → llada  dual-cache-parallel   5 / 8

RUNS=(
    "llada  baseline              5"
    "llada  baseline              8"
    "llada  parallel              5"
    "llada  parallel              8"
    "llada  prefix-cache-parallel 5"
    "llada  prefix-cache-parallel 8"
    "llada  dual-cache-parallel   5"
    "llada  dual-cache-parallel   8"
)

echo "================================================================"
echo " reproduce_gsm8k_table.sh  started $(date)"
echo " ${#RUNS[@]} runs queued"
echo "================================================================"

for run in "${RUNS[@]}"; do
    read -r mdl mode fs <<< "${run}"
    run_until_done "${mdl}" "${mode}" "${fs}"
done

echo ""
echo "================================================================"
echo " All runs finished $(date)"
echo " Printing summary table..."
echo "================================================================"

# ── Summary table ────────────────────────────────────────────────────────────

python3 - <<'PYEOF'
import json, glob, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE_EVALS  = os.path.join(ROOT, "evals_results", "GSM8K-table")
BASE_RESULTS = os.path.join(ROOT, "results",      "GSM8K-table")

def get_acc(model, mode, fewshot, gen_length):
    # Directory structure: BASE_EVALS/{model}/len{N}/{fewshot}shot/{mode}/*/results_*.json
    pattern = os.path.join(BASE_EVALS, model, f"len{gen_length}",
                           f"{fewshot}shot", mode, "*", "results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        d = json.load(f)
    r = d.get("results", {}).get("gsm8k", {})
    acc = r.get("exact_match,flexible-extract", float("nan"))
    return round(acc * 100, 1)

def get_speed(model, mode, fewshot, gen_length):
    """Compute tokens/s = sum(tokens) / sum(elapsed) across rank_0.jsonl."""
    # Directory structure: BASE_RESULTS/{model}/len{N}/{fewshot}shot/{mode}/rank_0.jsonl
    path = os.path.join(BASE_RESULTS, model, f"len{gen_length}",
                        f"{fewshot}shot", mode, "rank_0.jsonl")
    if not os.path.exists(path):
        return None
    total_tokens = 0.0
    total_elapsed = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total_tokens  += rec.get("tokens",  0)
                total_elapsed += rec.get("elapsed", 0)
            except json.JSONDecodeError:
                continue
    if total_elapsed == 0:
        return None
    return round(total_tokens / total_elapsed, 1)

llada_len = int(os.environ.get("LLADA_GEN_LENGTH", os.environ.get("GEN_LENGTH", 512)))
dream_len  = int(os.environ.get("DREAM_GEN_LENGTH", os.environ.get("GEN_LENGTH", 256)))

cols = [
    ("LLaDA",       "llada", "baseline",              llada_len),
    ("No Cache",    "llada", "parallel",              llada_len),
    ("PrefixCache", "llada", "prefix-cache-parallel", llada_len),
    ("DualCache",   "llada", "dual-cache-parallel",   llada_len),
]

COL_W = 16
header = f"{'Setting':<10}" + "".join(f"{c[0]:>{COL_W}}" for c in cols)
sep    = "-" * len(header)

print()
print(header)
print(sep)

# Compute baseline speed for speedup calculation (5-shot and 8-shot separately)
baseline_speed = {}
for fs in [5, 8]:
    spd = get_speed("llada", "baseline", fs, llada_len)
    baseline_speed[fs] = spd

for fs in [5, 8]:
    acc_row   = f"{fs}-shot    "
    speed_row = f"{'':10}"  # indent aligned with Setting column

    for col_name, model, mode, gen_len in cols:
        acc  = get_acc(model, mode, fs, gen_len)
        spd  = get_speed(model, mode, fs, gen_len)
        base = baseline_speed[fs]

        acc_cell = f"{acc:.1f}" if acc is not None else "N/A"

        if spd is not None:
            if base and base > 0:
                speedup = spd / base
                spd_cell = f"{spd:.1f} ({speedup:.1f}x)"
            else:
                spd_cell = f"{spd:.1f}"
        else:
            spd_cell = "N/A"

        acc_row   += f"{acc_cell:>{COL_W}}"
        speed_row += f"{spd_cell:>{COL_W}}"

    print(acc_row)
    print(speed_row)
    if fs != [5, 8][-1]:
        print()

print(sep)
PYEOF
