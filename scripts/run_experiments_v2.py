#!/usr/bin/env python3
"""
Single entrypoint for running Fast-dLLM v2 evaluation experiments.

Define experiment runs in the EXPERIMENTS list at the bottom of this file, then:

    python scripts/run_experiments_v2.py

Each run submits via srun and is retried automatically until lm-eval writes a
results JSON to the output directory. Run from a tmux session on the head node.

Unlike v1, v2 has no per-sample save file, so completion is detected by the
presence of lm-eval's results_*.json in the output directory.

Output paths:
    evals_results/v2/{experiment_name}/{tag}/   ← lm-eval JSON output
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

_print_lock = threading.Lock()


def _log(prefix: str, msg: str) -> None:
    with _print_lock:
        print(f"[{prefix}] {msg}", flush=True)

TASK_TOTAL: dict[str, int] = {
    "gsm8k": 1319,
    "minerva_math": 500,
    "ifeval": 541,
    "humaneval": 164
}


# ─── Config dataclasses ───────────────────────────────────────────────────────


@dataclass
class SrunConfig:
    account: str = "bdes-delta-gpu"
    time: str = "02:30:00"
    partition: str = "gpuA100x4"
    nodes: int = 1
    ntasks: int = 1
    gpus: int = 1
    mem: str = "32g"


@dataclass
class FastDLLMv2Config:
    """
    Parameters forwarded to Fast_dLLM_v2EvalHarness via --model_args.

    factor and threshold are mutually exclusive: when factor=True, threshold is
    omitted from model_args and factor-based decoding is used instead.
    """
    model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
    max_new_tokens: int = 2048
    bd_size: int = 32
    small_block_size: int = 8
    use_block_cache: bool = False
    threshold: float = 0.9
    show_speed: bool = True
    factor: bool = False
    factor_value: float = 1.0


@dataclass
class ExperimentRun:
    """
    A single evaluation run.

    tag: subdirectory under the experiment directory (e.g. "factor/f1p0").
    experiment_name: top-level grouping (e.g. "GSM8K-FastDLLMv2").
    task: lm-evaluation-harness task name.
    model: FastDLLMv2Config instance.
    batch_size: passed as --batch_size to lm-eval.
    num_fewshot: passed as --num_fewshot. Omitted when None.
    apply_chat_template: adds --apply_chat_template --fewshot_as_multiturn flags.
    """
    tag: str
    experiment_name: str
    task: str
    model: FastDLLMv2Config
    batch_size: int = 32
    num_fewshot: Optional[int] = None
    apply_chat_template: bool = True
    srun: SrunConfig = field(default_factory=SrunConfig)
    venv: str = "/u/ndate/venv/bin/activate"
    output_path: Optional[Path] = None


# ─── Path helpers ─────────────────────────────────────────────────────────────


def resolve_save_dir(run: ExperimentRun) -> Path:
    return ROOT_DIR / "results" / "v2" / run.experiment_name / run.tag


def resolve_output_path(run: ExperimentRun) -> Path:
    if run.output_path is not None:
        return run.output_path
    return ROOT_DIR / "evals_results" / "v2" / run.experiment_name / run.tag


def count_done(save_dir: Path) -> int:
    result_file = save_dir / "rank_0.jsonl"
    if not result_file.exists():
        return 0
    count = 0
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            if '"answer"' in line:
                count += 1
    return count


# ─── Model-args builder ───────────────────────────────────────────────────────


def _model_args(cfg: FastDLLMv2Config, save_dir: Path) -> str:
    parts = [
        f"model_path={cfg.model_path}",
        f"max_new_tokens={cfg.max_new_tokens}",
        f"bd_size={cfg.bd_size}",
        f"small_block_size={cfg.small_block_size}",
        f"show_speed={cfg.show_speed}",
        f"save_dir={save_dir}",
    ]
    if cfg.use_block_cache:
        parts.append("use_block_cache=True")
    if cfg.factor:
        parts.append("factor=True")
        parts.append(f"factor_value={cfg.factor_value}")
    else:
        parts.append(f"threshold={cfg.threshold}")
    return ",".join(parts)


# ─── Shell payload builder ────────────────────────────────────────────────────


def build_payload(run: ExperimentRun, save_dir: Path, output_path: Path) -> str:
    """Return the bash command string that srun will execute on the compute node."""
    model_args = _model_args(run.model, save_dir)
    fewshot_flag = f"--num_fewshot {run.num_fewshot} " if run.num_fewshot is not None else ""
    chat_flags = "--fewshot_as_multiturn --apply_chat_template " if run.apply_chat_template else ""

    eval_cmd = (
        f"accelerate launch eval.py "
        f"--tasks {run.task} "
        f"--batch_size {run.batch_size} "
        f"{fewshot_flag}"
        f"{chat_flags}"
        f"--confirm_run_unsafe_code "
        f"--model fast_dllm_v2 "
        f"--model_args '{model_args}' "
        f"--output_path {output_path} "
        f"--log_samples"
    )

    work_dir = ROOT_DIR / "v2"
    return (
        f"source {run.venv} && "
        f"export HF_ALLOW_CODE_EVAL=1 && "
        f"export HF_DATASETS_TRUST_REMOTE_CODE=true && "
        f"export LD_LIBRARY_PATH=/sw/rh9.4/python/miniforge3/lib:${{LD_LIBRARY_PATH:-}} && "
        f"export MASTER_ADDR=${{MASTER_ADDR:-localhost}} && "
        f"export MASTER_PORT=${{MASTER_PORT:-29500}} && "
        f"cd {work_dir} && "
        f"{eval_cmd}"
    )


# ─── Completion check ─────────────────────────────────────────────────────────


def is_complete(output_path: Path) -> bool:
    """Return True if lm-eval has written a results JSON anywhere under output_path."""
    return output_path.exists() and any(output_path.rglob("results_*.json"))


# ─── Per-experiment runner ────────────────────────────────────────────────────


def run_experiment(run: ExperimentRun) -> None:
    prefix = f"{run.experiment_name}/{run.tag}"
    save_dir = resolve_save_dir(run)
    output_path = resolve_output_path(run)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    total = TASK_TOTAL.get(run.task)
    if total is None:
        raise ValueError(f"Unknown task '{run.task}'. Add it to TASK_TOTAL.")

    _log(prefix, f"task={run.task} total={total}  model={run.model.model_path}")
    _log(prefix, f"save_dir={save_dir}")

    sr = run.srun
    srun_prefix = [
        "srun",
        "-A", sr.account,
        f"--time={sr.time}",
        f"--nodes={sr.nodes}",
        f"--ntasks={sr.ntasks}",
        f"--partition={sr.partition}",
        f"--gpus={sr.gpus}",
        f"--mem={sr.mem}",
        "bash", "-c",
    ]

    attempt = 0
    while True:
        attempt += 1
        done = count_done(save_dir)
        ts = time.strftime("%H:%M:%S")
        _log(prefix, f"{ts}  attempt #{attempt}  |  {done}/{total}")

        if done >= total and is_complete(output_path):
            _log(prefix, f"{ts}  complete ({done}/{total})")
            break

        payload = build_payload(run, save_dir, output_path)
        result = subprocess.run(srun_prefix + [payload])
        srun_exit = result.returncode

        done = count_done(save_dir)
        ts = time.strftime("%H:%M:%S")
        _log(prefix, f"{ts}  srun exited (code={srun_exit})  |  {done}/{total}")

        if done >= total and is_complete(output_path):
            _log(prefix, f"{ts}  complete!")
            break

        _log(prefix, f"{ts}  waiting 15s before re-queuing...")
        time.sleep(15)


def run_all(experiments: list[ExperimentRun], workers: int = 1) -> None:
    n = len(experiments)
    print(f"\nRunning {n} experiment(s) with {workers} parallel worker(s).\n")
    for exp in experiments:
        print(f"  • {exp.experiment_name}/{exp.tag}")
    print()

    running: set[str] = set()
    lock = threading.Lock()

    def _status_line() -> str:
        if running:
            return "running: " + ", ".join(sorted(running))
        return "running: (none)"

    def _run(run: ExperimentRun) -> None:
        label = f"{run.experiment_name}/{run.tag}"
        with lock:
            running.add(label)
            print(f"\n>>> STARTED  {label}  |  {_status_line()}", flush=True)
        try:
            run_experiment(run)
        finally:
            with lock:
                running.discard(label)
                print(f"\n>>> FINISHED {label}  |  {_status_line()}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run, exp): exp for exp in experiments}
        for future in as_completed(futures):
            exp = futures[future]
            exc = future.exception()
            if exc:
                label = f"{exp.experiment_name}/{exp.tag}"
                print(f"\n>>> FAILED   {label}: {exc}", flush=True)

    print("\nAll experiments complete.")


# ─── Experiment definitions ───────────────────────────────────────────────────
#
# Common patterns:
#
#   Threshold-based decoding:
#     FastDLLMv2Config(threshold=0.9)
#
#   Factor-based decoding (replaces threshold entirely):
#     FastDLLMv2Config(factor=True, factor_value=1.0)
#
#   Block cache:
#     FastDLLMv2Config(use_block_cache=True)
#
# Note: mmlu and gpqa use loglikelihood and are unaffected by threshold/factor.

EXPERIMENTS: list[ExperimentRun] = [

    # ── GSM8K — 0-shot ────────────────────────────────────────────────────────
    *[
        ExperimentRun(
            tag=f"threshold/t{str(t).replace('.', 'p')}",
            experiment_name="GSM8K-FastDLLMv2",
            task="gsm8k",
            num_fewshot=0,
            batch_size=32,
            model=FastDLLMv2Config(threshold=t),
        )
        for t in [0.9, 1.0]
    ],
    *[
        ExperimentRun(
            tag=f"factor/f{str(f).replace('.', 'p')}",
            experiment_name="GSM8K-FastDLLMv2",
            task="gsm8k",
            num_fewshot=0,
            batch_size=32,
            model=FastDLLMv2Config(factor=True, factor_value=f),
        )
        for f in [1.0]
    ],

    # ── MATH (minerva_math) — 0-shot ─────────────────────────────────────────
    *[
        ExperimentRun(
            tag=f"threshold/t{str(t).replace('.', 'p')}",
            experiment_name="MATH-FastDLLMv2",
            task="minerva_math",
            num_fewshot=0,
            batch_size=32,
            model=FastDLLMv2Config(threshold=t),
        )
        for t in [0.9, 1.0]
    ],
    *[
        ExperimentRun(
            tag=f"factor/f{str(f).replace('.', 'p')}",
            experiment_name="MATH-FastDLLMv2",
            task="minerva_math",
            num_fewshot=0,
            batch_size=32,
            model=FastDLLMv2Config(factor=True, factor_value=f),
        )
        for f in [1.0]
    ],

    # ── IFEval ────────────────────────────────────────────────────────────────
    ExperimentRun(
        tag="threshold/t0p9",
        experiment_name="IFEval-FastDLLMv2",
        task="ifeval",
        batch_size=32,
        model=FastDLLMv2Config(threshold=0.9),
    ),
    ExperimentRun(
        tag="factor/f1p0",
        experiment_name="IFEval-FastDLLMv2",
        task="ifeval",
        batch_size=32,
        model=FastDLLMv2Config(factor=True, factor_value=1.0),
    ),

    # ── MMLU — 5-shot (loglikelihood; threshold/factor irrelevant) ────────────
    ExperimentRun(
        tag="default",
        experiment_name="MMLU-FastDLLMv2",
        task="mmlu",
        num_fewshot=5,
        batch_size=1,
        model=FastDLLMv2Config(),
    ),

    # ── GPQA — n-shot (loglikelihood; threshold/factor irrelevant) ───────────
    ExperimentRun(
        tag="default",
        experiment_name="GPQA-FastDLLMv2",
        task="gpqa_main_n_shot",
        batch_size=1,
        model=FastDLLMv2Config(),
    ),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="number of experiments to run in parallel (default: 1)")
    args = parser.parse_args()

    run_all([
        ExperimentRun(
            tag="factor/f0p4-cache-block32-batch1",
            experiment_name="HumanEval-FastDLLMv2",
            task="humaneval",
            batch_size=1,
            model=FastDLLMv2Config(
                factor=True,
                factor_value=0.4,
                small_block_size=32,
                bd_size=32,
            ),
            srun=SrunConfig(time='00:20:00'),
        ),
    ], workers=args.jobs)


'''
Total tokens generated: 397227
Total time: 555.1s
Tokens/second: 715.5
Total NFE: 17716
Avg tokens/NFE: 22.42
fast_dllm_v2 (model_path=Efficient-Large-Model/Fast_dLLM_v2_1.5B,max_new_tokens=2048,bd_size=32,small_block_size=8,show_speed=True,save_dir=/projects/bdes/ndate/Fast-dLLM/results/v2/GSM8K-FastDLLMv2/factor/f1p0,use_block_cache=True,factor=True,factor_value=1.0), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 32
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.5512|±  |0.0137|
|     |       |strict-match    |     0|exact_match|↑  |0.0523|±  |0.0061|

[2026-05-01 02:40:44]  srun exited (code=0)  |  1319/1319
[2026-05-01 02:40:44]  Complete!

All experiments complete.
'''

'''
Total tokens generated: 386312
Total time: 651.0s
Tokens/second: 593.4
Total NFE: 21975
Avg tokens/NFE: 17.58
fast_dllm_v2 (model_path=Efficient-Large-Model/Fast_dLLM_v2_1.5B,max_new_tokens=2048,bd_size=32,small_block_size=8,show_speed=True,save_dir=/projects/bdes/ndate/Fast-dLLM/results/v2/GSM8K-FastDLLMv2/threshold/0.9,use_block_cache=True,threshold=0.9), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 32
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.6315|±  |0.0133|
|     |       |strict-match    |     0|exact_match|↑  |0.0508|±  |0.0060|

[2026-05-01 03:08:54]  srun exited (code=0)  |  1319/1319
[2026-05-01 03:08:54]  Complete!

All experiments complete.
'''

'''
Total tokens generated: 387547
Total time: 359.9s
Tokens/second: 1077.0
Total NFE: 8373
Avg tokens/NFE: 46.29
fast_dllm_v2 (model_path=Efficient-Large-Model/Fast_dLLM_v2_1.5B,max_new_tokens=2048,bd_size=32,small_block_size=32,show_speed=True,save_dir=/projects/bdes/ndate/Fast-dLLM/results/v2/GSM8K-FastDLLMv2/threshold/cache-factor-block32,use_block_cache=True,factor=True,factor_value=1), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 64
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.5353|±  |0.0137|
|     |       |strict-match    |     0|exact_match|↑  |0.0553|±  |0.0063|

[2026-05-01 16:59:23]  srun exited (code=0)  |  1319/1319
[2026-05-01 16:59:23]  Complete!

All experiments complete.
'''

'''
Total tokens generated: 385400
Total time: 396.8s
Tokens/second: 971.3
Total NFE: 9929
Avg tokens/NFE: 38.82
fast_dllm_v2 (model_path=Efficient-Large-Model/Fast_dLLM_v2_1.5B,max_new_tokens=2048,bd_size=32,small_block_size=32,show_speed=True,save_dir=/projects/bdes/ndate/Fast-dLLM/results/v2/GSM8K-FastDLLMv2/threshold/cache-0.9-block32,use_block_cache=True,threshold=0.9), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 64
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.5883|±  |0.0136|
|     |       |strict-match    |     0|exact_match|↑  |0.0500|±  |0.0060|

[2026-05-01 16:48:45]  srun exited (code=0)  |  1319/1319
[2026-05-01 16:48:45]  Complete!

All experiments complete.
'''

'''
HumanEval: with factor: 32/32: 0.29878048780487804, 33.512995313165746 tokens/NFE
           with threshold: 0.3719512195121951
'''


