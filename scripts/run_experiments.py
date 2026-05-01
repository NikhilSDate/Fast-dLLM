#!/usr/bin/env python3
"""
Single entrypoint for running Fast-dLLM evaluation experiments.

Define experiment runs in the EXPERIMENTS list at the bottom of this file, then:

    python scripts/run_experiments.py

Each run loops under srun until all samples are complete, then moves on to the
next run automatically. Run this script from inside a tmux session on the head
node — no further intervention needed.

Output paths follow the same naming conventions as the existing shell scripts:
    results/{experiment_name}/len{N}/{tag}/rank_0.jsonl   ← resumable results
    evals_results/{experiment_name}/len{N}/{tag}/          ← lm-eval JSON output
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent

TASK_TOTAL: dict[str, int] = {
    "gsm8k": 1319,
    "humaneval": 164,
}

# ─── Config dataclasses ──────────────────────────────────────────────────────


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
class LLaDaConfig:
    """
    Parameters for llada/eval_llada.py (--model llada_dist).

    steps: number of diffusion steps. Defaults to gen_length (baseline / one token
           per step). For parallel decoding set steps = gen_length // block_length.
    threshold: confidence threshold for parallel decoding. None = disabled.
    use_cache: enable prefix KV-cache.
    dual_cache: enable dual KV-cache (requires use_cache=True).
    variable_cache: enable variable-length prefix cache.
    """
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    gen_length: int = 512
    block_length: int = 32
    steps: Optional[int] = None        # defaults to gen_length when None
    threshold: Optional[float] = None
    use_cache: bool = False
    dual_cache: bool = False
    variable_cache: bool = False
    streaming: bool = False
    show_speed: bool = True


@dataclass
class DreamConfig:
    """
    Parameters for dream/eval.py (--model dream).

    diffusion_steps: number of denoising steps. Defaults to max_new_tokens when None.
    alg: "entropy" (baseline) or "confidence_threshold" (parallel decoding).
    threshold: confidence threshold used when alg="confidence_threshold".
    use_cache: enable prefix KV-cache.
    dual_cache: enable dual KV-cache (requires use_cache=True).
    escape_until: pass stop tokens through to the model (needed for HumanEval).
    """
    pretrained: str = "Dream-org/Dream-v0-Base-7B"
    max_new_tokens: int = 256
    diffusion_steps: Optional[int] = None  # defaults to max_new_tokens when None
    alg: str = "entropy"
    threshold: float = 0.9
    use_cache: bool = False
    dual_cache: bool = False
    add_bos_token: bool = True
    escape_until: bool = False


@dataclass
class ExperimentRun:
    """
    A single evaluation run.

    tag: subdirectory name under the len{N} directory, e.g. "baseline",
         "prefix-cache", "threshold/t0p9", "fixed/4tok-per-step".
    experiment_name: top-level grouping that maps to the directory prefix, e.g.
         "GSM8K-LLaDA", "Figure5/GSM8K", "HumanEval-Dream".
    task: lm-evaluation-harness task name ("gsm8k" or "humaneval").
    model: LLaDaConfig or DreamConfig instance.
    num_fewshot: passed as --num_fewshot. Omitted when None (e.g. HumanEval).
    srun: SLURM resource parameters.
    venv: path to the virtualenv activate script on the compute node.
    save_dir / output_path: override the auto-derived paths when set.
    """
    tag: str
    experiment_name: str
    task: str
    model: Union[LLaDaConfig, DreamConfig]
    num_fewshot: Optional[int] = None
    srun: SrunConfig = field(default_factory=SrunConfig)
    venv: str = "/u/jvancosampedro/fastdllm_venv/bin/activate"
    save_dir: Optional[Path] = None
    output_path: Optional[Path] = None


# ─── Path helpers ─────────────────────────────────────────────────────────────


def _gen_length(model: Union[LLaDaConfig, DreamConfig]) -> int:
    if isinstance(model, LLaDaConfig):
        return model.gen_length
    return model.max_new_tokens


def resolve_save_dir(run: ExperimentRun) -> Path:
    if run.save_dir is not None:
        return run.save_dir
    return ROOT_DIR / "results" / run.experiment_name / f"len{_gen_length(run.model)}" / run.tag


def resolve_output_path(run: ExperimentRun) -> Path:
    if run.output_path is not None:
        return run.output_path
    return ROOT_DIR / "evals_results" / run.experiment_name / f"len{_gen_length(run.model)}" / run.tag


# ─── Model-args builders ──────────────────────────────────────────────────────


def _llada_model_args(cfg: LLaDaConfig, save_dir: Path) -> str:
    steps = cfg.steps if cfg.steps is not None else cfg.gen_length
    parts = [
        f"model_path={cfg.model_path}",
        f"gen_length={cfg.gen_length}",
        f"steps={steps}",
        f"block_length={cfg.block_length}",
        f"show_speed={cfg.show_speed}",
        f"save_dir={save_dir}",
    ]
    if cfg.threshold is not None:
        parts.append(f"threshold={cfg.threshold}")
    if cfg.use_cache:
        parts.append("use_cache=True")
    if cfg.dual_cache:
        parts.append("dual_cache=True")
    if cfg.variable_cache:
        parts.append("variable_cache=True")
    if cfg.streaming:
        parts.append("streaming=True")
    return ",".join(parts)


def _dream_model_args(cfg: DreamConfig, save_dir: Path) -> str:
    diffusion_steps = cfg.diffusion_steps if cfg.diffusion_steps is not None else cfg.max_new_tokens
    parts = [
        f"pretrained={cfg.pretrained}",
        f"max_new_tokens={cfg.max_new_tokens}",
        f"add_bos_token={str(cfg.add_bos_token).lower()}",
        f"save_dir={save_dir}",
        f"diffusion_steps={diffusion_steps}",
        f"alg={cfg.alg}",
        f"threshold={cfg.threshold}",
    ]
    if cfg.use_cache:
        parts.append("use_cache=true")
    if cfg.dual_cache:
        parts.append("dual_cache=true")
    if cfg.escape_until:
        parts.append("escape_until=true")
    return ",".join(parts)


# ─── Shell payload builder ────────────────────────────────────────────────────


def build_payload(run: ExperimentRun, save_dir: Path, output_path: Path) -> str:
    """Return the bash command string that srun will execute on the compute node."""
    model = run.model

    if isinstance(model, LLaDaConfig):
        work_dir = ROOT_DIR
        model_args = _llada_model_args(model, save_dir)
        fewshot_flag = f"--num_fewshot {run.num_fewshot} " if run.num_fewshot is not None else ""
        eval_cmd = (
            f"accelerate launch llada/eval_llada.py "
            f"--tasks {run.task} "
            f"{fewshot_flag}"
            f"--confirm_run_unsafe_code "
            f"--model llada_dist "
            f"--model_args '{model_args}' "
            f"--output_path {output_path} "
            f"--log_samples"
        )
    else:
        work_dir = ROOT_DIR / "dream"
        model_args = _dream_model_args(model, save_dir)
        fewshot_flag = f"--num_fewshot {run.num_fewshot} " if run.num_fewshot is not None else ""
        eval_cmd = (
            f"accelerate launch eval.py "
            f"--model dream "
            f"--model_args '{model_args}' "
            f"--tasks {run.task} "
            f"{fewshot_flag}"
            f"--batch_size 1 "
            f"--confirm_run_unsafe_code "
            f"--output_path {output_path} "
            f"--log_samples"
        )

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


# ─── Progress check ───────────────────────────────────────────────────────────


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


# ─── Per-experiment runner ────────────────────────────────────────────────────


def run_experiment(run: ExperimentRun) -> None:
    save_dir = resolve_save_dir(run)
    output_path = resolve_output_path(run)

    total = TASK_TOTAL.get(run.task)
    if total is None:
        raise ValueError(f"Unknown task '{run.task}'. Add it to TASK_TOTAL.")

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    model_type = type(run.model).__name__
    model_id = run.model.pretrained if isinstance(run.model, DreamConfig) else run.model.model_path

    print()
    print("=" * 70)
    print(f"  {run.experiment_name}/{run.tag}")
    print(f"  task      : {run.task}  (total={total})")
    print(f"  model     : {model_type}  {model_id}")
    print(f"  save_dir  : {save_dir}")
    print(f"  output    : {output_path}")
    print("=" * 70)

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
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]  Attempt #{attempt}  |  {done}/{total}", flush=True)

        if done >= total:
            print(f"[{ts}]  Complete ({done}/{total}).", flush=True)
            break

        payload = build_payload(run, save_dir, output_path)
        result = subprocess.run(srun_prefix + [payload])
        srun_exit = result.returncode

        done = count_done(save_dir)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]  srun exited (code={srun_exit})  |  {done}/{total}", flush=True)

        if done >= total:
            print(f"[{ts}]  Complete!", flush=True)
            break

        print(f"[{ts}]  Waiting 15s before re-queuing...", flush=True)
        time.sleep(15)


def run_all(experiments: list[ExperimentRun]) -> None:
    n = len(experiments)
    print(f"\nRunning {n} experiment(s) sequentially.\n")
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{n}]  {exp.experiment_name}/{exp.tag}", flush=True)
        run_experiment(exp)
    print("\nAll experiments complete.")


# ─── Experiment definitions ───────────────────────────────────────────────────
#
# Edit EXPERIMENTS to define what you want to run.  Each ExperimentRun maps
# directly to one entry in the existing shell scripts.
#
# Common patterns:
#
#   Baseline (one token per step, no cache):
#     LLaDaConfig(gen_length=512, steps=512, block_length=32)
#
#   Prefix-cache only:
#     LLaDaConfig(..., use_cache=True)
#
#   Parallel decoding (confidence threshold):
#     LLaDaConfig(gen_length=512, steps=16, block_length=32, threshold=0.9)
#     # steps = gen_length // block_length for "full parallel" block decoding
#
#   Dual-cache parallel:
#     LLaDaConfig(..., steps=16, use_cache=True, dual_cache=True, threshold=0.9)
#
#   Dream baseline:
#     DreamConfig(max_new_tokens=256, diffusion_steps=256, alg="entropy")
#
#   Dream parallel:
#     DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9)

EXPERIMENTS: list[ExperimentRun] = [

    # ── LLaDA GSM8K — 5-shot ────────────────────────────────────────────────
    ExperimentRun(
        tag="baseline",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        model=LLaDaConfig(gen_length=512, steps=512, block_length=32),
    ),
    ExperimentRun(
        tag="prefix-cache",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        model=LLaDaConfig(gen_length=512, steps=512, block_length=32, use_cache=True),
    ),
    ExperimentRun(
        tag="parallel",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        model=LLaDaConfig(gen_length=512, steps=16, block_length=32, threshold=0.9),
    ),
    ExperimentRun(
        tag="prefix-cache-parallel",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        model=LLaDaConfig(gen_length=512, steps=16, block_length=32, use_cache=True, threshold=0.9),
    ),
    ExperimentRun(
        tag="dual-cache-parallel",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        model=LLaDaConfig(gen_length=512, steps=16, block_length=32, use_cache=True, dual_cache=True, threshold=0.9),
    ),
    ExperimentRun(
        tag="streaming-prefix-cache",
        experiment_name="GSM8K-LLaDA",
        task="gsm8k",
        num_fewshot=5,
        # steps=32 is steps-per-block (gen_length=256 / block_length=32 = 8 blocks;
        # same per-block budget as the prefix-cache baseline at this gen_length).
        model=LLaDaConfig(gen_length=256, steps=32, block_length=32, use_cache=True, streaming=True),
    ),

    # ── Dual-cache block-size sweep (GSM8K, 5-shot, gen_length=256) ─────────
    # steps defaults to gen_length=256; divisible by num_blocks for all sizes below.
    *[
        ExperimentRun(
            tag=f"dual-cache/block{bl}",
            experiment_name="GSM8K-LLaDA",
            task="gsm8k",
            num_fewshot=5,
            model=LLaDaConfig(gen_length=256, block_length=bl, use_cache=True, dual_cache=True),
        )
        for bl in [8, 16, 64]
    ],

    # ── Figure 5 — LLaDA threshold sweep (GSM8K, 5-shot) ───────────────────
    *[
        ExperimentRun(
            tag=f"threshold/t{str(t).replace('.', 'p')}",
            experiment_name="Figure5/GSM8K",
            task="gsm8k",
            num_fewshot=5,
            model=LLaDaConfig(gen_length=512, steps=512, block_length=32, threshold=t),
        )
        for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ],

    # ── Figure 5 — LLaDA fixed-step baselines ───────────────────────────────
    *[
        ExperimentRun(
            tag=f"fixed/{tok}tok-per-step",
            experiment_name="Figure5/GSM8K",
            task="gsm8k",
            num_fewshot=5,
            model=LLaDaConfig(gen_length=512, steps=512 // tok, block_length=32),
        )
        for tok in [1, 2, 4, 8]
    ],

    # ── Dream GSM8K — 5-shot ────────────────────────────────────────────────
    ExperimentRun(
        tag="baseline",
        experiment_name="GSM8K-Dream",
        task="gsm8k",
        num_fewshot=5,
        model=DreamConfig(max_new_tokens=256, diffusion_steps=256, alg="entropy"),
    ),
    ExperimentRun(
        tag="prefix-cache",
        experiment_name="GSM8K-Dream",
        task="gsm8k",
        num_fewshot=5,
        model=DreamConfig(max_new_tokens=256, diffusion_steps=256, alg="entropy", use_cache=True),
    ),
    ExperimentRun(
        tag="parallel",
        experiment_name="GSM8K-Dream",
        task="gsm8k",
        num_fewshot=5,
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9),
    ),
    ExperimentRun(
        tag="prefix-cache-parallel",
        experiment_name="GSM8K-Dream",
        task="gsm8k",
        num_fewshot=5,
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9, use_cache=True),
    ),
    ExperimentRun(
        tag="dual-cache-parallel",
        experiment_name="GSM8K-Dream",
        task="gsm8k",
        num_fewshot=5,
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9, use_cache=True, dual_cache=True),
    ),

    # ── LLaDA HumanEval ─────────────────────────────────────────────────────
    ExperimentRun(
        tag="baseline",
        experiment_name="humaneval",
        task="humaneval",
        model=LLaDaConfig(gen_length=256, steps=256, block_length=32),
    ),
    ExperimentRun(
        tag="prefix-cache",
        experiment_name="humaneval",
        task="humaneval",
        model=LLaDaConfig(gen_length=256, steps=256, block_length=32, use_cache=True),
    ),
    ExperimentRun(
        tag="parallel",
        experiment_name="humaneval",
        task="humaneval",
        model=LLaDaConfig(gen_length=256, steps=8, block_length=32, threshold=0.9),
    ),
    ExperimentRun(
        tag="prefix-cache-parallel",
        experiment_name="humaneval",
        task="humaneval",
        model=LLaDaConfig(gen_length=256, steps=8, block_length=32, use_cache=True, threshold=0.9),
    ),

    # ── Dream HumanEval ─────────────────────────────────────────────────────
    ExperimentRun(
        tag="baseline",
        experiment_name="HumanEval-Dream",
        task="humaneval",
        model=DreamConfig(max_new_tokens=256, diffusion_steps=256, alg="entropy", escape_until=True),
    ),
    ExperimentRun(
        tag="prefix-cache",
        experiment_name="HumanEval-Dream",
        task="humaneval",
        model=DreamConfig(max_new_tokens=256, diffusion_steps=256, alg="entropy", use_cache=True, escape_until=True),
    ),
    ExperimentRun(
        tag="parallel",
        experiment_name="HumanEval-Dream",
        task="humaneval",
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9, escape_until=True),
    ),
    ExperimentRun(
        tag="prefix-cache-parallel",
        experiment_name="HumanEval-Dream",
        task="humaneval",
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9, use_cache=True, escape_until=True),
    ),
    ExperimentRun(
        tag="dual-cache-parallel",
        experiment_name="HumanEval-Dream",
        task="humaneval",
        model=DreamConfig(max_new_tokens=256, diffusion_steps=8, alg="confidence_threshold", threshold=0.9, use_cache=True, dual_cache=True, escape_until=True),
    ),
]


if __name__ == "__main__":
    experiments = [
        # ── Streaming prefix-cache ────────────────────────────────────────────
        ExperimentRun(
            tag="streaming-prefix-cache",
            experiment_name="GSM8K-LLaDA",
            task="gsm8k",
            num_fewshot=5,
            model=LLaDaConfig(gen_length=256, steps=32, block_length=32, use_cache=True, streaming=True),
        ),

        # ── Dual-cache block-size sweep ───────────────────────────────────────
        *[
            ExperimentRun(
                tag=f"dual-cache/block{bl}",
                experiment_name="GSM8K-LLaDA",
                task="gsm8k",
                num_fewshot=5,
                model=LLaDaConfig(gen_length=256, block_length=bl, use_cache=True, dual_cache=True),
            )
            for bl in [8]
        ],
    ]

    run_all(experiments)
