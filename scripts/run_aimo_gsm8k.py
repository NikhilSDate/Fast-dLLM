#!/usr/bin/env python3
"""Run only the GSM8K experiments (LLaDA + Dream) but with the AIM-O dataset.

Usage:
    export PYTHONPATH=$PWD/experiments/patch:$PYTHONPATH
    python scripts/run_aimo_gsm8k.py

This script loads `scripts/run_experiments.py` and runs the subset of
experiments that produced the first two tables in results.md.
"""
from __future__ import annotations

import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Execute the run_experiments module and grab its globals
g = runpy.run_path(str(ROOT / "scripts" / "run_experiments.py"))
EXPERIMENTS = g.get("EXPERIMENTS", [])
run_all = g.get("run_all")

if run_all is None:
    raise SystemExit("Could not load run_all from scripts/run_experiments.py")

# Filter for the GSM8K experiments for both LLaDA and Dream
selected = [e for e in EXPERIMENTS if e.task == "gsm8k" and e.experiment_name in ("GSM8K-LLaDA", "GSM8K-Dream")]

if not selected:
    raise SystemExit("No GSM8K LLaDA/Dream experiments found in EXPERIMENTS")

print(f"Running {len(selected)} GSM8K experiments (LLaDA + Dream) with AIM-O dataset")
run_all(selected)
