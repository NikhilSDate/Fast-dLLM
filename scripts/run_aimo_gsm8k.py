#!/usr/bin/env python3
"""Run only the GSM8K experiments (LLaDA + Dream) with the AI-MO dataset.

Usage:
    python scripts/run_aimo_gsm8k.py

Dataset substitution:
  This script sets USE_AIMO_DATASET=1 to enable the sitecustomize patch, which
  transparently redirects all load_dataset("gsm8k") calls to the AI-MO dataset
  ("AI-MO/aimo-validation-aime", "default", split="train").

  See run_experiments.py docstring for details on the patch mechanism.

Note:
  To run GSM8K experiments with the original GSM8K dataset, use:
    python scripts/run_experiments.py
  The default behavior (without USE_AIMO_DATASET) preserves standard GSM8K.
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

# Filter for the AI-MO experiments (GSM8K-LLaDA-AIMO and GSM8K-Dream-AIMO)
selected = [e for e in EXPERIMENTS if e.experiment_name in ("GSM8K-LLaDA-AIMO", "GSM8K-Dream-AIMO")]

if not selected:
    raise SystemExit("No GSM8K-AIMO experiments found in EXPERIMENTS")

# Enable AI-MO dataset substitution for this run
import os
os.environ["USE_AIMO_DATASET"] = "1"

print(f"Running {len(selected)} GSM8K experiments (LLaDA + Dream) with AI-MO dataset")
run_all(selected)
