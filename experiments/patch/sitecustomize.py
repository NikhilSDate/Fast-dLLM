"""Startup patch: override `datasets.load_dataset` for GSM8K tasks.

Place this directory at the front of `PYTHONPATH` when running evaluations so
Python imports this `sitecustomize` at startup and patches the loader.
"""
from __future__ import annotations

try:
    import datasets as _datasets
except Exception:
    # If datasets isn't available at import time, nothing to patch now.
    _datasets = None

_orig_load_dataset = None

def _install_patch():
    global _orig_load_dataset
    if _datasets is None:
        return
    if getattr(_datasets, "load_dataset", None) is None:
        return
    if _orig_load_dataset is not None:
        return
    _orig_load_dataset = _datasets.load_dataset

    def _load_dataset_override(*args, **kwargs):
        # Intercept common gsm8k calls and substitute the AI-MO dataset.
        if len(args) >= 1 and isinstance(args[0], str) and args[0].lower() == "gsm8k":
            return _orig_load_dataset("AI-MO/aimo-validation-aime", "default", split=kwargs.get("split", "train"))
        # Fallback to original behaviour
        return _orig_load_dataset(*args, **kwargs)

    _datasets.load_dataset = _load_dataset_override

# Try to install immediately (works when datasets is importable now).
_install_patch()

def __getattr__(name):
    # If datasets becomes available later, install the patch lazily.
    if name == "_ensure_patch":
        _install_patch()
        return lambda: None
    raise AttributeError(name)
