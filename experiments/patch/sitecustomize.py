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

try:
    import transformers as _transformers
except Exception:
    _transformers = None

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
        # Intercept common gsm8k calls and optionally substitute the AI-MO dataset.
        # Only redirect if USE_AIMO_DATASET=1 is set; otherwise use original GSM8K.
        import os
        use_aimo = os.environ.get("USE_AIMO_DATASET", "").lower() in ("1", "true", "yes")
        if use_aimo and len(args) >= 1 and isinstance(args[0], str) and args[0].lower() == "gsm8k":
            return _orig_load_dataset("AI-MO/aimo-validation-aime", "default", split=kwargs.get("split", "train"))
        # Fallback to original behaviour
        return _orig_load_dataset(*args, **kwargs)

    _datasets.load_dataset = _load_dataset_override


def _install_transformers_compat():
    if _transformers is None:
        return
    # lm-eval expects this symbol on some releases; older/newer transformers may
    # only expose AutoModelForImageTextToText.
    if hasattr(_transformers, "AutoModelForImageTextToText"):
        img_text_cls = _transformers.AutoModelForImageTextToText
        _transformers.AutoModelForVision2Seq = img_text_cls

        # transformers is a LazyModule; keep __getattr__ resolution stable.
        class_to_module = getattr(_transformers, "_class_to_module", None)
        if isinstance(class_to_module, dict) and "AutoModelForVision2Seq" not in class_to_module:
            src = class_to_module.get("AutoModelForImageTextToText")
            if src is not None:
                class_to_module["AutoModelForVision2Seq"] = src

        objects = getattr(_transformers, "_objects", None)
        if isinstance(objects, dict) and "AutoModelForVision2Seq" not in objects:
            objects["AutoModelForVision2Seq"] = img_text_cls

    # Final safety net: patch LazyModule getattr so lookups never fail.
    try:
        from transformers.utils import import_utils as _tf_import_utils

        lazy_cls = getattr(_tf_import_utils, "_LazyModule", None)
        if lazy_cls is not None and not getattr(lazy_cls, "_fastdllm_vision2seq_patch", False):
            orig_getattr = lazy_cls.__getattr__

            def _patched_getattr(self, name):
                if self.__name__ == "transformers" and name == "AutoModelForVision2Seq":
                    try:
                        return orig_getattr(self, name)
                    except AttributeError:
                        return orig_getattr(self, "AutoModelForImageTextToText")
                return orig_getattr(self, name)

            lazy_cls.__getattr__ = _patched_getattr
            lazy_cls._fastdllm_vision2seq_patch = True
    except Exception:
        # Keep startup resilient; if patching fails we'll surface the original error.
        pass

# Try to install immediately (works when datasets is importable now).
_install_patch()
_install_transformers_compat()

def __getattr__(name):
    # If datasets becomes available later, install the patch lazily.
    if name == "_ensure_patch":
        _install_patch()
        return lambda: None
    raise AttributeError(name)
