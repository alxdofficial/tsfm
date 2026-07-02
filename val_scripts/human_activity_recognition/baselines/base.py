"""
Baseline adapter framework for evaluation protocol v2.

Each baseline is a small adapter file in this package. To add a baseline you
drop a `<name>.py` here that subclasses `ConSEAdapter` (closed-vocabulary
classifier -> softmax over the 87 global training labels, bridged with ConSE)
or `CosineAdapter` (text-aligned -> per-window embeddings + text prototypes),
overriding `setup()` plus its one tier method, and decorating it with
`@register`. The generic driver `run_baselines_v2.py` handles ground truth,
scoring, CIs and I/O — no per-baseline dispatch code.

Ground truth is derived only via `eval_v2.window_ground_truth` (offset-free);
the baselines' own v1 `get_window_labels` (min-subtraction, the HARTH bug) is
never used. See docs/baselines/EVALUATION_PROTOCOL_V2.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from val_scripts.human_activity_recognition import eval_v2
from val_scripts.human_activity_recognition.grouped_zero_shot import load_global_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCH_LIMU = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert"   # 20 Hz baseline data
CACHED_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"          # cached zero-shot classifiers

# Populated by @register at import time: name -> adapter instance.
REGISTRY: Dict[str, "BaselineAdapter"] = {}


def register(cls):
    """Class decorator: instantiate and add to REGISTRY under cls.name."""
    REGISTRY[cls.name] = cls()
    return cls


class BaselineAdapter:
    """Base adapter. Subclass ConSEAdapter or CosineAdapter, not this directly."""
    name: str = ""
    tier: str = ""  # "conse" | "cosine"

    def setup(self, device):
        """Load model + cached artifacts once; return an opaque state dict."""
        raise NotImplementedError


class ConSEAdapter(BaselineAdapter):
    """Closed-vocabulary classifier scored via the ConSE bridge."""
    tier = "conse"

    def window_probs(self, ds: str, state, device) -> np.ndarray:
        """Return per-test-window softmax over the 87 global labels: (N, 87),
        aligned 1:1 with benchmark_data/processed/limubert/{ds}/label_20_120.npy."""
        raise NotImplementedError


class CosineAdapter(BaselineAdapter):
    """Text-aligned model scored by cosine similarity (Tier-1, no bridge)."""
    tier = "cosine"

    def window_embeddings(self, ds: str, state, device) -> np.ndarray:
        """Per-window L2-normalized sensor embeddings: (N, D)."""
        raise NotImplementedError

    def encode_labels(self, L_D: List[str], state, device) -> np.ndarray:
        """Encode the target dataset's label strings into the same space: (L, D)."""
        raise NotImplementedError


# =============================================================================
# Shared ground truth + scoring (offset-free v2)
# =============================================================================

def load_gt(ds: str):
    """(L_D, gt_names, subjects, keep_idx) from the 20 Hz label file.

    keep_idx indexes the original N windows so any per-window model output
    aligns via output[keep_idx]. idx_to_label keys are int-cast (JSON strings).
    """
    cfg = eval_v2.load_label_config(ds)
    L_D = cfg["labels"]
    idx_to_label = {int(k): v for k, v in cfg["idx_to_label"].items()}
    labels_raw = np.load(str(BENCH_LIMU / ds / "label_20_120.npy"))
    gt_names, subjects, keep_idx = eval_v2.window_ground_truth(labels_raw, idx_to_label)
    return L_D, gt_names, subjects, keep_idx


def score(gt_names, pred_names, subjects, extra: dict = None) -> dict:
    """v2 metric bundle: macro-F1 (primary) + balanced-acc + CIs + per-class."""
    m = eval_v2.classification_metrics(gt_names, pred_names)
    m.update(eval_v2.subject_bootstrap_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    m["per_class_f1"] = eval_v2.per_class_f1(gt_names, pred_names)
    if extra:
        m.update(extra)
    return m


def global_labels() -> List[str]:
    return load_global_labels()
