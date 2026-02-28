"""
LLaSA evaluation for zero-shot HAR classification.

LLaSA (Large Language and Sensor Assistant) uses LIMU-BERT as its IMU encoder
combined with Vicuna-7B LLM for generative activity classification.

Unlike other baselines (embedding + classifier), LLaSA classifies by prompting
the LLM: "What activity?" -> parses text response -> maps to label.

Evaluation modes:
  1. Zero-shot closed-set: prompt with dataset's activity list
  2. Zero-shot open-set: prompt with all 87 global training labels

Uses the same LIMU-BERT preprocessed data (20 Hz, 120-step, 6ch) as other baselines.

Usage:
    python val_scripts/human_activity_recognition/evaluate_llasa.py [--max-per-class N]
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# LLaSA/LLaVA code
LLASA_REPO = PROJECT_ROOT / "auxiliary_repos" / "LLaSA"
LLASA_CODE = LLASA_REPO / "LLaSA"
sys.path.insert(0, str(LLASA_CODE))

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DIR / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# =============================================================================
# Configuration
# =============================================================================

# Note: The LLaSA paper primarily uses a 13B model. Both 7B and 13B are on HuggingFace.
# We use 7B at fp16 (~13GB) to fit in 24GB VRAM. 13B would require quantization.
MODEL_ID = "BASH-Lab/LLaSA-7B"
SEED = 42
MAX_NEW_TOKENS = 50
TEMPERATURE = 0  # greedy decoding
MAX_PER_CLASS = None  # None = all data, or set to e.g. 100 for faster eval

# Load dataset config
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]


# =============================================================================
# Label utilities (reuse from grouped_zero_shot)
# =============================================================================

from val_scripts.human_activity_recognition.grouped_zero_shot import (
    load_global_labels, map_local_to_global_labels,
    get_closed_set_mask, score_with_groups,
    score_exact, score_with_groups_from_names,
)


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# =============================================================================
# Data loading
# =============================================================================

def load_limubert_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load LIMU-BERT preprocessed data and mapping for a dataset."""
    ds_dir = LIMUBERT_DIR / dataset_name
    if not ds_dir.exists():
        raise FileNotFoundError(f"No LIMU-BERT data for {dataset_name} at {ds_dir}")

    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
    with open(ds_dir / "mapping.json") as f:
        mapping = json.load(f)
    return data, labels, mapping


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels via majority vote (consistent with all baselines)."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array([
        np.bincount(row.astype(int)).argmax() for row in act_labels
    ], dtype=np.int64)
    return window_labels + t


def sample_balanced(data, labels, mapping, max_per_class):
    """Sample up to max_per_class examples per activity class."""
    idx_to_activity = {v: k for k, v in mapping["activity_to_idx"].items()}
    activity_indices = get_window_labels(labels)

    selected = []
    for cls_idx in sorted(idx_to_activity.keys()):
        cls_mask = activity_indices == cls_idx
        cls_indices = np.where(cls_mask)[0]
        if len(cls_indices) > max_per_class:
            rng = np.random.RandomState(SEED + cls_idx)
            cls_indices = rng.choice(cls_indices, max_per_class, replace=False)
        selected.extend(cls_indices.tolist())

    selected = sorted(selected)
    return data[selected], labels[selected]


# =============================================================================
# LLaSA model loading
# =============================================================================

def load_llasa_model(device="cuda"):
    """Load LLaSA-7B model from HuggingFace."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(MODEL_ID)
    print(f"Loading LLaSA model: {MODEL_ID} (name: {model_name})")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_ID, None, model_name,
        device=device,
    )
    print(f"Model loaded. Context length: {context_len}")
    return tokenizer, model, image_processor


# =============================================================================
# Inference
# =============================================================================

def classify_sample(
    model, tokenizer, sensor_data: np.ndarray,
    activity_list: List[str], device="cuda",
) -> str:
    """Classify a single sensor window using LLaSA.

    Args:
        sensor_data: (120, 6) numpy array at 20 Hz
        activity_list: list of activity class names

    Returns:
        Predicted activity string (lowercase) or "unclear"
    """
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    # Format class list for prompt
    classes_str = ", ".join(activity_list)
    answer_format = "The identified class is: "
    query = (
        f"The given IMU sensor data can be associated with one of the following "
        f"classes: {classes_str}. "
        f"Write only the name of the identified class in the format "
        f"'{answer_format}'"
    )

    # Prepend image token
    qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    # Build conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Prepare sensor tensor
    sensor_tensor = torch.from_numpy(sensor_data).unsqueeze(0)  # (1, 120, 6)
    sensor_tensor = sensor_tensor.to(device=model.device, dtype=model.dtype)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=sensor_tensor,
            image_sizes=[(120, 6)],
            do_sample=False,
            temperature=None,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Parse response
    if answer_format in output_text:
        pred = output_text.split(answer_format)[-1]
        pred = re.sub(r'[^\w\s]', '', pred).strip().lower()
        # Handle underscored vs spaced labels
        pred_underscore = pred.replace(" ", "_")
        for label in activity_list:
            if pred == label.lower() or pred_underscore == label.lower():
                return label
        # Fuzzy match: check containment in both directions
        for label in activity_list:
            ll = label.lower()
            if ll in pred or ll in pred_underscore or pred in ll or pred_underscore in ll:
                return label

    return "unclear"


def evaluate_dataset_closed_set(
    model, tokenizer, data, labels, mapping, dataset_name, device="cuda",
) -> Dict:
    """Zero-shot closed-set: prompt with dataset's own activity labels."""
    activities = get_dataset_labels(dataset_name)
    idx_to_activity = {v: k for k, v in mapping["activity_to_idx"].items()}
    activity_indices = get_window_labels(labels)

    y_true = []
    y_pred = []
    n_unclear = 0

    for i in tqdm(range(len(data)), desc=f"  {dataset_name} closed-set"):
        sensor_window = data[i]  # (120, 6)
        true_label = idx_to_activity[activity_indices[i]]

        pred_label = classify_sample(model, tokenizer, sensor_window, activities, device)

        y_true.append(true_label)
        y_pred.append(pred_label)
        if pred_label == "unclear":
            n_unclear += 1

    # Exact match scoring
    exact = score_exact(y_pred, y_true)
    # Group match scoring
    group = score_with_groups_from_names(y_pred, y_true)

    return {
        "accuracy_exact": exact["accuracy"], "f1_macro_exact": exact["f1_macro"],
        "f1_weighted_exact": exact["f1_weighted"],
        "accuracy_group": group["accuracy"], "f1_macro_group": group["f1_macro"],
        "f1_weighted_group": group["f1_weighted"],
        "n_samples": len(data),
        "n_unclear": n_unclear,
        "n_classes": len(activities),
    }


def evaluate_dataset_open_set(
    model, tokenizer, data, labels, mapping, dataset_name, device="cuda",
) -> Dict:
    """Zero-shot open-set: prompt with all 87 global training labels."""
    global_labels = GLOBAL_LABELS
    idx_to_activity = {v: k for k, v in mapping["activity_to_idx"].items()}
    activity_indices = get_window_labels(labels)

    y_true = []
    y_pred = []
    n_unclear = 0

    for i in tqdm(range(len(data)), desc=f"  {dataset_name} open-set"):
        sensor_window = data[i]
        true_label = idx_to_activity[activity_indices[i]]

        pred_label = classify_sample(model, tokenizer, sensor_window, global_labels, device)

        y_true.append(true_label)
        y_pred.append(pred_label)
        if pred_label == "unclear":
            n_unclear += 1

    # Exact match scoring
    exact = score_exact(y_pred, y_true)
    # Group match scoring
    group = score_with_groups_from_names(y_pred, y_true)

    return {
        "accuracy_exact": exact["accuracy"], "f1_macro_exact": exact["f1_macro"],
        "f1_weighted_exact": exact["f1_weighted"],
        "accuracy_group": group["accuracy"], "f1_macro_group": group["f1_macro"],
        "f1_weighted_group": group["f1_weighted"],
        "n_samples": len(data),
        "n_unclear": n_unclear,
    }


# =============================================================================
# Results display
# =============================================================================

def print_results_table(results: Dict):
    """Print results in a fixed-width table."""
    header = (f"{'Dataset':<16} "
              f"{'Open Exact':>11} {'Open Group':>11} "
              f"{'Close Exact':>12} {'Close Group':>12}")
    print(f"\n{'='*80}")
    print("LLaSA-7B Zero-Shot Results")
    print(f"{'='*80}")
    print(header)
    print("-" * 80)
    for ds, metrics in results.items():
        zs_open = metrics.get("zero_shot_open_set", {})
        zs_close = metrics.get("zero_shot_closed_set", {})
        print(f"{ds:<16} "
              f"{zs_open.get('accuracy_exact', 0):>10.1f}% "
              f"{zs_open.get('accuracy_group', 0):>10.1f}% "
              f"{zs_close.get('accuracy_exact', 0):>11.1f}% "
              f"{zs_close.get('accuracy_group', 0):>11.1f}%")
    print(f"{'='*80}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLaSA zero-shot HAR evaluation")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max samples per class (default: all data)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to evaluate (default: all zero-shot)")
    parser.add_argument("--skip-open-set", action="store_true",
                        help="Skip open-set evaluation (faster)")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    tokenizer, model, image_processor = load_llasa_model(device)

    # Select datasets
    test_datasets = args.datasets if args.datasets else TEST_DATASETS
    max_per_class = args.max_per_class or MAX_PER_CLASS

    all_results = {}

    for test_ds in test_datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating LLaSA on {test_ds}")
        print(f"{'='*60}")

        # Check if data exists
        ds_dir = LIMUBERT_DIR / test_ds
        if not ds_dir.exists():
            print(f"  SKIPPED: No LIMU-BERT data at {ds_dir}")
            continue

        # Load data
        data, labels, mapping = load_limubert_data(test_ds)
        print(f"  Loaded {len(data)} windows, {len(mapping['activity_to_idx'])} classes")

        # Sample if needed
        if max_per_class is not None:
            data, labels = sample_balanced(data, labels, mapping, max_per_class)
            print(f"  Sampled to {len(data)} windows ({max_per_class} per class)")

        ds_results = {}

        # 1. Zero-shot closed-set
        print(f"\n  --- Zero-shot Closed-Set ---")
        t0 = time.time()
        ds_results["zero_shot_closed_set"] = evaluate_dataset_closed_set(
            model, tokenizer, data, labels, mapping, test_ds, device)
        elapsed = time.time() - t0
        r = ds_results["zero_shot_closed_set"]
        print(f"  Exact={r['accuracy_exact']:.1f}%, Group={r['accuracy_group']:.1f}%, "
              f"unclear={r['n_unclear']}/{r['n_samples']} ({elapsed:.0f}s)")

        # 2. Zero-shot open-set (optional, slower due to 87 labels in prompt)
        if not args.skip_open_set:
            print(f"\n  --- Zero-shot Open-Set ---")
            t0 = time.time()
            ds_results["zero_shot_open_set"] = evaluate_dataset_open_set(
                model, tokenizer, data, labels, mapping, test_ds, device)
            elapsed = time.time() - t0
            r = ds_results["zero_shot_open_set"]
            print(f"  Exact={r['accuracy_exact']:.1f}%, Group={r['accuracy_group']:.1f}%, "
                  f"unclear={r['n_unclear']}/{r['n_samples']} ({elapsed:.0f}s)")

        all_results[test_ds] = ds_results

    # Print summary
    print_results_table(all_results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "llasa_evaluation.json"

    save_data = {}
    for ds, metrics in all_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            if isinstance(metric_vals, dict):
                save_data[ds][metric_name] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metric_vals.items()
                }
            else:
                save_data[ds][metric_name] = metric_vals

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
