#!/usr/bin/env python3
"""Generate ablation results markdown from evaluation JSON files.

Reads test_output/ablation_evaluation/*/tsfm_evaluation.json and produces
docs/ablations.md with per-dataset tables and averages.

Usage:
    python scripts/generate_ablation_results.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "test_output" / "ablation_evaluation"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "ablation_results.md"

# Dataset categorization
MAIN_DATASETS = ["motionsense", "realworld", "shoaib", "opportunity"]
SEVERE_OOD_DATASETS = ["mobiact", "vtt_coniot", "harth"]

# Display names
DATASET_NAMES = {
    "motionsense": "MotionSense",
    "realworld": "RealWorld",
    "shoaib": "Shoaib",
    "opportunity": "Opportunity",
    "mobiact": "MobiAct",
    "vtt_coniot": "VTT-ConIoT",
    "harth": "HARTH",
}

ABLATION_DISPLAY = {
    "ablation_baseline": "Baseline (all enabled)",
    "ablation_no_channel_fusion": "No Channel-Text Fusion",
    "ablation_no_label_bank": "No Learnable Label Bank",
    "ablation_no_soft_targets": "No Soft Targets",
    "ablation_no_signal_aug": "No Signal Augmentation",
    "ablation_no_text_aug": "No Text Augmentation",
}

ABLATION_ORDER = [
    "ablation_baseline",
    "ablation_no_channel_fusion",
    "ablation_no_label_bank",
    "ablation_no_soft_targets",
    "ablation_no_signal_aug",
    "ablation_no_text_aug",
]

METRICS = [
    ("zero_shot_open_set", "ZS Open-Set Acc"),
    ("zero_shot_closed_set", "ZS Closed-Set Acc"),
    ("1pct_supervised", "1% Supervised Acc"),
    ("10pct_supervised", "10% Supervised Acc"),
]


def load_all_results():
    """Load all ablation evaluation results."""
    results = {}
    for d in sorted(EVAL_DIR.iterdir()):
        rf = d / "tsfm_evaluation.json"
        if rf.exists():
            results[d.name] = json.load(open(rf))
    return results


def get_accuracy(results, ablation, dataset, metric):
    """Get accuracy for a specific ablation/dataset/metric, or None if missing."""
    if ablation not in results:
        return None
    if dataset not in results[ablation]:
        return None
    if metric not in results[ablation][dataset]:
        return None
    return results[ablation][dataset][metric].get("accuracy")


def get_f1(results, ablation, dataset, metric):
    """Get F1 for a specific ablation/dataset/metric, or None if missing."""
    if ablation not in results:
        return None
    if dataset not in results[ablation]:
        return None
    if metric not in results[ablation][dataset]:
        return None
    return results[ablation][dataset][metric].get("f1_macro")


def compute_avg(values):
    """Compute average of non-None values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt(val, delta=False):
    """Format a value as percentage string."""
    if val is None:
        return "—"
    if delta:
        return f"{val:+.1f}%"
    return f"{val:.1f}%"


def generate_summary_table(results, datasets, label):
    """Generate a summary table for a set of datasets."""
    lines = []
    ordered_ablations = [a for a in ABLATION_ORDER if a in results]

    for metric_key, metric_label in METRICS:
        lines.append(f"#### {metric_label}")
        lines.append("")

        # Header
        header = f"| {'Ablation':<30} |"
        separator = f"|:{'-'*29}-|"
        for ds in datasets:
            if ds in list(results.values())[0] if results else []:
                name = DATASET_NAMES.get(ds, ds)
                header += f" {name:>12} |"
                separator += f" {'-'*11}:|"
        header += f" {'**AVG**':>8} |"
        separator += f" {'-'*7}:|"
        lines.append(header)
        lines.append(separator)

        # Rows
        baseline_avgs = {}
        for abl in ordered_ablations:
            display = ABLATION_DISPLAY.get(abl, abl)
            if abl == "ablation_baseline":
                display = f"**{display}**"
            row = f"| {display:<30} |"
            accs = []
            for ds in datasets:
                acc = get_accuracy(results, abl, ds, metric_key)
                if acc is not None:
                    accs.append(acc)
                    row += f" {acc:>11.1f}% |"
                else:
                    row += f" {'—':>12} |"
            avg = compute_avg(accs)
            if abl == "ablation_baseline":
                baseline_avgs[metric_key] = avg
                row += f" **{fmt(avg)}** |" if avg is not None else f" {'—':>8} |"
            else:
                row += f" {fmt(avg):>8} |" if avg is not None else f" {'—':>8} |"
            lines.append(row)

        lines.append("")

    return lines, baseline_avgs


def generate_delta_table(results, datasets, label):
    """Generate a delta-vs-baseline table."""
    lines = []
    ordered_ablations = [a for a in ABLATION_ORDER if a in results]
    baseline = "ablation_baseline"
    if baseline not in results:
        return lines

    for metric_key, metric_label in METRICS:
        lines.append(f"#### {metric_label} (Delta vs Baseline)")
        lines.append("")

        header = f"| {'Ablation':<30} |"
        separator = f"|:{'-'*29}-|"
        for ds in datasets:
            if ds in results.get(baseline, {}):
                name = DATASET_NAMES.get(ds, ds)
                header += f" {name:>12} |"
                separator += f" {'-'*11}:|"
        header += f" {'**AVG**':>8} |"
        separator += f" {'-'*7}:|"
        lines.append(header)
        lines.append(separator)

        for abl in ordered_ablations:
            if abl == baseline:
                continue
            display = ABLATION_DISPLAY.get(abl, abl)
            row = f"| {display:<30} |"
            deltas = []
            for ds in datasets:
                acc = get_accuracy(results, abl, ds, metric_key)
                base_acc = get_accuracy(results, baseline, ds, metric_key)
                if acc is not None and base_acc is not None:
                    delta = acc - base_acc
                    deltas.append(delta)
                    row += f" {delta:>+11.1f}% |"
                else:
                    row += f" {'—':>12} |"
            avg_delta = compute_avg(deltas)
            row += f" {fmt(avg_delta, delta=True):>8} |" if avg_delta is not None else f" {'—':>8} |"
            lines.append(row)

        lines.append("")

    return lines


def generate_per_dataset_table(results, dataset):
    """Generate a detailed table for one dataset."""
    lines = []
    ds_name = DATASET_NAMES.get(dataset, dataset)
    ordered_ablations = [a for a in ABLATION_ORDER if a in results]

    header = f"| {'Ablation':<30} | {'ZS Open':>9} | {'ZS Closed':>10} | {'1% Sup':>9} | {'10% Sup':>9} |"
    separator = f"|:{'-'*29}-| {'-'*8}:| {'-'*9}:| {'-'*8}:| {'-'*8}:|"
    lines.append(header)
    lines.append(separator)

    for abl in ordered_ablations:
        display = ABLATION_DISPLAY.get(abl, abl)
        if abl == "ablation_baseline":
            display = f"**{display}**"

        zso = get_accuracy(results, abl, dataset, "zero_shot_open_set")
        zsc = get_accuracy(results, abl, dataset, "zero_shot_closed_set")
        s1 = get_accuracy(results, abl, dataset, "1pct_supervised")
        s10 = get_accuracy(results, abl, dataset, "10pct_supervised")

        row = f"| {display:<30} | {fmt(zso):>9} | {fmt(zsc):>10} | {fmt(s1):>9} | {fmt(s10):>9} |"
        lines.append(row)

    return lines


def main():
    results = load_all_results()
    if not results:
        print("No results found. Run scripts/eval_ablations.sh first.")
        return

    # Determine which datasets are actually present
    all_datasets = set()
    for abl_results in results.values():
        all_datasets.update(abl_results.keys())

    main_ds = [ds for ds in MAIN_DATASETS if ds in all_datasets]
    ood_ds = [ds for ds in SEVERE_OOD_DATASETS if ds in all_datasets]
    all_ds = main_ds + ood_ds

    lines = []
    lines.append("# Ablation Study Results")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Six ablation studies, each disabling one component while keeping everything else at baseline.")
    lines.append("All runs use small_deep (d=384, 8 layers), 100 epochs, memory bank (512), no GradCache.")
    lines.append("")
    lines.append("| Ablation | What's Disabled |")
    lines.append("|----------|----------------|")
    for abl in ABLATION_ORDER:
        if abl in results:
            lines.append(f"| {ABLATION_DISPLAY[abl]} | {_get_description(abl)} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # =========================================================
    # Average summary tables
    # =========================================================
    lines.append("## Summary: Average Accuracy")
    lines.append("")

    # All datasets average
    lines.append("### All Datasets Average")
    lines.append("")
    summary_lines, _ = generate_summary_table(results, all_ds, "All")
    lines.extend(summary_lines)

    # Main datasets average
    lines.append("### Main Datasets Average")
    lines.append(f"")
    lines.append(f"Datasets: {', '.join(DATASET_NAMES.get(ds, ds) for ds in main_ds)}")
    lines.append("")
    summary_lines, _ = generate_summary_table(results, main_ds, "Main")
    lines.extend(summary_lines)

    # Severe OOD average
    lines.append("### Severe OOD Datasets Average")
    lines.append(f"")
    lines.append(f"Datasets: {', '.join(DATASET_NAMES.get(ds, ds) for ds in ood_ds)}")
    lines.append("")
    summary_lines, _ = generate_summary_table(results, ood_ds, "OOD")
    lines.extend(summary_lines)

    lines.append("---")
    lines.append("")

    # =========================================================
    # Delta vs baseline
    # =========================================================
    lines.append("## Delta vs Baseline")
    lines.append("")

    lines.append("### Main Datasets")
    lines.append("")
    delta_lines = generate_delta_table(results, main_ds, "Main")
    lines.extend(delta_lines)

    lines.append("### Severe OOD Datasets")
    lines.append("")
    delta_lines = generate_delta_table(results, ood_ds, "OOD")
    lines.extend(delta_lines)

    lines.append("---")
    lines.append("")

    # =========================================================
    # Per-dataset tables
    # =========================================================
    lines.append("## Per-Dataset Results")
    lines.append("")

    lines.append("### Main Datasets")
    lines.append("")
    for ds in main_ds:
        ds_name = DATASET_NAMES.get(ds, ds)
        lines.append(f"#### {ds_name}")
        lines.append("")
        per_ds_lines = generate_per_dataset_table(results, ds)
        lines.extend(per_ds_lines)
        lines.append("")

    lines.append("### Severe OOD Datasets")
    lines.append("")
    for ds in ood_ds:
        ds_name = DATASET_NAMES.get(ds, ds)
        lines.append(f"#### {ds_name}")
        lines.append("")
        per_ds_lines = generate_per_dataset_table(results, ds)
        lines.extend(per_ds_lines)
        lines.append("")

    # =========================================================
    # Training log metrics
    # =========================================================
    lines.append("---")
    lines.append("")
    lines.append("## Training Metrics (from logs)")
    lines.append("")
    lines.append("Best validation accuracy and last unseen (motionsense) accuracy from training logs.")
    lines.append("")
    lines.append("| Ablation | Best Val Acc | Last Unseen Acc |")
    lines.append("|----------|:-----------:|:--------------:|")

    log_dirs = {
        "ablation_baseline": PROJECT_ROOT / "training_output" / "runpod_ablations" / "ablation_baseline.log",
        "ablation_no_channel_fusion": PROJECT_ROOT / "training_output" / "runpod_ablations" / "ablation_no_channel_fusion.log",
        "ablation_no_label_bank": PROJECT_ROOT / "training_output" / "runpod_ablations" / "ablation_no_label_bank.log",
        "ablation_no_soft_targets": PROJECT_ROOT / "training_output" / "runpod_ablations" / "ablation_no_soft_targets.log",
        "ablation_no_signal_aug": PROJECT_ROOT / "training_output" / "ablation_no_signal_aug.log",
        "ablation_no_text_aug": PROJECT_ROOT / "training_output" / "ablation_no_text_aug.log",
    }

    import re
    for abl in ABLATION_ORDER:
        if abl not in results:
            continue
        display = ABLATION_DISPLAY.get(abl, abl)
        log_path = log_dirs.get(abl)
        val_acc = "—"
        unseen_acc = "—"
        if log_path and log_path.exists():
            text = log_path.read_text()
            val_matches = re.findall(r"Acc: ([\d.]+)%", text)
            unseen_matches = re.findall(r"Unseen.*?Acc: ([\d.]+)%", text)
            if val_matches:
                val_acc = f"{max(float(v) for v in val_matches):.1f}%"
            if unseen_matches:
                unseen_acc = f"{float(unseen_matches[-1]):.1f}%"
        lines.append(f"| {display} | {val_acc} | {unseen_acc} |")

    lines.append("")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Written to {OUTPUT_PATH}")
    print(f"  {len(results)} ablations, {len(all_ds)} datasets")


def _get_description(abl):
    descs = {
        "ablation_baseline": "Nothing (control)",
        "ablation_no_channel_fusion": "Gated cross-attention between sensor tokens and channel text",
        "ablation_no_label_bank": "Learnable attention pooling for text embeddings (uses frozen mean-pool)",
        "ablation_no_soft_targets": "Soft targets + batch-mean similarity normalization in InfoNCE",
        "ablation_no_signal_aug": "Jitter + scale augmentation on sensor data",
        "ablation_no_text_aug": "Label synonyms/templates + Hz/window suffix on channel descriptions",
    }
    return descs.get(abl, "")


if __name__ == "__main__":
    main()
