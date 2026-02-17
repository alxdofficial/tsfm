"""
Generate a combined Markdown results table from all baseline evaluation JSONs.

4-metric framework:
  - Zero-shot open-set (all models)
  - Zero-shot closed-set (all models)
  - 1% supervised end-to-end fine-tuning (all models)
  - 10% supervised end-to-end fine-tuning (all models)

Results are split into:
  - Main test datasets (MotionSense, RealWorld, MobiAct) — 85-100% label coverage
  - Severe out-of-domain (VTT-ConIoT) — 50% label coverage, reported separately

Reads from test_output/baseline_evaluation/*.json and writes:
  test_output/baseline_evaluation/results_table.md
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("test_output/baseline_evaluation")

# Map JSON filenames to display names (ordered)
BASELINE_FILES = [
    ("tsfm_evaluation.json", "TSFM (ours)"),
    ("limubert_evaluation.json", "LiMU-BERT"),
    ("moment_evaluation.json", "MOMENT"),
    ("crosshar_evaluation.json", "CrossHAR"),
    ("lanhar_evaluation.json", "LanHAR"),
]

# Main test datasets (85-100% label coverage) — used for primary averages
MAIN_DATASETS = ["mobiact", "motionsense", "realworld"]

# Severe out-of-domain dataset (50% label coverage) — reported separately
OOD_DATASETS = ["vtt_coniot"]

# All metric keys in display order
METRIC_TABLES = [
    ("Zero-Shot Open-Set", "zero_shot_open_set"),
    ("Zero-Shot Closed-Set", "zero_shot_closed_set"),
    ("1% Supervised", "1pct_supervised"),
    ("10% Supervised", "10pct_supervised"),
]


def load_results():
    """Load all available result files, preserving order."""
    results = []
    for filename, display_name in BASELINE_FILES:
        path = OUTPUT_DIR / filename
        if path.exists():
            with open(path) as f:
                results.append((display_name, json.load(f)))
            print(f"  Loaded {filename}")
        else:
            print(f"  Skipped {filename} (not found)")
    return results


def _metric_row(baseline_name, baseline_data, metric_key, datasets):
    """Build table cells for one metric across datasets."""
    cells = ""
    for ds in datasets:
        if ds in baseline_data and metric_key in baseline_data[ds]:
            acc = baseline_data[ds][metric_key].get("accuracy", 0.0)
            f1 = baseline_data[ds][metric_key].get("f1_macro", 0.0)
            cells += f" {acc:.1f} | {f1:.1f} |"
        else:
            cells += " - | - |"
    return cells


def generate_table(results):
    """Generate Markdown table from results."""
    if not results:
        return "No results found.\n"

    # Determine which datasets are present
    all_datasets = set()
    for _, baseline_data in results:
        all_datasets.update(baseline_data.keys())
    main_ds = sorted(d for d in MAIN_DATASETS if d in all_datasets)
    ood_ds = sorted(d for d in OOD_DATASETS if d in all_datasets)

    lines = []
    lines.append("# Baseline Evaluation Results\n")
    lines.append("Generated from `test_output/baseline_evaluation/`\n")

    # --- Per-dataset tables (main datasets only) ---
    for table_title, metric_key in METRIC_TABLES:
        lines.append(f"\n## {table_title}\n")

        # Header
        header = "| Model |"
        separator = "| :--- |"
        for ds in main_ds:
            header += f" {ds} Acc | {ds} F1 |"
            separator += " ---: | ---: |"
        lines.append(header)
        lines.append(separator)

        # Rows — all models
        for baseline_name, baseline_data in results:
            row = f"| **{baseline_name}** |"
            row += _metric_row(baseline_name, baseline_data, metric_key, main_ds)
            lines.append(row)

    # --- Average across main datasets ---
    lines.append("\n## Average Across Main Datasets\n")
    lines.append("*Averaged over MotionSense, RealWorld, MobiAct (85-100% label coverage)*\n")

    header = "| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |"
    separator = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.append(header)
    lines.append(separator)

    all_metric_keys = [mk for _, mk in METRIC_TABLES]

    for baseline_name, baseline_data in results:
        row = f"| **{baseline_name}** |"
        for metric_key in all_metric_keys:
            accs = []
            f1s = []
            for ds in main_ds:
                if ds in baseline_data and metric_key in baseline_data[ds]:
                    accs.append(baseline_data[ds][metric_key].get("accuracy", 0.0))
                    f1s.append(baseline_data[ds][metric_key].get("f1_macro", 0.0))
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
            row += f" {avg_acc:.1f} | {avg_f1:.1f} |"
        lines.append(row)

    # --- VTT-ConIoT (severe out-of-domain) ---
    if ood_ds:
        lines.append("\n## Severe Out-of-Domain: VTT-ConIoT\n")
        lines.append("*50% label coverage — 8/16 activities have no training equivalent*\n")

        header = "| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |"
        separator = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        lines.append(header)
        lines.append(separator)

        for baseline_name, baseline_data in results:
            row = f"| **{baseline_name}** |"
            for metric_key in all_metric_keys:
                ds = ood_ds[0]
                if ds in baseline_data and metric_key in baseline_data[ds]:
                    acc = baseline_data[ds][metric_key].get("accuracy", 0.0)
                    f1 = baseline_data[ds][metric_key].get("f1_macro", 0.0)
                    row += f" {acc:.1f} | {f1:.1f} |"
                else:
                    row += " - | - |"
            lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    print("Generating combined results table...")
    results = load_results()

    if not results:
        print("No result files found. Run evaluations first.")
        return

    table = generate_table(results)

    output_path = OUTPUT_DIR / "results_table.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)

    print(f"\nResults table written to: {output_path}")
    print("\nPreview:\n")
    print(table)


if __name__ == "__main__":
    main()
