#!/bin/bash
# Evaluate all ablation checkpoints using evaluate_tsfm.py
# and generate a comparison table.
#
# Usage:
#   bash scripts/eval_ablations.sh              # evaluate all ablations
#   bash scripts/eval_ablations.sh baseline      # evaluate one ablation
#
# Each ablation's best.pt is evaluated on all test datasets.
# Results are saved to test_output/ablation_evaluation/<ablation_name>/tsfm_evaluation.json
# A summary comparison is printed at the end.

set -e
cd "$(dirname "$0")/.."

# Activate venv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Map ablation names to checkpoint directories
declare -A CHECKPOINTS

# RunPod ablations
for d in training_output/runpod_ablations/*ablation_*/; do
    name=$(basename "$d" | sed 's/^[0-9_]*ablation_/ablation_/')  # strip timestamp
    if [ -f "$d/best.pt" ]; then
        CHECKPOINTS["$name"]="$d/best.pt"
    fi
done

# Local ablations
for d in training_output/semantic_alignment/*ablation_*/; do
    name=$(basename "$d" | sed 's/^[0-9_]*ablation_/ablation_/')
    if [ -f "$d/best.pt" ] && [ -f "$d/epoch_100.pt" ]; then
        # Only include completed runs (have epoch_100)
        CHECKPOINTS["$name"]="$d/best.pt"
    fi
done

# Deduplicate: keep the latest (longest path sorts last)
declare -A BEST_CHECKPOINTS
for name in "${!CHECKPOINTS[@]}"; do
    BEST_CHECKPOINTS["$name"]="${CHECKPOINTS[$name]}"
done

EVAL_OUTPUT_BASE="test_output/ablation_evaluation"
mkdir -p "$EVAL_OUTPUT_BASE"

# Filter to specific ablation if argument given
if [ -n "$1" ]; then
    if [ -z "${BEST_CHECKPOINTS[$1]}" ]; then
        # Try with ablation_ prefix
        key="ablation_$1"
        if [ -z "${BEST_CHECKPOINTS[$key]}" ]; then
            echo "Error: Unknown ablation '$1'"
            echo "Available: ${!BEST_CHECKPOINTS[*]}"
            exit 1
        fi
        FILTER="$key"
    else
        FILTER="$1"
    fi
fi

echo "============================================"
echo "  Ablation Evaluation"
echo "  $(date)"
echo "============================================"
echo ""
echo "Ablations to evaluate:"
for name in $(echo "${!BEST_CHECKPOINTS[@]}" | tr ' ' '\n' | sort); do
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi
    echo "  $name -> ${BEST_CHECKPOINTS[$name]}"
done
echo ""

# Run evaluation for each ablation
for name in $(echo "${!BEST_CHECKPOINTS[@]}" | tr ' ' '\n' | sort); do
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi

    checkpoint="${BEST_CHECKPOINTS[$name]}"
    out_dir="$EVAL_OUTPUT_BASE/$name"
    results_file="$out_dir/tsfm_evaluation.json"

    # Skip if already evaluated
    if [ -f "$results_file" ]; then
        echo ">>> SKIP $name (already evaluated, delete $results_file to re-run)"
        continue
    fi

    echo ""
    echo "======================================================================"
    echo ">>> Evaluating: $name"
    echo ">>>   Checkpoint: $checkpoint"
    echo "======================================================================"

    # Run evaluation
    TSFM_CHECKPOINT="$checkpoint" python val_scripts/human_activity_recognition/evaluate_tsfm.py

    # Move results to ablation-specific directory
    mkdir -p "$out_dir"
    cp test_output/baseline_evaluation/tsfm_evaluation.json "$results_file"

    echo ">>> Results saved to $results_file"
done

echo ""
echo ""
echo "======================================================================"
echo "  ABLATION COMPARISON SUMMARY"
echo "======================================================================"
echo ""

# Generate comparison table from JSON results
python3 -c "
import json, os, sys
from pathlib import Path

eval_dir = Path('$EVAL_OUTPUT_BASE')
ablations = {}

for d in sorted(eval_dir.iterdir()):
    results_file = d / 'tsfm_evaluation.json'
    if results_file.exists():
        with open(results_file) as f:
            ablations[d.name] = json.load(f)

if not ablations:
    print('No results found.')
    sys.exit(0)

# Collect all datasets
datasets = set()
for results in ablations.values():
    datasets.update(results.keys())
datasets = sorted(datasets)

metrics = ['zero_shot_open_set', 'zero_shot_closed_set', '1pct_supervised', '10pct_supervised']
metric_labels = ['ZS Open', 'ZS Closed', '1% Sup', '10% Sup']

# Print per-dataset comparison
for metric, label in zip(metrics, metric_labels):
    print(f'--- {label} Accuracy ---')
    header = f'{\"Ablation\":<30}'
    for ds in datasets:
        header += f'{ds:>14}'
    header += f'{\"  AVG\":>8}'
    print(header)
    print('-' * len(header))

    for abl_name in sorted(ablations.keys()):
        results = ablations[abl_name]
        row = f'{abl_name:<30}'
        accs = []
        for ds in datasets:
            if ds in results and metric in results[ds]:
                acc = results[ds][metric].get('accuracy', 0)
                row += f'{acc:>13.1f}%'
                accs.append(acc)
            else:
                row += f'{\"N/A\":>14}'
        avg = sum(accs) / len(accs) if accs else 0
        row += f'{avg:>7.1f}%'
        print(row)
    print()

# Print delta vs baseline
baseline_key = [k for k in ablations if 'baseline' in k]
if baseline_key:
    baseline_key = baseline_key[0]
    print()
    print('--- Delta vs Baseline (ZS Open Acc) ---')
    header = f'{\"Ablation\":<30}'
    for ds in datasets:
        header += f'{ds:>14}'
    header += f'{\"  AVG\":>8}'
    print(header)
    print('-' * len(header))

    baseline = ablations[baseline_key]
    for abl_name in sorted(ablations.keys()):
        results = ablations[abl_name]
        row = f'{abl_name:<30}'
        deltas = []
        for ds in datasets:
            if ds in results and 'zero_shot_open_set' in results[ds] and ds in baseline and 'zero_shot_open_set' in baseline[ds]:
                acc = results[ds]['zero_shot_open_set'].get('accuracy', 0)
                base_acc = baseline[ds]['zero_shot_open_set'].get('accuracy', 0)
                delta = acc - base_acc
                row += f'{delta:>+13.1f}%'
                deltas.append(delta)
            else:
                row += f'{\"N/A\":>14}'
        avg = sum(deltas) / len(deltas) if deltas else 0
        row += f'{avg:>+7.1f}%'
        print(row)
    print()
"

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  $(date)"
echo "============================================"
