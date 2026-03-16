#!/usr/bin/env python3
"""Quick patch size sweep for TSFM on specified datasets.

Only evaluates zero-shot closed-set accuracy (fast, no fine-tuning).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank
from val_scripts.human_activity_recognition.evaluate_tsfm import (
    load_raw_data,
    get_dataset_metadata,
    get_window_labels,
    extract_tsfm_embeddings,
    evaluate_zero_shot_closed_set,
    CHECKPOINT_PATH,
)

PATCH_SIZES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
DATASETS = ["shoaib", "opportunity", "harth"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading TSFM model from {CHECKPOINT_PATH}...")
    model, checkpoint, hyperparams_path = load_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    print("Model and label bank loaded.")

    results = {}

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Patch size sweep: {ds}")
        print(f"{'='*60}")

        raw_data, raw_labels, sr = load_raw_data(ds)
        labels = get_window_labels(raw_labels)
        meta = get_dataset_metadata(ds)
        channel_descs = meta['channel_descriptions']

        print(f"  Sampling rate: {sr}Hz, Window size: {raw_data.shape[1]}")

        ds_results = {}
        for ps in PATCH_SIZES:
            n_patches = raw_data.shape[1] / (sr * ps)
            if n_patches < 1:
                print(f"  Patch {ps}s: SKIP (window too short)")
                continue

            print(f"\n  Patch size: {ps}s ({int(sr * ps)} samples/patch, {n_patches:.1f} patches/window)")

            embeddings = extract_tsfm_embeddings(
                model, raw_data, device, sr, channel_descs,
                patch_size_sec=ps,
            )

            zs_result = evaluate_zero_shot_closed_set(
                embeddings, labels, ds, label_bank, device,
            )

            acc = zs_result['accuracy']
            f1 = zs_result['f1_macro']
            ds_results[ps] = {'accuracy': acc, 'f1_macro': f1}
            print(f"  -> ZS-Closed Acc={acc:.1f}%, F1={f1:.1f}%")

        results[ds] = ds_results

    # Summary table
    print(f"\n\n{'='*80}")
    print("PATCH SIZE SWEEP RESULTS (Zero-Shot Closed-Set Accuracy %)")
    print(f"{'='*80}")
    header = f"{'Dataset':<15}"
    for ps in PATCH_SIZES:
        header += f" {ps}s".rjust(8)
    header += "   Range"
    print(header)
    print("-" * 80)

    for ds in DATASETS:
        row = f"{ds:<15}"
        accs = []
        for ps in PATCH_SIZES:
            if ps in results[ds]:
                acc = results[ds][ps]['accuracy']
                accs.append(acc)
                row += f" {acc:>6.1f}%"
            else:
                row += "     N/A"
        if accs:
            row += f"   {max(accs) - min(accs):.1f}"
        print(row)

    print(f"\nBest patch sizes:")
    for ds in DATASETS:
        if results[ds]:
            best_ps = max(results[ds], key=lambda ps: results[ds][ps]['accuracy'])
            best_acc = results[ds][best_ps]['accuracy']
            default_acc = results[ds].get(1.0, {}).get('accuracy', 0)
            delta = best_acc - default_acc
            print(f"  {ds}: {best_ps}s (Acc={best_acc:.1f}%), delta from 1.0s: {delta:+.1f}%")


if __name__ == "__main__":
    main()
