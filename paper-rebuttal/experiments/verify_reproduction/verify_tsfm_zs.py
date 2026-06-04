"""
Verification harness (read-only): does the CURRENT branch's code load the
small_deep_v2 checkpoint and reproduce the deployed zero-shot numbers?

ZS uses ALL test windows (no split / no subsampling / no augmentation), so it is
deterministic given the weights — any mismatch means the master refactor broke
checkpoint loading or the forward path. We compare against the committed reference
JSON test_output/baseline_evaluation/tsfm_evaluation_small_deep_v2.json.

Run:
    TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt \
    ./.venv/bin/python paper-rebuttal/experiments/verify_reproduction/verify_tsfm_zs.py [ds1 ds2 ...]
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Default the checkpoint to small_deep_v2 unless overridden.
os.environ.setdefault(
    "TSFM_CHECKPOINT",
    str(ROOT / "training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt"),
)

import val_scripts.human_activity_recognition.evaluate_tsfm as E  # noqa: E402

REF_PATH = ROOT / "test_output/baseline_evaluation/tsfm_evaluation_small_deep_v2.json"
REF = json.load(open(REF_PATH))

# Subset chosen for speed + coverage: easy/hard/placement/OOD.
DEFAULT_DS = ["motionsense", "mobiact", "shoaib", "vtt_coniot", "opportunity"]
TOL = 0.5  # pp; deterministic ZS should match to ~0


def run_ds(model, label_bank, is_per_patch, device, ds):
    meta = E.get_dataset_metadata(ds)
    raw_data, raw_labels, sr = E.load_raw_data(ds)
    test_labels = E.get_window_labels(raw_labels)
    ch_descs = meta["channel_descriptions"]
    has_gyro = meta["has_gyro"]

    test_emb = E.extract_tsfm_embeddings(
        model, raw_data, device, sampling_rate=sr,
        channel_descriptions=ch_descs, patch_size_sec=E.PATCH_SIZE_SEC, has_gyro=has_gyro,
    )
    out = {}
    out["zero_shot_open_set"] = E.evaluate_zero_shot_open_set(
        test_emb, test_labels, ds, label_bank, device)
    out["zero_shot_closed_set"] = E.evaluate_zero_shot_closed_set(
        test_emb, test_labels, ds, label_bank, device)
    if is_per_patch:
        patch_embs, patch_masks = E.extract_tsfm_per_patch_embeddings(
            model, raw_data, device, sampling_rate=sr,
            channel_descriptions=ch_descs, patch_size_sec=E.PATCH_SIZE_SEC, has_gyro=has_gyro,
        )
        out["zero_shot_open_set_mv"] = E.evaluate_zero_shot_majority_vote(
            patch_embs, patch_masks, test_labels, ds, label_bank, device, open_set=True)
        out["zero_shot_closed_set_mv"] = E.evaluate_zero_shot_majority_vote(
            patch_embs, patch_masks, test_labels, ds, label_bank, device, open_set=False)
    return out


def main():
    datasets = sys.argv[1:] or DEFAULT_DS
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {E.CHECKPOINT_PATH}")
    print(f"Reference : {REF_PATH.name}\nDevice    : {device}\n")

    model, checkpoint, hp = E.load_tsfm_model(E.CHECKPOINT_PATH, device)
    label_bank = E.load_label_bank(checkpoint, device, hp)
    is_per_patch = hasattr(model, "semantic_head") and model.semantic_head.per_patch_prediction
    print(f"Loaded OK. semantic_dim={model.semantic_dim}, per_patch={is_per_patch}\n")

    keys = ["zero_shot_open_set", "zero_shot_open_set_mv",
            "zero_shot_closed_set", "zero_shot_closed_set_mv"]
    print(f"{'dataset':<14}{'metric':<26}{'got':>8}{'ref':>8}{'Δ':>8}  status")
    print("-" * 80)
    worst = 0.0
    for ds in datasets:
        got = run_ds(model, label_bank, is_per_patch, device, ds)
        for k in keys:
            if k not in got or k not in REF.get(ds, {}):
                continue
            g = got[k]["accuracy"]; r = REF[ds][k]["accuracy"]; d = g - r
            worst = max(worst, abs(d))
            status = "OK" if abs(d) <= TOL else "*** MISMATCH ***"
            print(f"{ds:<14}{k:<26}{g:>8.2f}{r:>8.2f}{d:>+8.2f}  {status}")
        print()
    print("-" * 80)
    verdict = "PASS" if worst <= TOL else "FAIL"
    print(f"VERDICT: {verdict}  (max |Δ| = {worst:.3f} pp, tol = {TOL})")


if __name__ == "__main__":
    main()
