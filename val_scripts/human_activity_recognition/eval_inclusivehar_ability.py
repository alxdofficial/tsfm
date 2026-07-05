"""Ability-stratified ZS-XD for InclusiveHAR.

Reports HALO zero-shot macro-F1 separately for able-bodied subjects (UserID 1-10)
vs subjects with physical disabilities (UserID 11-20), using the exact same scoring
path as evaluate_tsfm_v2 — just split per-window predictions by ability group. This
delivers the inclusivity claim (does the model recognize activities as well for
people with disabilities as for able-bodied users?).

Usage:
    TSFM_CHECKPOINT=.../best.pt python val_scripts/human_activity_recognition/eval_inclusivehar_ability.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from val_scripts.human_activity_recognition import eval_common as v1
from val_scripts.human_activity_recognition import eval_v2 as ev2
import val_scripts.human_activity_recognition.evaluate_tsfm_v2 as E

DS = "inclusivehar"


def per_window_soft_preds(model, label_bank, raw_data, labels_ld, device, sr, ch_descs, has_gyro):
    """Replicates evaluate_tsfm_v2.evaluate_zs_xd's soft-pooled per-window prediction."""
    with torch.no_grad():
        label_embs = label_bank.encode(labels_ld, normalize=True).to(device)
    patch_embs, patch_masks = v1.extract_tsfm_per_patch_embeddings(
        model, raw_data, device, sampling_rate=sr,
        channel_descriptions=ch_descs, patch_size_sec=v1.PATCH_SIZE_SEC, has_gyro=has_gyro,
    )
    with torch.no_grad():
        le = label_embs
        if le.dim() == 3:
            sims = torch.einsum("npd,lkd->npkl", patch_embs.to(device), le).max(dim=2).values
        else:
            sims = torch.einsum("npd,ld->npl", patch_embs.to(device), le)
    sims = sims.float().cpu().numpy()
    masks = patch_masks.cpu().numpy().astype(bool)
    return ev2.segment_predictions(sims, masks, labels_ld, mode="soft")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.environ.get(
        "TSFM_CHECKPOINT",
        "training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt",
    )
    model, checkpoint, hp = E.load_model(ckpt, device)
    label_bank = E.load_label_bank(checkpoint, device, hp)

    cfg = ev2.load_label_config(DS)
    labels_ld = cfg["labels"]
    idx_to_label = {int(k): v for k, v in cfg["idx_to_label"].items()}
    raw_data, raw_labels, sr = v1.load_raw_data(DS)
    gt_names, subjects, keep_idx = ev2.window_ground_truth(raw_labels, idx_to_label)
    raw_data = raw_data[keep_idx]
    meta = v1.get_dataset_metadata(DS)
    ch_descs = E.channel_descriptions_for(meta, "native", sr)

    preds = per_window_soft_preds(
        model, label_bank, raw_data, labels_ld, device, sr, ch_descs, meta["has_gyro"])

    # subject stored-idx -> raw UserID -> ability (0 able-bodied, 1 disabled)
    md = json.load(open(ROOT / f"benchmark_data/processed/tsfm_eval/{DS}/metadata.json"))
    idx_to_uid = {int(v): int(k) for k, v in md["subject_to_idx"].items()}
    manifest = json.load(open(ROOT / f"data/{DS}/manifest.json"))
    ability = {int(k): int(v) for k, v in manifest.get("ability_by_subject", {}).items()}

    gt = np.array(gt_names)
    pr = np.array(preds)
    subj = np.array(subjects)
    abil = np.array([ability.get(idx_to_uid.get(int(s), -1), -1) for s in subj])

    def score(mask):
        m = ev2.classification_metrics(gt[mask].tolist(), pr[mask].tolist())
        return {
            "f1_macro": round(m["f1_macro"], 2),
            "accuracy": round(m["accuracy"], 2),
            "balanced_accuracy": round(m["balanced_accuracy"], 2),
            "n_windows": int(mask.sum()),
            "n_subjects": int(len(np.unique(subj[mask]))),
        }

    out = {
        "dataset": DS,
        "checkpoint": ckpt,
        "overall": score(np.ones(len(gt), bool)),
        "able_bodied": score(abil == 0),
        "disabled": score(abil == 1),
    }
    out["ability_gap_f1"] = round(out["able_bodied"]["f1_macro"] - out["disabled"]["f1_macro"], 2)

    print("\nInclusiveHAR ability-stratified ZS-XD (HALO, macro-F1):")
    for k in ("overall", "able_bodied", "disabled"):
        s = out[k]
        print(f"  {k:12s}: F1={s['f1_macro']:5.1f}  bAcc={s['balanced_accuracy']:5.1f}  "
              f"Acc={s['accuracy']:5.1f}  ({s['n_windows']} windows, {s['n_subjects']} subj)")
    print(f"  ability gap (able - disabled) F1 = {out['ability_gap_f1']:+.1f}")

    outp = ROOT / "test_output/eval_v2/inclusivehar_ability_stratified.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
