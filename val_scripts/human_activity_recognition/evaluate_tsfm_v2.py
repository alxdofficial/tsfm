"""
TSFM/HALO evaluation under protocol v2 (see eval_v2.py and docs/v2/design_evaluation.md).

Settings per test dataset:
  * ZS-XD  — zero-shot classification against the dataset's own pre-registered
             label strings. Per-patch soft-logit pooling (primary), hard
             majority vote and mean-pooled-session scoring as diagnostics.
             Macro-F1 primary + balanced accuracy, subject-stratified
             bootstrap CIs.
  * FS-1% / FS-10% — end-to-end fine-tuning with SUBJECT-DISJOINT train/val/test
             splits (v1 used a random window split -> subject leakage).

Fairness flags:
  --channel-text {native,neutral}  rich manifest descriptions vs generic ones
  --eval-rate {native,20}          native rate vs anti-aliased 20 Hz resample

Usage:
    TSFM_CHECKPOINT=... python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py
    python ... evaluate_tsfm_v2.py --datasets motionsense shoaib --zs-only
"""

import argparse
import copy
import json
import os
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse model loading, metadata, and embedding extraction from the v1 evaluator.
from val_scripts.human_activity_recognition import evaluate_tsfm as v1
from val_scripts.human_activity_recognition import eval_v2 as ev2
from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank

OUTPUT_DIR = PROJECT_ROOT / "test_output" / "eval_v2"

# v2 default checkpoint = the headline model (v1 pointed at a superseded one).
_DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "training_output" / "semantic_alignment"
    / "small_deep_v2_4b3fdd6" / "best.pt"
)

NEUTRAL_CHANNEL_TEMPLATE = [
    "Accelerometer X-axis",
    "Accelerometer Y-axis",
    "Accelerometer Z-axis",
    "Gyroscope X-axis",
    "Gyroscope Y-axis",
    "Gyroscope Z-axis",
]


# =============================================================================
# Data preparation
# =============================================================================

def resample_windows(data: np.ndarray, sr_from: float, sr_to: float) -> np.ndarray:
    """Anti-aliased polyphase resampling of (N, W, C) windows along time."""
    from scipy.signal import resample_poly
    frac = Fraction(sr_to / sr_from).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return resample_poly(data, up, down, axis=1).astype(np.float32)


def channel_descriptions_for(meta: dict, mode: str, sampling_rate: float) -> List[str]:
    """Channel descriptions under the chosen fairness mode."""
    if mode == "native":
        descs = meta["channel_descriptions"]
        if sampling_rate != meta["sampling_rate_hz"]:
            # Rewrite the rate suffix so the text matches the actual input rate.
            descs = [
                d.replace(
                    f"(sampled at {meta['sampling_rate_hz']:.0f}Hz",
                    f"(sampled at {sampling_rate:.0f}Hz",
                )
                for d in descs
            ]
        return descs
    if mode == "neutral":
        n = meta["n_real_channels"]
        return [
            f"{NEUTRAL_CHANNEL_TEMPLATE[i]} (sampled at {sampling_rate:.0f}Hz, "
            f"{v1.PATCH_SIZE_SEC:.1f}s window)"
            for i in range(n)
        ]
    raise ValueError(f"unknown channel-text mode: {mode}")


# =============================================================================
# Zero-shot (ZS-XD)
# =============================================================================

def evaluate_zs_xd(
    model,
    label_bank,
    raw_data: np.ndarray,
    gt_names: List[str],
    subjects: np.ndarray,
    labels_ld: List[str],
    device: torch.device,
    sampling_rate: float,
    ch_descs: List[str],
    has_gyro: bool,
    bootstrap_B: int,
) -> Dict[str, dict]:
    """Zero-shot against the dataset's own vocabulary L_D."""
    with torch.no_grad():
        label_embs = label_bank.encode(labels_ld, normalize=True).to(device)

    patch_embs, patch_masks = v1.extract_tsfm_per_patch_embeddings(
        model, raw_data, device,
        sampling_rate=sampling_rate,
        channel_descriptions=ch_descs,
        patch_size_sec=v1.PATCH_SIZE_SEC,
        has_gyro=has_gyro,
    )

    # Per-patch similarities (N, P, L) — computed once, pooled multiple ways.
    with torch.no_grad():
        le = label_embs
        if le.dim() == 3:  # multi-prototype (L, K, D) -> max over K later
            sims = torch.einsum("npd,lkd->npkl", patch_embs.to(device), le)
            sims = sims.max(dim=2).values
        else:
            sims = torch.einsum("npd,ld->npl", patch_embs.to(device), le)
    sims = sims.float().cpu().numpy()
    masks = patch_masks.cpu().numpy().astype(bool)

    out = {}
    preds_soft = ev2.segment_predictions(sims, masks, labels_ld, mode="soft")
    m = ev2.classification_metrics(gt_names, preds_soft)
    m.update(ev2.subject_bootstrap_ci(gt_names, preds_soft, subjects, B=bootstrap_B))
    m["per_class_f1"] = ev2.per_class_f1(gt_names, preds_soft)
    out["zs_xd"] = m  # primary

    preds_vote = ev2.segment_predictions(sims, masks, labels_ld, mode="vote")
    out["zs_xd_vote"] = ev2.classification_metrics(gt_names, preds_vote)

    # Mean-pooled session embedding diagnostic
    emb = v1.extract_tsfm_embeddings(
        model, raw_data, device,
        sampling_rate=sampling_rate,
        channel_descriptions=ch_descs,
        patch_size_sec=v1.PATCH_SIZE_SEC,
        has_gyro=has_gyro,
    )
    with torch.no_grad():
        le2 = label_embs if label_embs.dim() == 2 else label_embs.max(dim=1).values
        sess_sims = (torch.from_numpy(emb).to(device).float() @ le2.T).cpu().numpy()
    preds_pool = ev2.predict_from_similarity(sess_sims, labels_ld)
    out["zs_xd_meanpool"] = ev2.classification_metrics(gt_names, preds_pool)

    return out


# =============================================================================
# Few-shot with subject-disjoint splits
# =============================================================================

def evaluate_fewshot_subject_disjoint(
    model,
    label_bank,
    raw_data: np.ndarray,
    gt_names: List[str],
    subjects: np.ndarray,
    labels_ld: List[str],
    device: torch.device,
    sampling_rate: float,
    ch_descs: List[str],
    has_gyro: bool,
    label_rate: float,
    tag: str,
    seed: int,
) -> Dict[str, float]:
    """End-to-end fine-tuning as in v1, but with subject-disjoint splits."""
    name_to_idx = {n: i for i, n in enumerate(labels_ld)}
    y = np.array([name_to_idx[n] for n in gt_names], dtype=np.int64)

    tr_idx, va_idx, te_idx = ev2.subject_disjoint_split(subjects, seed=seed)
    tr_idx = ev2.balanced_subsample_indices(tr_idx, gt_names, rate=label_rate, seed=seed)

    print(f"  [{tag} FT v2] subject-disjoint: train={len(tr_idx)} "
          f"val={len(va_idx)} test={len(te_idx)} windows "
          f"({len(np.unique(subjects[tr_idx]))}/{len(np.unique(subjects[va_idx]))}/"
          f"{len(np.unique(subjects[te_idx]))} subjects)")

    with torch.no_grad():
        text_embs = label_bank.encode(labels_ld, normalize=True).to(device)
        if text_embs.dim() == 3:
            text_embs = text_embs.max(dim=1).values

    ft_model = copy.deepcopy(model)
    ft_model.train()
    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=v1.FINETUNE_ENCODER_LR,
        weight_decay=v1.FINETUNE_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(raw_data[idx]).float(),
            torch.from_numpy(y[idx]).long(),
        )
        return DataLoader(ds, batch_size=v1.FINETUNE_BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(tr_idx, True)
    val_loader = make_loader(va_idx, False)
    test_loader = make_loader(te_idx, False)
    seq_len = raw_data.shape[1]

    best_val, best_state, patience = -1.0, None, 0
    pbar = tqdm(range(v1.FINETUNE_EPOCHS), desc=f"TSFM-v2 | FT {tag}", leave=True)
    for _ in pbar:
        ft_model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            emb = v1._forward_batch(ft_model, batch_data, device, sampling_rate,
                                    ch_descs, seq_len, has_gyro=has_gyro)
            logits = emb @ text_embs.T / v1.FINETUNE_TEMPERATURE
            loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_acc = v1.compute_cosine_accuracy(
            ft_model, val_loader, text_embs, device,
            sampling_rate, ch_descs, seq_len, has_gyro=has_gyro)
        if val_acc > best_val:
            best_val, best_state, patience = val_acc, copy.deepcopy(ft_model.state_dict()), 0
        else:
            patience += 1
        pbar.set_postfix(val_acc=f"{val_acc:.3f}", best=f"{best_val:.3f}")
        if patience >= v1.FINETUNE_PATIENCE:
            break

    if best_state is not None:
        ft_model.load_state_dict(best_state)

    ft_model.train(False)
    preds, gts = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            emb = v1._forward_batch(ft_model, batch_data, device, sampling_rate,
                                    ch_descs, seq_len, has_gyro=has_gyro)
            logits = emb @ text_embs.T / v1.FINETUNE_TEMPERATURE
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            gts.extend(batch_labels.numpy().tolist())

    gt_n = [labels_ld[i] for i in gts]
    pred_n = [labels_ld[i] for i in preds]
    metrics = ev2.classification_metrics(gt_n, pred_n)
    metrics.update(ev2.subject_bootstrap_ci(gt_n, pred_n, subjects[te_idx], B=500))
    metrics["n_train_windows"] = int(len(tr_idx))
    metrics["split"] = "subject_disjoint"

    del best_state, ft_model
    torch.cuda.empty_cache()
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of test datasets (default: all with v2 label configs)")
    ap.add_argument("--checkpoint", default=os.environ.get("TSFM_CHECKPOINT", _DEFAULT_CHECKPOINT))
    ap.add_argument("--channel-text", choices=["native", "neutral"], default="native")
    ap.add_argument("--eval-rate", choices=["native", "20"], default="native")
    ap.add_argument("--zs-only", action="store_true", help="skip few-shot fine-tuning")
    ap.add_argument("--bootstrap", type=int, default=ev2.BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=ev2.BOOTSTRAP_SEED)
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Protocol: v2 | checkpoint: {args.checkpoint}")
    print(f"channel-text={args.channel_text} eval-rate={args.eval_rate}")
    model, checkpoint, hyperparams_path = load_model(args.checkpoint, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)

    datasets = args.datasets or [
        p.stem for p in sorted(ev2.LABEL_CONFIG_DIR.glob("*.json"))
    ]

    all_results = {"_protocol": {
        "version": "v2",
        "checkpoint": args.checkpoint,
        "channel_text": args.channel_text,
        "eval_rate": args.eval_rate,
        "pooling": "soft_logit(tau=%.2f)" % ev2.SOFT_POOL_TAU,
        "primary_metric": "f1_macro over GT-present classes",
        "splits": "zero-shot: none needed; few-shot: subject-disjoint",
        "seed": args.seed,
    }}

    for ds in datasets:
        print(f"\n{'='*60}\nTSFM v2 | {ds}\n{'='*60}")
        cfg = ev2.load_label_config(ds)
        labels_ld = cfg["labels"]
        idx_to_label = {int(k): v for k, v in cfg["idx_to_label"].items()}

        raw_data, raw_labels, sr = v1.load_raw_data(ds)
        gt_names, subjects, keep_idx = ev2.window_ground_truth(raw_labels, idx_to_label)
        raw_data = raw_data[keep_idx]

        meta = v1.get_dataset_metadata(ds)
        eval_sr = sr
        if args.eval_rate == "20" and sr != 20.0:
            raw_data = resample_windows(raw_data, sr, 20.0)
            eval_sr = 20.0
        ch_descs = channel_descriptions_for(meta, args.channel_text, eval_sr)

        print(f"  {len(gt_names)} windows | {len(labels_ld)} classes | "
              f"{len(np.unique(subjects))} subjects | rate {eval_sr:.0f}Hz")

        ds_results = {"sampling_rate_hz": eval_sr, "n_windows": len(gt_names)}
        ds_results.update(evaluate_zs_xd(
            model, label_bank, raw_data, gt_names, subjects, labels_ld, device,
            eval_sr, ch_descs, meta["has_gyro"], args.bootstrap))

        zs = ds_results["zs_xd"]
        print(f"  ZS-XD (soft): F1={zs['f1_macro']:.1f} "
              f"[{zs['f1_macro_ci_lo']:.1f},{zs['f1_macro_ci_hi']:.1f}] "
              f"bAcc={zs['balanced_accuracy']:.1f} Acc={zs['accuracy']:.1f}")
        print(f"  ZS-XD (vote): F1={ds_results['zs_xd_vote']['f1_macro']:.1f} | "
              f"(meanpool): F1={ds_results['zs_xd_meanpool']['f1_macro']:.1f}")

        if not args.zs_only:
            for rate, tag in ((0.01, "1pct"), (0.10, "10pct")):
                ds_results[f"fs_{tag}"] = evaluate_fewshot_subject_disjoint(
                    model, label_bank, raw_data, gt_names, subjects, labels_ld,
                    device, eval_sr, ch_descs, meta["has_gyro"],
                    label_rate=rate, tag=tag, seed=args.seed)
                fs = ds_results[f"fs_{tag}"]
                print(f"  FS-{tag}: F1={fs['f1_macro']:.1f} Acc={fs['accuracy']:.1f} "
                      f"(subject-disjoint)")

        all_results[ds] = ds_results

    out_path = Path(args.out) if args.out else (
        OUTPUT_DIR / f"tsfm_v2_{args.channel_text}_{args.eval_rate}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
