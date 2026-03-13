"""
HARTH Failure Analysis for Paper.

Generates all data and figures needed to explain why zero-shot fails on HARTH
(back/thigh-mounted accelerometers only, no gyro, gravity-contaminated) while
supervised recovery is strong (78.3% at 10%).

Outputs:
  1. UMAP embedding visualization (HARTH vs training data)
  2. Confusion matrix (zero-shot closed-set)
  3. Cosine similarity distributions
  4. Supervised recovery curve (0.1% to 10%)

Usage:
    python val_scripts/human_activity_recognition/harth_analysis.py
"""

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models"))

from val_scripts.human_activity_recognition.model_loading import load_model as load_tsfm_model, load_label_bank
from val_scripts.human_activity_recognition.evaluation_metrics import compute_similarity
from val_scripts.human_activity_recognition.evaluate_tsfm import (
    get_dataset_metadata, load_raw_data, get_window_labels, get_dataset_labels,
    extract_tsfm_embeddings, extract_tsfm_per_patch_embeddings,
    prepare_train_test_split, _forward_batch,
    GLOBAL_LABELS, DATASET_CONFIG, DATA_CHANNELS, PATCH_SIZE_SEC,
    CLASSIFIER_SEED, TRAINING_RATE, VALI_RATE,
    FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, FINETUNE_ENCODER_LR,
    FINETUNE_WEIGHT_DECAY, FINETUNE_PATIENCE, FINETUNE_TEMPERATURE,
)
from model.token_text_encoder import LearnableLabelBank

# ---- Config ----
CHECKPOINT_PATH = os.environ.get(
    "TSFM_CHECKPOINT",
    str(PROJECT_ROOT / "training_output" / "semantic_alignment" / "small_deep_v2_4b3fdd6" / "best.pt"),
)
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "harth_analysis"
HARTH_DS = "harth"
# Training datasets to compare embeddings against
COMPARISON_TRAIN_DS = ["uci_har", "pamap2", "wisdm"]
# Supervised fractions for recovery curve
SUPERVISED_FRACTIONS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
# Max training samples per dataset for embedding comparison (to keep UMAP manageable)
MAX_EMBED_SAMPLES = 2000


def extract_harth_analysis_data(model, label_bank, device):
    """Phase 1: Extract all raw data from GPU. Returns dict of numpy arrays."""
    results = {}

    # ---- Load HARTH data ----
    meta = get_dataset_metadata(HARTH_DS)
    raw_data, raw_labels, sr = load_raw_data(HARTH_DS)
    test_labels = get_window_labels(raw_labels)
    test_activities = get_dataset_labels(HARTH_DS)
    ch_descs = meta['channel_descriptions']
    has_gyro = meta['has_gyro']

    print(f"\nHARTH: {raw_data.shape[0]} windows, {len(test_activities)} classes, "
          f"sr={sr}Hz, has_gyro={has_gyro}")
    print(f"  Classes: {test_activities}")
    print(f"  Channel descriptions: {ch_descs}")

    # ---- Extract HARTH embeddings ----
    print("\n[1/6] Extracting HARTH embeddings...")
    harth_emb = extract_tsfm_embeddings(
        model, raw_data, device, sampling_rate=sr,
        channel_descriptions=ch_descs, patch_size_sec=PATCH_SIZE_SEC,
        has_gyro=has_gyro,
    )
    results['harth_embeddings'] = harth_emb
    results['harth_labels'] = test_labels
    results['harth_activities'] = test_activities
    results['harth_n_samples'] = len(test_labels)

    # ---- Extract training dataset embeddings for comparison ----
    print("\n[2/6] Extracting training dataset embeddings for comparison...")
    train_embeddings_all = []
    train_labels_all = []
    train_dataset_ids = []

    for ds in COMPARISON_TRAIN_DS:
        print(f"  Loading {ds}...")
        ds_meta = get_dataset_metadata(ds)
        ds_data, ds_labels_raw, ds_sr = load_raw_data(ds)
        ds_labels = get_window_labels(ds_labels_raw)
        ds_activities = get_dataset_labels(ds)

        # Subsample for UMAP
        n = min(MAX_EMBED_SAMPLES, len(ds_data))
        idx = np.random.choice(len(ds_data), n, replace=False)
        ds_data_sub = ds_data[idx]
        ds_labels_sub = ds_labels[idx]

        ds_emb = extract_tsfm_embeddings(
            model, ds_data_sub, device, sampling_rate=ds_sr,
            channel_descriptions=ds_meta['channel_descriptions'],
            patch_size_sec=PATCH_SIZE_SEC, has_gyro=ds_meta['has_gyro'],
        )
        train_embeddings_all.append(ds_emb)
        # Store labels as activity names
        ds_label_names = [ds_activities[l] for l in ds_labels_sub]
        train_labels_all.extend(ds_label_names)
        train_dataset_ids.extend([ds] * n)
        print(f"    {ds}: {n} samples, {ds_emb.shape}")

    results['train_embeddings'] = np.concatenate(train_embeddings_all, axis=0)
    results['train_label_names'] = train_labels_all
    results['train_dataset_ids'] = train_dataset_ids

    # ---- Encode text label embeddings ----
    print("\n[3/6] Encoding text label embeddings...")
    with torch.no_grad():
        # All 87 training labels
        all_label_emb = label_bank.encode(GLOBAL_LABELS, normalize=True).cpu().numpy()
        # HARTH-specific labels
        harth_label_emb = label_bank.encode(test_activities, normalize=True).cpu().numpy()

    results['all_label_embeddings'] = all_label_emb
    results['all_label_names'] = GLOBAL_LABELS
    results['harth_label_embeddings'] = harth_label_emb

    # ---- Zero-shot predictions + similarity matrix ----
    print("\n[4/6] Computing zero-shot predictions and similarities...")
    harth_emb_t = torch.from_numpy(harth_emb).float().to(device)
    harth_label_emb_t = torch.from_numpy(harth_label_emb).float().to(device)
    all_label_emb_t = torch.from_numpy(all_label_emb).float().to(device)

    # Closed-set similarities (N, 12)
    closed_sims = compute_similarity(harth_emb_t, harth_label_emb_t).cpu().numpy()
    closed_preds = closed_sims.argmax(axis=1)

    # Open-set similarities (N, 87)
    open_sims = compute_similarity(harth_emb_t, all_label_emb_t).cpu().numpy()
    open_preds = open_sims.argmax(axis=1)

    results['closed_set_similarities'] = closed_sims
    results['closed_set_predictions'] = closed_preds
    results['open_set_similarities'] = open_sims
    results['open_set_predictions'] = open_preds

    # ---- Supervised recovery curve ----
    print("\n[5/6] Running supervised recovery curve...")
    recovery_results = {}
    for frac in SUPERVISED_FRACTIONS:
        label_tag = f"{frac*100:.1f}%"
        print(f"\n  --- {label_tag} supervised ---")
        result = run_supervised_finetune(
            model, label_bank, raw_data, test_labels, device,
            sampling_rate=sr, channel_descriptions=ch_descs,
            label_rate=frac, label_tag=label_tag, has_gyro=has_gyro,
            test_activities=test_activities,
        )
        recovery_results[label_tag] = result
        print(f"  {label_tag}: Acc={result['accuracy']:.1f}%, F1={result['f1_macro']:.1f}%, "
              f"n_train={result['n_train_samples']}")

    results['recovery_curve'] = recovery_results

    # ---- Embedding stats ----
    print("\n[6/6] Computing embedding statistics...")
    harth_norms = np.linalg.norm(harth_emb, axis=1)
    train_norms = np.linalg.norm(results['train_embeddings'], axis=1)
    results['harth_embedding_norms'] = harth_norms
    results['train_embedding_norms'] = train_norms

    # Cross-domain cosine similarity (HARTH centroid vs training centroids)
    harth_centroid = harth_emb.mean(axis=0)
    harth_centroid /= np.linalg.norm(harth_centroid)
    train_centroid = results['train_embeddings'].mean(axis=0)
    train_centroid /= np.linalg.norm(train_centroid)
    results['centroid_cosine_sim'] = float(harth_centroid @ train_centroid)

    return results


def run_supervised_finetune(model, label_bank, raw_data, test_labels, device,
                            sampling_rate, channel_descriptions, label_rate,
                            label_tag, has_gyro, test_activities):
    """Run supervised fine-tuning at a given label fraction. Returns metrics dict."""
    num_test_classes = len(test_activities)

    train_data, train_labels_arr, val_data, val_labels_arr, test_data, test_labels_arr = \
        prepare_train_test_split(
            raw_data, test_labels,
            training_rate=TRAINING_RATE, vali_rate=VALI_RATE,
            label_rate=label_rate, seed=CLASSIFIER_SEED, balance=True,
        )

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0,
                'n_samples': 0, 'n_train_samples': 0, 'n_classes': num_test_classes}

    with torch.no_grad():
        text_embs = label_bank.encode(test_activities, normalize=True).to(device)

    ft_model = copy.deepcopy(model)
    ft_model.train()
    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=FINETUNE_ENCODER_LR, weight_decay=FINETUNE_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_ds = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels_arr).long())
    val_ds = TensorDataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_labels_arr).long())
    test_ds = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels_arr).long())
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    seq_len = raw_data.shape[1]

    pbar = tqdm(range(FINETUNE_EPOCHS), desc=f"FT {label_tag}", leave=False)
    for epoch in pbar:
        ft_model.train()
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            emb = _forward_batch(ft_model, batch_data, device, sampling_rate,
                                 channel_descriptions, seq_len, has_gyro=has_gyro)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
            loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        ft_model.train(False)
        correct, total = 0, 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                emb = _forward_batch(ft_model, batch_data, device, sampling_rate,
                                     channel_descriptions, seq_len, has_gyro=has_gyro)
                logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)
        val_acc = correct / max(total, 1) * 100
        pbar.set_postfix(val_acc=f"{val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(ft_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_PATIENCE:
                break

    # Test with best model
    if best_state is not None:
        ft_model.load_state_dict(best_state)
    ft_model.train(False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            emb = _forward_batch(ft_model, batch_data, device, sampling_rate,
                                 channel_descriptions, seq_len, has_gyro=has_gyro)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    del ft_model
    torch.cuda.empty_cache()

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w,
            'n_samples': len(all_labels), 'n_train_samples': len(train_data),
            'n_classes': num_test_classes}


def generate_figures(results, output_dir):
    """Phase 2: Generate all figures from saved data (CPU only)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from umap import UMAP

    output_dir.mkdir(parents=True, exist_ok=True)
    harth_activities = results['harth_activities']

    # =====================================================================
    # Figure 1: UMAP Embedding Visualization
    # =====================================================================
    print("\nGenerating UMAP embedding visualization...")
    harth_emb = results['harth_embeddings']
    train_emb = results['train_embeddings']
    all_label_emb = results['all_label_embeddings']

    # Subsample HARTH for UMAP
    n_harth = min(MAX_EMBED_SAMPLES, len(harth_emb))
    harth_idx = np.random.choice(len(harth_emb), n_harth, replace=False)
    harth_emb_sub = harth_emb[harth_idx]
    harth_labels_sub = results['harth_labels'][harth_idx]

    # Combine all embeddings for joint UMAP
    combined = np.concatenate([train_emb, harth_emb_sub, all_label_emb], axis=0)
    n_train = len(train_emb)
    n_harth_s = len(harth_emb_sub)

    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    proj = reducer.fit_transform(combined)

    train_proj = proj[:n_train]
    harth_proj = proj[n_train:n_train + n_harth_s]
    label_proj = proj[n_train + n_harth_s:]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot training data (light gray, small)
    ax.scatter(train_proj[:, 0], train_proj[:, 1], c='#cccccc', s=3, alpha=0.3,
               label=f'Training data ({n_train} samples)', zorder=1)

    # Plot HARTH data (colored by activity)
    cmap = plt.cm.tab20
    unique_labels = np.unique(harth_labels_sub)
    for i, label_idx in enumerate(unique_labels):
        mask = harth_labels_sub == label_idx
        name = harth_activities[label_idx] if label_idx < len(harth_activities) else f"class_{label_idx}"
        ax.scatter(harth_proj[mask, 0], harth_proj[mask, 1],
                   c=[cmap(i / max(len(unique_labels) - 1, 1))], s=15, alpha=0.6,
                   label=f'HARTH: {name}', zorder=2)

    # Plot text label anchors
    for i, name in enumerate(GLOBAL_LABELS):
        color = '#ff4444' if name in harth_activities else '#888888'
        size = 80 if name in harth_activities else 30
        marker = '*' if name in harth_activities else 'D'
        ax.scatter(label_proj[i, 0], label_proj[i, 1], c=color, s=size,
                   marker=marker, edgecolors='black', linewidth=0.5, zorder=3)
        if name in harth_activities:
            ax.annotate(name, (label_proj[i, 0], label_proj[i, 1]),
                        fontsize=6, ha='center', va='bottom', color='#ff4444',
                        fontweight='bold')

    ax.set_title('HARTH Embeddings vs Training Data in Shared Embedding Space\n'
                 '(Gray=training, Colored=HARTH, Stars=HARTH text labels, Diamonds=other text labels)',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=7, markerscale=2, framealpha=0.8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    fig.savefig(output_dir / 'umap_harth_vs_training.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved umap_harth_vs_training.png")

    # =====================================================================
    # Figure 2: Confusion Matrix (Zero-Shot Closed-Set)
    # =====================================================================
    print("Generating confusion matrix...")
    gt_labels = results['harth_labels']
    pred_labels = results['closed_set_predictions']

    cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(harth_activities)))
    # Normalize by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of true class')

    ax.set_xticks(range(len(harth_activities)))
    ax.set_yticks(range(len(harth_activities)))
    ax.set_xticklabels(harth_activities, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(harth_activities, fontsize=8)

    # Annotate cells
    for i in range(len(harth_activities)):
        for j in range(len(harth_activities)):
            val = cm_norm[i, j]
            count = cm[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}\n({count})', ha='center', va='center',
                        fontsize=6, color=color)

    ax.set_xlabel('Predicted Activity', fontsize=11)
    ax.set_ylabel('True Activity', fontsize=11)
    ax.set_title('HARTH Zero-Shot Closed-Set Confusion Matrix\n'
                 '(Row-normalized: fraction of each true class predicted as each label)',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix_harth.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved confusion_matrix_harth.png")

    # Also save which label gets the most predictions overall
    pred_counts = np.bincount(pred_labels, minlength=len(harth_activities))
    print(f"  Prediction distribution: {dict(zip(harth_activities, pred_counts.tolist()))}")

    # =====================================================================
    # Figure 3: Cosine Similarity Distributions
    # =====================================================================
    print("Generating similarity distributions...")
    closed_sims = results['closed_set_similarities']

    # For each sample, get similarity of correct label vs max wrong label
    correct_sims = []
    max_wrong_sims = []
    max_overall_sims = []
    for i in range(len(gt_labels)):
        gt = gt_labels[i]
        sims = closed_sims[i]
        correct_sims.append(sims[gt])
        wrong_mask = np.ones(len(sims), dtype=bool)
        wrong_mask[gt] = False
        max_wrong_sims.append(sims[wrong_mask].max())
        max_overall_sims.append(sims.max())

    correct_sims = np.array(correct_sims)
    max_wrong_sims = np.array(max_wrong_sims)
    max_overall_sims = np.array(max_overall_sims)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Distribution of all similarities
    ax = axes[0]
    ax.hist(closed_sims.flatten(), bins=100, color='#4488cc', alpha=0.7, density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('All Pairwise Similarities\n(HARTH embeddings vs HARTH text labels)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Correct vs Max-Wrong
    ax = axes[1]
    ax.hist(correct_sims, bins=80, alpha=0.6, color='#22aa44', label='Correct label sim', density=True)
    ax.hist(max_wrong_sims, bins=80, alpha=0.6, color='#cc4444', label='Max wrong label sim', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Correct Label vs Max Wrong Label\n(Overlap = model cannot distinguish)')
    ax.legend()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel C: Margin distribution (correct - max_wrong)
    ax = axes[2]
    margins = correct_sims - max_wrong_sims
    ax.hist(margins, bins=80, color='#cc8844', alpha=0.7, density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    frac_correct = (margins > 0).mean()
    ax.set_xlabel('Margin (correct sim - max wrong sim)')
    ax.set_ylabel('Density')
    ax.set_title(f'Similarity Margin Distribution\n({frac_correct:.1%} have positive margin)')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'similarity_distributions.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved similarity_distributions.png")
    print(f"  Mean correct sim: {correct_sims.mean():.4f}, Mean max-wrong sim: {max_wrong_sims.mean():.4f}")
    print(f"  Mean margin: {margins.mean():.4f}, Fraction positive: {frac_correct:.4f}")

    # =====================================================================
    # Figure 4: Supervised Recovery Curve
    # =====================================================================
    print("Generating recovery curve...")
    recovery = results['recovery_curve']

    fracs = []
    accs = []
    f1s = []
    n_trains = []
    for label_tag in sorted(recovery.keys(), key=lambda x: float(x.strip('%'))):
        r = recovery[label_tag]
        fracs.append(float(label_tag.strip('%')))
        accs.append(r['accuracy'])
        f1s.append(r['f1_macro'])
        n_trains.append(r['n_train_samples'])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # HALO recovery curve
    ax1.plot(fracs, accs, 'o-', color='#2266cc', linewidth=2.5, markersize=8,
             label='HALO (ours) Accuracy', zorder=3)
    ax1.plot(fracs, f1s, 's--', color='#2266cc', linewidth=1.5, markersize=6,
             alpha=0.7, label='HALO (ours) F1 Macro', zorder=3)

    # Add zero-shot baseline (horizontal line)
    zs_acc = results.get('zs_closed_acc', 2.5)
    ax1.axhline(y=zs_acc, color='#cc4444', linestyle=':', linewidth=1.5,
                label=f'HALO zero-shot ({zs_acc:.1f}%)')

    # Annotate n_train on each point
    for i, (f, a, n) in enumerate(zip(fracs, accs, n_trains)):
        ax1.annotate(f'n={n}', (f, a), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=7, color='#2266cc')

    ax1.set_xlabel('Labeled Data Fraction (%)', fontsize=12)
    ax1.set_ylabel('Accuracy / F1 (%)', fontsize=12)
    ax1.set_title('HARTH Supervised Recovery Curve\n'
                  '(From 2.5% zero-shot to 78.3% at 10%: representations are strong, alignment needs recalibration)',
                  fontsize=11)
    ax1.set_xscale('log')
    ax1.set_xticks(fracs)
    ax1.set_xticklabels([f'{f}%' for f in fracs])
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'recovery_curve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved recovery_curve.png")

    # =====================================================================
    # Save numerical summary
    # =====================================================================
    summary = {
        'harth_n_samples': results['harth_n_samples'],
        'harth_n_classes': len(harth_activities),
        'harth_activities': harth_activities,
        'centroid_cosine_similarity': results['centroid_cosine_sim'],
        'embedding_norms': {
            'harth_mean': float(results['harth_embedding_norms'].mean()),
            'harth_std': float(results['harth_embedding_norms'].std()),
            'train_mean': float(results['train_embedding_norms'].mean()),
            'train_std': float(results['train_embedding_norms'].std()),
        },
        'zero_shot_closed_set': {
            'accuracy': float(accuracy_score(gt_labels, pred_labels) * 100),
            'prediction_distribution': dict(zip(harth_activities, pred_counts.tolist())),
        },
        'similarity_stats': {
            'mean_correct_sim': float(correct_sims.mean()),
            'std_correct_sim': float(correct_sims.std()),
            'mean_max_wrong_sim': float(max_wrong_sims.mean()),
            'std_max_wrong_sim': float(max_wrong_sims.std()),
            'mean_margin': float(margins.mean()),
            'fraction_positive_margin': float(frac_correct),
            'mean_max_overall_sim': float(max_overall_sims.mean()),
        },
        'recovery_curve': {k: {
            'accuracy': v['accuracy'],
            'f1_macro': v['f1_macro'],
            'n_train_samples': v['n_train_samples'],
        } for k, v in recovery.items()},
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved analysis_summary.json")

    return summary


def validate_outputs(results, summary, output_dir):
    """Phase 3: Validate all outputs for correctness."""
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    errors = []

    # Check 1: Embedding shapes
    harth_emb = results['harth_embeddings']
    n_harth = results['harth_n_samples']
    if harth_emb.shape[0] != n_harth:
        errors.append(f"HARTH embedding count mismatch: {harth_emb.shape[0]} vs {n_harth}")
    else:
        print(f"  [OK] HARTH embeddings: {harth_emb.shape}")

    # Check 2: Embeddings are L2-normalized
    norms = np.linalg.norm(harth_emb, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01):
        errors.append(f"HARTH embeddings not normalized: mean norm={norms.mean():.4f}")
    else:
        print(f"  [OK] HARTH embeddings normalized (mean norm={norms.mean():.4f})")

    # Check 3: Confusion matrix rows sum to sample count per class
    gt_labels = results['harth_labels']
    pred_labels = results['closed_set_predictions']
    cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(results['harth_activities'])))
    if cm.sum() != len(gt_labels):
        errors.append(f"Confusion matrix total {cm.sum()} != {len(gt_labels)}")
    else:
        print(f"  [OK] Confusion matrix sums to {cm.sum()} (= total samples)")

    # Check 4: Similarity range
    closed_sims = results['closed_set_similarities']
    if closed_sims.min() < -1.01 or closed_sims.max() > 1.01:
        errors.append(f"Similarities out of [-1,1]: [{closed_sims.min():.4f}, {closed_sims.max():.4f}]")
    else:
        print(f"  [OK] Similarities in valid range [{closed_sims.min():.4f}, {closed_sims.max():.4f}]")

    # Check 5: Recovery curve is monotonically non-decreasing (allow small noise)
    recovery = results['recovery_curve']
    sorted_fracs = sorted(recovery.keys(), key=lambda x: float(x.strip('%')))
    prev_acc = 0
    monotonic = True
    for frac in sorted_fracs:
        acc = recovery[frac]['accuracy']
        if acc < prev_acc - 5:  # Allow small noise
            monotonic = False
            errors.append(f"Recovery curve not monotonic: {frac}={acc:.1f}% < prev={prev_acc:.1f}%")
        prev_acc = acc
    if monotonic:
        print(f"  [OK] Recovery curve approximately monotonic")

    # Check 6: Zero-shot accuracy matches expected (should be ~2.5%)
    zs_acc = accuracy_score(gt_labels, pred_labels) * 100
    if zs_acc > 15:
        errors.append(f"Zero-shot accuracy suspiciously high: {zs_acc:.1f}% (expected ~2.5%)")
    else:
        print(f"  [OK] Zero-shot accuracy {zs_acc:.1f}% (expected near-zero)")

    # Check 7: Output files exist
    expected_files = ['umap_harth_vs_training.png', 'confusion_matrix_harth.png',
                      'similarity_distributions.png', 'recovery_curve.png', 'analysis_summary.json']
    for f in expected_files:
        path = output_dir / f
        if not path.exists():
            errors.append(f"Missing output file: {f}")
        else:
            size = path.stat().st_size
            print(f"  [OK] {f} ({size / 1024:.0f} KB)")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    [FAIL] {e}")
    else:
        print(f"\n  All {7} checks passed!")

    return len(errors) == 0


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading TSFM model from {CHECKPOINT_PATH}...")
    model, checkpoint, hyperparams_path = load_tsfm_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    print("Model and label bank loaded")

    # Phase 1: GPU - extract all data
    print("\n" + "=" * 60)
    print("PHASE 1: Extracting data (GPU)")
    print("=" * 60)
    results = extract_harth_analysis_data(model, label_bank, device)

    # Compute zero-shot accuracy for reference
    gt = results['harth_labels']
    pred = results['closed_set_predictions']
    results['zs_closed_acc'] = accuracy_score(gt, pred) * 100

    # Free GPU memory before figures
    del model, label_bank, checkpoint
    torch.cuda.empty_cache()

    # Phase 2: CPU - generate figures
    print("\n" + "=" * 60)
    print("PHASE 2: Generating figures (CPU)")
    print("=" * 60)
    summary = generate_figures(results, OUTPUT_DIR)

    # Phase 3: Validate
    valid = validate_outputs(results, summary, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"HARTH analysis complete. Outputs in: {OUTPUT_DIR}")
    print(f"Validation: {'PASSED' if valid else 'FAILED'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
