"""
Benchmark a trained model against published baselines.

Evaluates a single model against NLS-HAR, GOAT, LanHAR, and CrossHAR
baselines with matching evaluation protocols per paper.

Modes:
  1. NLS-HAR: Closed-set macro F1 on MotionSense, MobiAct
  2. GOAT: Closed-set macro F1 on RealWorld, Realdisp, Opportunity, Daphnet FoG
  3. LanHAR: 4-activity subset macro F1 on MotionSense, Shoaib
  4. CrossHAR: 4-activity subset accuracy on MotionSense, Shoaib

Usage:
    python val_scripts/human_activity_recognition/benchmark_baselines.py
"""

import torch
from torch.amp import autocast
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from torch.utils.data import DataLoader

from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from datasets.imu_pretraining_dataset.multi_dataset_loader import IMUPretrainingDataset
from imu_activity_recognition_encoder.token_text_encoder import LearnableLabelBank
from datasets.imu_pretraining_dataset.label_augmentation import DATASET_CONFIGS
from datasets.imu_pretraining_dataset.label_groups import LABEL_GROUPS, get_group_for_label

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint to benchmark
CHECKPOINT_PATH = "training_output/semantic_alignment/20260124_033735/best.pt"

# Patch sizes per dataset (match training config)
PATCH_SIZE_PER_DATASET = {
    'motionsense': 1.5,
    'mobiact': 1.5,
    'realworld': 1.5,
    'shoaib': 1.5,
    'opportunity': 1.5,
    'realdisp': 1.5,
    'daphnet_fog': 1.5,
}

# Evaluation settings
BATCH_SIZE = 32
MAX_SESSIONS_PER_DATASET = 10000

# Output directory
OUTPUT_DIR = "test_output/benchmark_baselines"

# =============================================================================
# Published Baselines
# =============================================================================

BASELINES = {
    # NLS-HAR (AAAI 2025): True zero-shot, DistilBERT text encoder,
    # template: "This is wearable sensor data for a person engaged in {activity}"
    # Closed-set argmax over dataset's own labels, exact match, macro F1.
    # Most direct comparison — both use zero-shot text-aligned embeddings.
    'nls_har': {
        'name': 'NLS-HAR',
        'metric': 'Closed-Set Macro F1',
        'protocol': 'zero-shot',
        'datasets': {
            'motionsense': {'f1': 38.97},
            'mobiact': {'f1': 16.93},
        },
    },
    # GOAT (IMWUT 2024): NOT zero-shot — fine-tunes on ~7% of target data.
    # Uses raw label strings, closed-set argmax, exact match, macro F1.
    # Best results per dataset use DIFFERENT text encoders (CLIP vs BERT).
    # Note: Opportunity (17 gesture activities) and Daphnet FoG (3 classes
    # including "Null") use fundamentally different label sets than ours,
    # making direct comparison invalid. Only RealWorld and Realdisp are comparable.
    'goat': {
        'name': 'GOAT',
        'metric': 'Closed-Set Macro F1',
        'protocol': 'few-shot (fine-tunes on ~7% target data)',
        'datasets': {
            'realworld': {'f1': 78.49},   # GOAT-CLIP
            'realdisp': {'f1': 81.39},     # GOAT-CLIP
            # Opportunity EXCLUDED: GOAT uses 17 gesture activities (Open door 1,
            # Close fridge, etc.) vs our 4 locomotion activities. Incomparable.
            # Daphnet FoG EXCLUDED: GOAT uses 3 classes (Null, No freeze, Freeze)
            # vs our 2 classes (walking, freezing_gait). Incomparable.
        },
    },
    # LanHAR (IMWUT 2025): Single-source transfer (train on 1 dataset, test on another).
    # Uses GPT-4-generated descriptions, BERT encoder, embedding similarity.
    # 4-activity protocol: walking, going upstairs, going downstairs, sitting.
    # Numbers are from specific source->target pairs (UCI->Motion, UCI->Shoaib).
    'lanhar': {
        'name': 'LanHAR',
        'metric': '4-Activity Macro F1',
        'protocol': 'single-source transfer (train on 1 dataset)',
        'datasets': {
            'motionsense': {'f1': 76.0},   # UCI->MotionSense (best single-source)
            'shoaib': {'f1': 71.2},         # UCI->Shoaib
        },
    },
    # CrossHAR (via HAR-DoReMi, IMWUT 2024): Multi-source supervised (3 source datasets).
    # No text encoder — purely sensor-based self-supervised + supervised fine-tuning.
    # Uses "Still" (sitting+standing MERGED), Walking, Upstairs, Downstairs.
    # Numbers from HAR-DoReMi reproduction, NOT original CrossHAR paper.
    'crosshar': {
        'name': 'CrossHAR',
        'metric': '4-Activity Accuracy',
        'protocol': 'multi-source supervised (train on 3 datasets)',
        'datasets': {
            'motionsense': {'accuracy': 78.26},  # HSU->Motion
            'shoaib': {'accuracy': 73.67},         # HMU->Shoaib
        },
    },
}

# LanHAR 4-activity protocol: walking, upstairs, downstairs, sitting
# (sitting and standing are SEPARATE — standing is excluded)
LANHAR_4_ACTIVITIES = ["walking", "walking_upstairs", "walking_downstairs", "sitting"]

# CrossHAR 4-activity protocol: still (sit+stand merged), walking, upstairs, downstairs
# CrossHAR merges sitting+standing into "still", making classification easier
CROSSHAR_4_ACTIVITIES = ["walking", "walking_upstairs", "walking_downstairs", "sitting"]
CROSSHAR_STILL_GROUPS = {"sitting", "standing"}  # These get merged to "still"


# =============================================================================
# Model Loading (reused from compare_models.py)
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SemanticAlignmentModel, dict, dict]:
    """Load model from checkpoint with architecture from hyperparameters.json."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')

    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        enc_cfg = hyperparams.get('encoder', {})
        head_cfg = hyperparams.get('semantic_head', {})
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        raise FileNotFoundError(
            f"hyperparameters.json not found at {hyperparams_path}."
        )

    encoder = IMUActivityRecognitionEncoder(
        d_model=enc_cfg.get('d_model', 384),
        num_heads=enc_cfg.get('num_heads', 8),
        num_temporal_layers=enc_cfg.get('num_temporal_layers', 4),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        use_cross_channel=enc_cfg.get('use_cross_channel', True),
        cnn_channels=enc_cfg.get('cnn_channels', [32, 64]),
        cnn_kernel_sizes=enc_cfg.get('cnn_kernel_sizes', [5]),
        target_patch_size=enc_cfg.get('target_patch_size', 64),
        use_channel_encoding=enc_cfg.get('use_channel_encoding', False)
    )

    semantic_head = SemanticAlignmentHead(
        d_model=enc_cfg.get('d_model', 384),
        d_model_fused=384,
        output_dim=384,
        num_temporal_layers=head_cfg.get('num_temporal_layers', 2),
        num_heads=enc_cfg.get('num_heads', 8),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        num_fusion_queries=head_cfg.get('num_fusion_queries', 4),
        use_fusion_self_attention=head_cfg.get('use_fusion_self_attention', True),
        num_pool_queries=head_cfg.get('num_pool_queries', 4),
        use_pool_self_attention=head_cfg.get('use_pool_self_attention', True)
    )

    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=token_cfg.get('num_heads', 4),
        dropout=enc_cfg.get('dropout', 0.1)
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if unexpected_keys:
        other_unexpected = [k for k in unexpected_keys if 'channel_encoding' not in k]
        if other_unexpected:
            print(f"  Warning: Unexpected keys: {other_unexpected[:5]}...")
    if missing_keys:
        print(f"  Warning: Missing keys: {missing_keys[:5]}...")

    model.to(device)
    model.eval()

    model_info = {
        'epoch': epoch,
        'checkpoint_path': str(checkpoint_path)
    }

    return model, model_info, checkpoint


def load_label_bank(checkpoint: dict, device: torch.device, hyperparams_path: Path) -> LearnableLabelBank:
    """Load LearnableLabelBank with trained state from checkpoint."""
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        token_cfg = {}

    label_bank = LearnableLabelBank(
        device=device,
        num_heads=token_cfg.get('num_heads', 4),
        num_queries=token_cfg.get('num_queries', 4),
        dropout=0.1
    )

    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        print("  Loaded trained LearnableLabelBank state")
    else:
        print("  Warning: No label_bank_state_dict in checkpoint")

    label_bank.eval()
    return label_bank


# =============================================================================
# Dataset helpers
# =============================================================================

def get_raw_labels_for_dataset(dataset_name: str) -> List[str]:
    """Get canonical activity labels for a dataset."""
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in DATASET_CONFIGS:
        return sorted(list(DATASET_CONFIGS[dataset_name_lower]['synonyms'].keys()))
    else:
        print(f"Warning: No config found for {dataset_name}")
        return ['unknown']


def create_dataset_loader(dataset_name: str, patch_size: Optional[float] = None) -> DataLoader:
    """Create a DataLoader for a single dataset."""
    ps = patch_size or PATCH_SIZE_PER_DATASET.get(dataset_name, 1.5)
    dataset = IMUPretrainingDataset(
        data_root='data',
        datasets=[dataset_name],
        split='val',
        patch_size_per_dataset={dataset_name: ps},
        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
        seed=42
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=IMUPretrainingDataset.collate_fn
    )


# =============================================================================
# Closed-Set Evaluation (NLS-HAR / GOAT protocol)
# =============================================================================

def compute_closed_set_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
    dataset_labels: List[str]
) -> Dict[str, float]:
    """
    Compute closed-set metrics (NLS-HAR / GOAT protocol).

    Restricts predictions to ONLY the target dataset's labels,
    then computes macro F1 and accuracy.
    """
    model.eval()
    label_bank.eval()

    with torch.no_grad():
        label_embeddings = label_bank.encode(dataset_labels, normalize=True)
        label_embeddings = label_embeddings.to(device)

    all_gt_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Closed-set", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                attention_mask=attention_mask)

            similarity = imu_emb @ label_embeddings.T
            pred_indices = similarity.argmax(dim=1).cpu().numpy()
            pred_labels = [dataset_labels[i] for i in pred_indices]

            all_gt_labels.extend(label_texts)
            all_pred_labels.extend(pred_labels)

    return {
        'f1_macro': f1_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0) * 100,
        'accuracy': accuracy_score(all_gt_labels, all_pred_labels) * 100,
        'n_samples': len(all_gt_labels),
        'n_classes': len(dataset_labels),
    }


# =============================================================================
# 4-Activity Subset Evaluation (LanHAR / CrossHAR protocol)
# =============================================================================

def compute_shared_activity_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
    shared_activities: List[str] = None,
) -> Dict[str, float]:
    """
    Compute metrics on only the shared activity subset.

    Filters samples to shared activities (by group mapping), then does
    closed-set evaluation using only those label embeddings.

    Used for LanHAR / CrossHAR comparison (4-activity protocol).

    Returns:
        Dict with f1_macro, accuracy, per_class metrics, n_samples, n_classes
    """
    if shared_activities is None:
        shared_activities = LANHAR_4_ACTIVITIES

    model.eval()
    label_bank.eval()

    # Build group mapping for filtering
    shared_groups = set()
    for activity in shared_activities:
        group = get_group_for_label(activity)
        shared_groups.add(group)

    # Encode only the shared activity labels
    with torch.no_grad():
        label_embeddings = label_bank.encode(shared_activities, normalize=True)
        label_embeddings = label_embeddings.to(device)

    all_gt_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  4-activity", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            # Filter to only shared activities (by group)
            keep_indices = []
            mapped_gt = []
            for i, label in enumerate(label_texts):
                group = get_group_for_label(label)
                if group in shared_groups:
                    keep_indices.append(i)
                    # Map to the canonical shared activity name
                    for sa in shared_activities:
                        if get_group_for_label(sa) == group:
                            mapped_gt.append(sa)
                            break

            if not keep_indices:
                continue

            # Select only matching samples
            keep_indices_t = torch.tensor(keep_indices, device=device)
            data_filtered = data[keep_indices_t]
            channel_mask_filtered = channel_mask[keep_indices_t]
            attention_mask_filtered = attention_mask[keep_indices_t]

            # Rebuild metadata for filtered samples
            sr_filtered = [sampling_rates[i] for i in keep_indices]
            ps_filtered = [patch_sizes[i] for i in keep_indices]
            cd_filtered = [channel_descriptions[i] for i in keep_indices]

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model(data_filtered, cd_filtered, channel_mask_filtered, sr_filtered, ps_filtered,
                                attention_mask=attention_mask_filtered)

            similarity = imu_emb @ label_embeddings.T
            pred_indices = similarity.argmax(dim=1).cpu().numpy()
            pred_labels = [shared_activities[i] for i in pred_indices]

            all_gt_labels.extend(mapped_gt)
            all_pred_labels.extend(pred_labels)

    if not all_gt_labels:
        return {
            'f1_macro': 0.0,
            'accuracy': 0.0,
            'n_samples': 0,
            'n_classes': len(shared_activities),
        }

    # Per-class breakdown
    per_class = {}
    for activity in shared_activities:
        indices = [i for i, gt in enumerate(all_gt_labels) if gt == activity]
        if indices:
            correct = sum(1 for i in indices if all_pred_labels[i] == activity)
            per_class[activity] = {
                'n_samples': len(indices),
                'accuracy': correct / len(indices) * 100,
            }

    return {
        'f1_macro': f1_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0) * 100,
        'accuracy': accuracy_score(all_gt_labels, all_pred_labels) * 100,
        'n_samples': len(all_gt_labels),
        'n_classes': len(shared_activities),
        'per_class': per_class,
    }


def compute_crosshar_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute metrics matching CrossHAR protocol.

    CrossHAR merges sitting+standing into "still", giving 4 classes:
    still, walking, upstairs, downstairs.

    We predict over our 5 candidate labels (walking, walking_upstairs,
    walking_downstairs, sitting, standing), then merge sitting/standing
    predictions into "still" for metric computation.
    """
    model.eval()
    label_bank.eval()

    # Candidate labels: the 4 LanHAR activities + standing
    candidate_labels = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing"]

    # Groups that map to "still"
    still_groups = {get_group_for_label("sitting"), get_group_for_label("standing")}
    target_groups = still_groups | {get_group_for_label("walking"),
                                     get_group_for_label("walking_upstairs"),
                                     get_group_for_label("walking_downstairs")}

    with torch.no_grad():
        label_embeddings = label_bank.encode(candidate_labels, normalize=True)
        label_embeddings = label_embeddings.to(device)

    all_gt = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  CrossHAR", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            # Filter to target activities
            keep_indices = []
            mapped_gt = []
            for i, label in enumerate(label_texts):
                group = get_group_for_label(label)
                if group in target_groups:
                    keep_indices.append(i)
                    # Map ground truth
                    if group in still_groups:
                        mapped_gt.append("still")
                    elif group == get_group_for_label("walking"):
                        mapped_gt.append("walking")
                    elif group == get_group_for_label("walking_upstairs"):
                        mapped_gt.append("upstairs")
                    elif group == get_group_for_label("walking_downstairs"):
                        mapped_gt.append("downstairs")

            if not keep_indices:
                continue

            keep_indices_t = torch.tensor(keep_indices, device=device)
            data_filtered = data[keep_indices_t]
            channel_mask_filtered = channel_mask[keep_indices_t]
            attention_mask_filtered = attention_mask[keep_indices_t]
            sr_filtered = [sampling_rates[i] for i in keep_indices]
            ps_filtered = [patch_sizes[i] for i in keep_indices]
            cd_filtered = [channel_descriptions[i] for i in keep_indices]

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model(data_filtered, cd_filtered, channel_mask_filtered, sr_filtered, ps_filtered,
                                attention_mask=attention_mask_filtered)

            similarity = imu_emb @ label_embeddings.T
            pred_indices = similarity.argmax(dim=1).cpu().numpy()

            # Map predictions to CrossHAR classes
            for idx in pred_indices:
                pred_label = candidate_labels[idx]
                if pred_label in ("sitting", "standing"):
                    all_pred.append("still")
                elif pred_label == "walking":
                    all_pred.append("walking")
                elif pred_label == "walking_upstairs":
                    all_pred.append("upstairs")
                elif pred_label == "walking_downstairs":
                    all_pred.append("downstairs")

            all_gt.extend(mapped_gt)

    if not all_gt:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': 4}

    crosshar_classes = ["still", "walking", "upstairs", "downstairs"]
    per_class = {}
    for cls in crosshar_classes:
        indices = [i for i, gt in enumerate(all_gt) if gt == cls]
        if indices:
            correct = sum(1 for i in indices if all_pred[i] == cls)
            per_class[cls] = {'n_samples': len(indices), 'accuracy': correct / len(indices) * 100}

    return {
        'accuracy': accuracy_score(all_gt, all_pred) * 100,
        'f1_macro': f1_score(all_gt, all_pred, average='macro', zero_division=0) * 100,
        'n_samples': len(all_gt),
        'n_classes': 4,
        'per_class': per_class,
    }


# =============================================================================
# Output Formatting
# =============================================================================

def print_comparison_table(results: Dict):
    """Print paper-ready baseline comparison tables."""
    print()
    print("=" * 70)
    print("BASELINE COMPARISON TABLE")
    print("=" * 70)

    # --- NLS-HAR ---
    print()
    print("--- NLS-HAR (Closed-Set Macro F1) [zero-shot, DistilBERT] ---")
    print(f"{'Dataset':<16}{'NLS-HAR':>10}{'Ours':>10}{'Delta':>12}")
    baseline = BASELINES['nls_har']
    for ds_name, bl_values in baseline['datasets'].items():
        bl_f1 = bl_values['f1']
        our_f1 = results.get(f'nls_har_{ds_name}_f1', 0.0)
        delta = our_f1 - bl_f1
        sign = '+' if delta >= 0 else ''
        print(f"{ds_name:<16}{bl_f1:>9.2f}%{our_f1:>9.2f}%{sign}{delta:>10.2f}pp")

    # --- GOAT ---
    print()
    print("--- GOAT (Closed-Set Macro F1) [few-shot: fine-tunes on ~7% target] ---")
    print(f"{'Dataset':<16}{'GOAT':>10}{'Ours':>10}{'Delta':>12}")
    baseline = BASELINES['goat']
    for ds_name, bl_values in baseline['datasets'].items():
        bl_f1 = bl_values['f1']
        our_f1 = results.get(f'goat_{ds_name}_f1', 0.0)
        delta = our_f1 - bl_f1
        sign = '+' if delta >= 0 else ''
        print(f"{ds_name:<16}{bl_f1:>9.2f}%{our_f1:>9.2f}%{sign}{delta:>10.2f}pp")

    # --- LanHAR ---
    print()
    print("--- LanHAR (4-Activity F1) [single-source transfer, BERT] ---")
    print(f"{'Dataset':<16}{'LanHAR':>10}{'Ours':>10}{'Delta':>12}")
    baseline = BASELINES['lanhar']
    for ds_name, bl_values in baseline['datasets'].items():
        bl_f1 = bl_values['f1']
        our_f1 = results.get(f'lanhar_{ds_name}_f1', 0.0)
        delta = our_f1 - bl_f1
        sign = '+' if delta >= 0 else ''
        print(f"{ds_name:<16}{bl_f1:>9.2f}%{our_f1:>9.2f}%{sign}{delta:>10.2f}pp")

    # --- CrossHAR ---
    print()
    print("--- CrossHAR (4-Class Accuracy, sit+stand='still') [multi-source supervised] ---")
    print(f"{'Dataset':<16}{'CrossHAR':>10}{'Ours':>10}{'Delta':>12}")
    baseline = BASELINES['crosshar']
    for ds_name, bl_values in baseline['datasets'].items():
        bl_acc = bl_values['accuracy']
        our_acc = results.get(f'crosshar_{ds_name}_acc', 0.0)
        delta = our_acc - bl_acc
        sign = '+' if delta >= 0 else ''
        print(f"{ds_name:<16}{bl_acc:>9.2f}%{our_acc:>9.2f}%{sign}{delta:>10.2f}pp")

    print("=" * 70)

    # Protocol notes
    print()
    print("Protocol Notes:")
    print("  NLS-HAR:  Most direct comparison. Both use zero-shot text-aligned embeddings.")
    print("            NLS-HAR uses DistilBERT; we use SentenceBERT + learnable pooling.")
    print("  GOAT:     NOT zero-shot. GOAT fine-tunes on ~7% of target dataset labels.")
    print("            Results from GOAT-CLIP (best per-dataset variant).")
    print("            Opportunity/Daphnet FoG excluded (incompatible label sets).")
    print("  LanHAR:   Single-source transfer (trains on 1 dataset, tests on another).")
    print("            Uses GPT-4-generated activity descriptions with fine-tuned BERT.")
    print("            Numbers are best single-source pairs (UCI->target).")
    print("  CrossHAR: Supervised multi-source (trains on 3 labeled datasets).")
    print("            No text encoder (sensor-only). Merges sitting+standing='still'.")
    print("            Numbers from HAR-DoReMi reproduction (not original paper).")


# =============================================================================
# Main
# =============================================================================

def run_benchmark():
    """Run benchmark against all published baselines."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print()
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    model, model_info, checkpoint = load_model(CHECKPOINT_PATH, device)
    hyperparams_path = Path(CHECKPOINT_PATH).parent / 'hyperparameters.json'
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    print(f"  Loaded model from epoch {model_info['epoch']}")

    results = {}
    detailed_results = {}

    # =========================================================================
    # Mode 1: NLS-HAR Comparison (Closed-Set Macro F1)
    # =========================================================================
    print()
    print("=" * 60)
    print("MODE 1: NLS-HAR Comparison (MotionSense, MobiAct)")
    print("=" * 60)

    for ds_name in BASELINES['nls_har']['datasets']:
        print(f"\n  Evaluating {ds_name}...")
        dataset_labels = get_raw_labels_for_dataset(ds_name)
        print(f"    Labels ({len(dataset_labels)}): {dataset_labels}")

        loader = create_dataset_loader(ds_name)
        metrics = compute_closed_set_metrics(model, label_bank, loader, device, dataset_labels)

        results[f'nls_har_{ds_name}_f1'] = metrics['f1_macro']
        results[f'nls_har_{ds_name}_acc'] = metrics['accuracy']
        detailed_results[f'nls_har_{ds_name}'] = metrics
        print(f"    F1 (macro): {metrics['f1_macro']:.2f}%, Accuracy: {metrics['accuracy']:.2f}% ({metrics['n_samples']} samples)")

    # =========================================================================
    # Mode 2: GOAT Comparison (Closed-Set Macro F1)
    # =========================================================================
    print()
    print("=" * 60)
    print("MODE 2: GOAT Comparison (RealWorld, Realdisp, Opportunity, Daphnet FoG)")
    print("=" * 60)

    for ds_name in BASELINES['goat']['datasets']:
        print(f"\n  Evaluating {ds_name}...")
        dataset_labels = get_raw_labels_for_dataset(ds_name)
        print(f"    Labels ({len(dataset_labels)}): {dataset_labels}")

        loader = create_dataset_loader(ds_name)
        metrics = compute_closed_set_metrics(model, label_bank, loader, device, dataset_labels)

        results[f'goat_{ds_name}_f1'] = metrics['f1_macro']
        results[f'goat_{ds_name}_acc'] = metrics['accuracy']
        detailed_results[f'goat_{ds_name}'] = metrics
        print(f"    F1 (macro): {metrics['f1_macro']:.2f}%, Accuracy: {metrics['accuracy']:.2f}% ({metrics['n_samples']} samples)")

    # =========================================================================
    # Mode 3: LanHAR Comparison (4-Activity F1, no sit+stand merge)
    # =========================================================================
    print()
    print("=" * 60)
    print("MODE 3: LanHAR Comparison (4-Activity Subset)")
    print(f"  Activities: {LANHAR_4_ACTIVITIES}")
    print("=" * 60)

    for ds_name in BASELINES['lanhar']['datasets']:
        print(f"\n  Evaluating {ds_name} (LanHAR protocol)...")

        loader = create_dataset_loader(ds_name)
        metrics = compute_shared_activity_metrics(model, label_bank, loader, device, LANHAR_4_ACTIVITIES)

        results[f'lanhar_{ds_name}_f1'] = metrics['f1_macro']
        detailed_results[f'lanhar_{ds_name}'] = metrics

        print(f"    F1 (macro): {metrics['f1_macro']:.2f}%, Accuracy: {metrics['accuracy']:.2f}% ({metrics['n_samples']} samples)")
        if 'per_class' in metrics:
            for activity, cls_metrics in metrics['per_class'].items():
                print(f"      {activity}: {cls_metrics['accuracy']:.1f}% ({cls_metrics['n_samples']} samples)")

    # =========================================================================
    # Mode 4: CrossHAR Comparison (4-Class Accuracy, sit+stand merged to "still")
    # =========================================================================
    print()
    print("=" * 60)
    print("MODE 4: CrossHAR Comparison (sit+stand='still')")
    print("  Classes: still (sit+stand), walking, upstairs, downstairs")
    print("=" * 60)

    for ds_name in BASELINES['crosshar']['datasets']:
        print(f"\n  Evaluating {ds_name} (CrossHAR protocol)...")

        loader = create_dataset_loader(ds_name)
        metrics = compute_crosshar_metrics(model, label_bank, loader, device)

        results[f'crosshar_{ds_name}_acc'] = metrics['accuracy']
        detailed_results[f'crosshar_{ds_name}'] = metrics

        print(f"    Accuracy: {metrics['accuracy']:.2f}%, F1 (macro): {metrics['f1_macro']:.2f}% ({metrics['n_samples']} samples)")
        if 'per_class' in metrics:
            for cls, cls_metrics in metrics['per_class'].items():
                print(f"      {cls}: {cls_metrics['accuracy']:.1f}% ({cls_metrics['n_samples']} samples)")

    # =========================================================================
    # Print comparison tables
    # =========================================================================
    print_comparison_table(results)

    # =========================================================================
    # Save results
    # =========================================================================
    save_data = {
        'checkpoint': CHECKPOINT_PATH,
        'epoch': model_info['epoch'],
        'summary': results,
        'detailed': {},
        'baselines': BASELINES,
    }
    # Convert detailed results to JSON-safe format
    for key, val in detailed_results.items():
        save_data['detailed'][key] = {
            k: v for k, v in val.items()
            if isinstance(v, (int, float, str, dict, list))
        }

    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    run_benchmark()
