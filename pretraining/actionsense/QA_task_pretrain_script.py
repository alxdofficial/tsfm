"""Joint pretraining script for ActionSense QA with TSFM encoder + LLaMA."""
import math
import os
import re
import string
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.ActionSenseQADataset import ActionSenseQADataset, actionsenseqa_collate
from encoder.TSFMEncoder import TSFMEncoder
from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor
from pretraining.actionsense.heads import SensorQALLMHead
from training_utils import (
    configure_device_and_amp,
    build_warmup_cosine_scheduler,
    sanity_check_optimizer,
    count_params,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEBUG_ROOT = os.path.join("debug", "pretraining", "actionsense_qa")
LOSS_PLOT_PATH = os.path.join(DEBUG_ROOT, "train_val_loss.png")
STEP_LOSS_PLOT_PATH = os.path.join(DEBUG_ROOT, "train_batch_loss.png")
EM_PLOT_PATH = os.path.join(DEBUG_ROOT, "train_val_exact_match.png")
F1_PLOT_PATH = os.path.join(DEBUG_ROOT, "train_val_f1.png")
BLEU_ROUGE_PLOT_PATH = os.path.join(DEBUG_ROOT, "train_val_bleu_rouge.png")
GRAD_PLOT_PATH = os.path.join(DEBUG_ROOT, "encoder_grad_norms.png")
GEN_PLOT_DIR = os.path.join(DEBUG_ROOT, "generations")
CHECKPOINT_DIR = os.path.join("checkpoints", "actionsense_qa")


class Config:
    # Data
    patch_size = 1000
    context_size = -1  # <=0 means keep the full sequence (no truncation)
    qa_base_dir = "data/actionsenseqa/data"
    qa_csv = "data/actionsenseqa/data/qa_pairs.csv"
    manifest_csv = "data/actionsenseqa/data/manifest.csv"
    val_ratio = 0.2
    split_seed = 42

    # Training
    epochs = 100
    batch_size = 16
    num_workers = 8
    lr = 1e-4
    weight_decay = 0.05
    grad_clip = 1.0
    loss_plot_every = 10

    # Models
    llama_model_name = "meta-llama/Llama-3.2-1B"
    encoder_feature_dim = 1024
    encoding_dim = 1024
    llama_dropout = 0.1
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.05
    use_lora = True
    log_mode = "info"  # options: "debug", "info", "silent"
    generation_every = 30

    # Misc
    checkpoint_every = 5


CFG = Config()


def _log_allowed(level: str) -> bool:
    mode = getattr(CFG, "log_mode", "info")
    if level == "error":
        return True
    if level == "warn":
        return mode != "silent"
    if level == "info":
        return mode in {"info", "debug"}
    if level == "debug":
        return mode == "debug"
    return True


def log_message(message: str, level: str = "info") -> None:
    if _log_allowed(level):
        print(message)


def log_debug(message: str) -> None:
    log_message(message, level="debug")


def log_info(message: str) -> None:
    log_message(message, level="info")


def log_warn(message: str) -> None:
    log_message(message, level="warn")


_ARTICLES = {"a", "an", "the"}
_EPS = 1e-8


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b({})\b".format("|".join(_ARTICLES)), " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ensure_tokens(text_or_tokens) -> List[str]:
    if isinstance(text_or_tokens, str):
        normalized = _normalize_answer(text_or_tokens)
        return normalized.split() if normalized else []
    return [str(tok) for tok in text_or_tokens if str(tok)]


def _f1_score(prediction, ground_truth) -> float:
    pred_tokens = _ensure_tokens(prediction)
    truth_tokens = _ensure_tokens(ground_truth)
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu4_score(prediction, ground_truth, max_n: int = 4) -> float:
    pred_tokens = _ensure_tokens(prediction)
    truth_tokens = _ensure_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    max_order = min(max_n, len(pred_tokens), len(truth_tokens))
    if max_order == 0:
        return 0.0
    log_precision_sum = 0.0
    for n in range(1, max_order + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(truth_tokens[i : i + n]) for i in range(len(truth_tokens) - n + 1)
        )
        overlap = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
        possible = len(pred_tokens) - n + 1
        if possible <= 0:
            continue
        precision = overlap / possible if possible > 0 else 0.0
        precision = max(precision, _EPS)
        log_precision_sum += math.log(precision)
    log_precision = log_precision_sum / max_order
    ref_len = len(truth_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        return 0.0
    bp = 1.0
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(log_precision)


def _rouge_l_score(prediction, ground_truth) -> float:
    pred_tokens = _ensure_tokens(prediction)
    truth_tokens = _ensure_tokens(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j + 1]
            if pred_tokens[i] == truth_tokens[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp
    lcs = dp[n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2 * precision * recall / denom


def _init_metric_state() -> Dict[str, float]:
    return {
        "token_correct": 0.0,
        "token_total": 0.0,
        "em_sum": 0.0,
        "f1_sum": 0.0,
        "bleu_sum": 0.0,
        "rougeL_sum": 0.0,
        "sample_count": 0.0,
    }


def _update_metrics(
    state: Dict[str, float],
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
) -> None:
    batch_size = preds.size(0)
    mask = mask.to(dtype=torch.bool)
    for idx in range(batch_size):
        mask_i = mask[idx]
        if not mask_i.any():
            continue
        preds_i = preds[idx][mask_i]
        labels_i = labels[idx][mask_i]
        correct = (preds_i == labels_i).sum().item()
        total = mask_i.sum().item()
        state["token_correct"] += float(correct)
        state["token_total"] += float(total)
        pred_ids = preds_i.detach().cpu().tolist()
        label_ids = labels_i.detach().cpu().tolist()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        norm_pred = _normalize_answer(pred_text)
        norm_label = _normalize_answer(label_text)
        pred_tokens = norm_pred.split() if norm_pred else []
        label_tokens = norm_label.split() if norm_label else []
        state["sample_count"] += 1.0
        state["em_sum"] += 1.0 if pred_tokens == label_tokens else 0.0
        state["f1_sum"] += _f1_score(pred_tokens, label_tokens)
        state["bleu_sum"] += _bleu4_score(pred_tokens, label_tokens)
        state["rougeL_sum"] += _rouge_l_score(pred_tokens, label_tokens)


def _summarize_metrics(state: Dict[str, float]) -> Dict[str, float]:
    sample_count = max(state["sample_count"], 1.0)
    token_total = max(state["token_total"], 1.0)
    return {
        "token_accuracy": state["token_correct"] / token_total,
        "exact_match": state["em_sum"] / sample_count,
        "f1": state["f1_sum"] / sample_count,
        "bleu4": state["bleu_sum"] / sample_count,
        "rougeL": state["rougeL_sum"] / sample_count,
    }


def build_processors() -> List:
    return [
        CorrelationSummaryProcessor(),
        FrequencyFeatureProcessor(),
        HistogramFeatureProcessor(),
        StatisticalFeatureProcessor(),
    ]



def build_dataloaders(device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ActionSenseQADataset(
        base_dir=CFG.qa_base_dir,
        qa_csv_path=CFG.qa_csv,
        manifest_csv_path=CFG.manifest_csv,
        split="train",
        val_ratio=CFG.val_ratio,
        split_seed=CFG.split_seed,
        patch_size=CFG.patch_size,
        log_mode=CFG.log_mode,
    )
    val_dataset = ActionSenseQADataset(
        base_dir=CFG.qa_base_dir,
        qa_csv_path=CFG.qa_csv,
        manifest_csv_path=CFG.manifest_csv,
        split="val",
        val_ratio=CFG.val_ratio,
        split_seed=CFG.split_seed,
        patch_size=CFG.patch_size,
        log_mode=CFG.log_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=actionsenseqa_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=actionsenseqa_collate,
    )
    return train_loader, val_loader


class SensorQAModel(nn.Module):
    def __init__(self, encoder: TSFMEncoder, qa_head: SensorQALLMHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.qa_head = qa_head
        self._debug_logged = False

    def forward(
        self,
        patches: torch.Tensor,
        pad_mask: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = {"patches": patches, "pad_mask": pad_mask}
        if not self._debug_logged:
            pad_shape = None if pad_mask is None else pad_mask.shape
            log_debug(
                f"[DEBUG] SensorQAModel input patches shape={patches.shape} dtype={patches.dtype} pad_mask shape={pad_shape}"
            )
            metadata = batch.get("metadata")
            if metadata is not None:
                subjects = metadata.get("subject", [])
                activities = metadata.get("activity_name", [])
                sensors = metadata.get("sensor_path", [])
                for idx, (q, a) in enumerate(zip(questions, answers)):
                    subj = subjects[idx] if idx < len(subjects) else None
                    act = activities[idx] if idx < len(activities) else None
                    sensor_path = sensors[idx] if idx < len(sensors) else None
                    log_debug(
                        f"[DEBUG] Sample {idx}: subject={subj}, activity={act}, sensor_path={sensor_path}"
                    )
                    log_debug(f"        Q: {q}")
                    log_debug(f"        A: {a}")
        encoded = self.encoder.encode_batch(batch)
        tokens = encoded["tokens"]
        if not self._debug_logged:
            log_debug(
                f"[DEBUG] Encoder tokens shape={tokens.shape} dtype={tokens.dtype}"
            )
            self._debug_logged = True
        loss, info = self.qa_head(tokens, pad_mask, questions, answers)
        info["tokens"] = tokens.detach()
        info["pad_mask"] = pad_mask
        if "small_features" in encoded:
            info["small_features"] = encoded["small_features"].detach()
        info["per_channel_features"] = encoded.get("features", torch.empty(0)).detach()
        return loss, info


def build_models(processors: List, device: torch.device) -> Tuple[SensorQAModel, SensorQALLMHead]:
    encoder = TSFMEncoder(
        processors=processors,
        feature_dim=CFG.encoder_feature_dim,
        encoding_dim=CFG.encoding_dim,
        pretraining_args={},
        recon_head=None,
    ).to(device)

    qa_head = SensorQALLMHead(
        llama_model_name=CFG.llama_model_name,
        feature_dim=CFG.encoder_feature_dim,
        attn_heads=8,
        attn_dropout=CFG.llama_dropout,
        lora_rank=CFG.lora_rank,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        use_lora=CFG.use_lora,
        log_mode=CFG.log_mode,
    ).to(device)

    model = SensorQAModel(encoder, qa_head).to(device)

    log_info(
        f"[INIT] Encoder params: {count_params(encoder):.2f}M | "
        f"LLM params: {count_params(qa_head.llama):.2f}M"
    )

    return model, qa_head


def plot_loss_curves(train_epochs: List[float], val_epochs: List[float]) -> None:
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    plt.figure(figsize=(8, 4))
    epochs = list(range(1, len(train_epochs) + 1))
    plt.plot(epochs, train_epochs, label="Train Loss", marker="o")
    plt.plot(epochs, val_epochs, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ActionSense QA Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()


def plot_batch_loss(loss_history: List[float]) -> None:
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Batch Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(STEP_LOSS_PLOT_PATH)
    plt.close()


def plot_metric_curve(
    train_values: List[float],
    val_values: List[float],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    if not train_values or not val_values:
        return
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    plt.figure(figsize=(8, 4))
    epochs = list(range(1, len(train_values) + 1))
    plt.plot(epochs, train_values, label="Train", marker="o")
    plt.plot(epochs, val_values, label="Val", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bleu_rouge_curve(
    train_bleu: List[float],
    val_bleu: List[float],
    train_rouge: List[float],
    val_rouge: List[float],
    out_path: str,
) -> None:
    if not train_bleu or not val_bleu or not train_rouge or not val_rouge:
        return
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    plt.figure(figsize=(8, 4))
    epochs = list(range(1, len(train_bleu) + 1))
    plt.plot(epochs, train_bleu, label="Train BLEU-4", marker="o")
    plt.plot(epochs, val_bleu, label="Val BLEU-4", marker="o")
    plt.plot(epochs, train_rouge, label="Train ROUGE-L", marker="s")
    plt.plot(epochs, val_rouge, label="Val ROUGE-L", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("ActionSense QA BLEU-4 & ROUGE-L")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grad_history(grad_history: Dict[str, List[float]]) -> None:
    if not grad_history:
        return
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    keys = list(grad_history.keys())
    cols = 2
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    x = None
    for idx, key in enumerate(keys):
        ax = axes[idx]
        values = grad_history[key]
        steps = list(range(1, len(values) + 1))
        x = steps
        ax.plot(steps, values, label=key)
        ax.set_title(key)
        ax.set_ylabel("||grad||")
    for ax in axes[len(keys):]:
        ax.axis("off")
    if x is not None:
        for ax in axes[:len(keys)]:
            ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(GRAD_PLOT_PATH)
    plt.close(fig)


def save_generation_visual(
    epoch: int,
    step: int,
    sample_idx: int,
    question: str,
    pred_answer: str,
    gt_answer: str,
    patches_sample: torch.Tensor,
    small_features_sample: torch.Tensor,
    tokens_sample: torch.Tensor,
    fused_sample: torch.Tensor,
    pad_mask_sample: torch.Tensor,
    timestamps_sample: Optional[torch.Tensor],
) -> None:
    if getattr(CFG, "log_mode", "info") == "silent":
        return
    os.makedirs(GEN_PLOT_DIR, exist_ok=True)
    patches_np = patches_sample.detach().cpu().numpy() if patches_sample.numel() else None
    small_np = small_features_sample.detach().cpu().numpy() if small_features_sample.numel() else None
    tokens_np = tokens_sample.detach().cpu().numpy() if tokens_sample.numel() else None
    fused_np = fused_sample.detach().cpu().numpy() if fused_sample.numel() else None
    pad_mask_np = pad_mask_sample.detach().cpu().numpy() if pad_mask_sample.numel() else None
    timestamps_np: Optional[np.ndarray] = None
    if timestamps_sample is not None:
        if isinstance(timestamps_sample, torch.Tensor):
            if timestamps_sample.numel():
                timestamps_np = timestamps_sample.detach().cpu().numpy()
        else:
            try:
                timestamps_np = np.asarray(timestamps_sample, dtype=np.float64)
            except Exception:
                timestamps_np = None
        if timestamps_np is not None and timestamps_np.size == 0:
            timestamps_np = None

    if patches_np is None or patches_np.size == 0:
        return

    num_patches = patches_np.shape[0]
    if pad_mask_np is None or pad_mask_np.size == 0:
        pad_mask_np = np.ones(num_patches, dtype=bool)
    pad_mask_np = pad_mask_np.astype(bool)
    valid_indices = (
        np.where(pad_mask_np)[0] if pad_mask_np.size else np.arange(num_patches)
    )
    if valid_indices.size == 0:
        valid_indices = np.arange(num_patches)
    last_patch_idx = int(valid_indices[-1])

    # Estimate activity and patch durations in seconds, if timestamps are available
    activity_duration_s: Optional[float] = None
    patch_duration_s: Optional[float] = None
    if timestamps_np is not None and timestamps_np.size > 1:
        patch_length = patches_np.shape[1] if patches_np.ndim >= 2 else 0
        valid_patch_count = int(pad_mask_np.sum()) if pad_mask_np.size else num_patches
        if patch_length and valid_patch_count:
            total_valid_steps = valid_patch_count * patch_length
            idx_start = max(timestamps_np.size - total_valid_steps, 0)
            ts_valid = timestamps_np[idx_start:]
            if ts_valid.size > 1:
                activity_duration_s = float(ts_valid[-1] - ts_valid[0])
                diffs = np.diff(ts_valid)
                positive_diffs = diffs[diffs > 0]
                if positive_diffs.size:
                    median_dt = float(np.median(positive_diffs))
                    patch_duration_s = median_dt * patch_length
                elif activity_duration_s > 0 and total_valid_steps > 1:
                    avg_dt = activity_duration_s / (total_valid_steps - 1)
                    patch_duration_s = avg_dt * patch_length

    # Flatten all patches over time for channel line plot
    sensor_sequence = patches_np.reshape(-1, patches_np.shape[2])  # (P*T, D)
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    axes = axes.flatten()

    # Line plot for entire sequence (first 10 channels)
    axes[0].plot(sensor_sequence[:, : min(10, sensor_sequence.shape[1])])
    axes[0].set_title(
        f"Sensor signals (channels 1-{min(10, sensor_sequence.shape[1])})"
    )
    axes[0].set_xlabel("Time index (patch concatenated)")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Value")

    # Pad mask visualization
    axes[1].imshow(pad_mask_np[:, np.newaxis], aspect="auto", cmap="Greys")
    axes[1].set_title("Pad mask (black = valid patch)")
    axes[1].set_ylabel("Patch index")
    axes[1].set_xticks([])
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    if small_np is not None and small_np.size > 0:
        small_last = small_np[last_patch_idx]  # (D,K)
        axes[2].imshow(small_last, aspect="auto", cmap="plasma")
        axes[2].set_title(
            f"Small features (patch {last_patch_idx+1}/{num_patches})"
        )
        axes[2].set_xlabel("Feature index")
        axes[2].set_ylabel("Channel")
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[2].axis("off")

    if tokens_np is not None and tokens_np.size > 0:
        tokens_last = tokens_np[last_patch_idx]  # (D,F)
        axes[3].imshow(tokens_last, aspect="auto", cmap="magma")
        axes[3].set_title(
            f"Encoder tokens (patch {last_patch_idx+1}/{num_patches})"
        )
        axes[3].set_xlabel("Feature index")
        axes[3].set_ylabel("Channel")
        axes[3].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[3].axis("off")

    fused_stats: Optional[np.ndarray] = None
    fused_diff_stats: Optional[np.ndarray] = None
    if fused_np is not None and fused_np.size > 0:
        fused_stats = np.std(fused_np, axis=1)
        # Use percentile-based limits to emphasise variation without clipping outliers too hard
        vmin, vmax = np.percentile(fused_np, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            centered = fused_np - np.nanmean(fused_np)
            scale = np.nanstd(centered) or 1.0
            vmin, vmax = centered.min(), centered.max()
            if vmin == vmax:
                vmin, vmax = -scale, scale
            fused_plot = centered
        else:
            fused_plot = fused_np
        axes[4].imshow(fused_plot, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[4].set_title(f"Fused embeddings (all {num_patches} patches)")
        axes[4].set_xlabel("Feature index")
        axes[4].set_ylabel("Patch index")
        axes[4].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[4].yaxis.set_major_locator(MaxNLocator(integer=True))
        valid_fused = fused_np[pad_mask_np] if pad_mask_np.any() else fused_np
        if valid_fused.shape[0] > 1:
            diffs = np.diff(valid_fused, axis=0)
            fused_diff_stats = np.linalg.norm(diffs, axis=1)
    else:
        axes[4].axis("off")

    axes[5].axis("off")
    duration_lines: List[str] = []
    if activity_duration_s is not None:
        duration_lines.append(f"Activity duration: {activity_duration_s:.2f}s")
    if patch_duration_s is not None:
        duration_lines.append(f"Patch span: {patch_duration_s:.2f}s")

    text = (
        f"Question:\n{question}\n\nPredicted:\n{pred_answer}\n\nGround Truth:\n{gt_answer}\n\n"
        f"Total patches: {num_patches}, valid: {valid_indices.size}, last plotted patch index: {last_patch_idx}"
    )
    if duration_lines:
        text += "\n" + "\n".join(duration_lines)
    if fused_stats is not None and fused_stats.size:
        fused_stats = fused_stats[pad_mask_np] if pad_mask_np.any() else fused_stats
        text += (
            "\n" +
            f"Fused per-patch std (LayerNorm) -> min {fused_stats.min():.4f}, mean {fused_stats.mean():.4f}, max {fused_stats.max():.4f}"
        )
    if fused_diff_stats is not None and fused_diff_stats.size:
        text += (
            "\n" +
            f"Consecutive patch L2 diff -> min {fused_diff_stats.min():.4f}, mean {fused_diff_stats.mean():.4f}, max {fused_diff_stats.max():.4f}"
        )
    axes[5].text(0.02, 0.5, text, fontsize=11, ha="left", va="center", wrap=True)

    fig.suptitle(f"Epoch {epoch} Step {step} Sample {sample_idx}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(GEN_PLOT_DIR, f"gen_e{epoch:03d}_s{step:06d}_i{sample_idx}.png")
    fig.savefig(out_path)
    plt.close(fig)
    log_info(f"[INFO] Saved generation visualization -> {out_path}")


def save_joint_checkpoint(encoder: TSFMEncoder, qa_head: SensorQALLMHead, epoch: int) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    encoder_path = os.path.join(CHECKPOINT_DIR, f"encoder_e{epoch}.pt")
    torch.save(encoder.state_dict(), encoder_path)
    qa_head.save_checkpoint(CHECKPOINT_DIR, epoch)


def train():
    device, amp_ctx, scaler = configure_device_and_amp()
    processors = build_processors()
    train_loader, val_loader = build_dataloaders(device)
    model, qa_head = build_models(processors, device)

    param_groups = [
        {"params": model.encoder.parameters(), "lr": CFG.lr},
        {"params": model.qa_head.parameters(), "lr": CFG.lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=CFG.weight_decay, betas=(0.9, 0.95))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=CFG.epochs,
        steps_per_epoch=max(1, len(train_loader)),
    )
    sanity_check_optimizer(model.named_parameters(), optimizer)

    tokenizer = model.qa_head.tokenizer

    model.train()
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []
    train_epoch_em: List[float] = []
    val_epoch_em: List[float] = []
    train_epoch_f1: List[float] = []
    val_epoch_f1: List[float] = []
    train_epoch_bleu: List[float] = []
    val_epoch_bleu: List[float] = []
    train_epoch_rouge: List[float] = []
    val_epoch_rouge: List[float] = []
    train_batch_history: List[float] = []
    debug_batch_meta_logged = False
    encoder_grad_logged = False

    grad_groups = model.encoder.grad_groups()
    if getattr(model.encoder, "recon_head", None) is None:
        grad_groups.pop("recon_head", None)
    grad_history = {k: [] for k in grad_groups}
    def record_encoder_grad_norms() -> Dict[str, float]:
        step_norms: Dict[str, float] = {k: float("nan") for k in grad_groups}
        for name, param in model.encoder.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if torch.isnan(grad).any():
                for key in grad_groups:
                    if any(name.startswith(pref) for pref in grad_groups[key]):
                        step_norms[key] = float("nan")
                continue
            grad_sq = grad.float().pow(2).sum().item()
            for key, prefixes in grad_groups.items():
                if any(name.startswith(pref) for pref in prefixes):
                    if math.isnan(step_norms[key]):
                        step_norms[key] = 0.0
                    step_norms[key] += grad_sq
        for key, value in step_norms.items():
            if value >= 0.0:
                grad_history[key].append(value ** 0.5)
            else:
                grad_history[key].append(float("nan"))
        return {k: (v ** 0.5 if v >= 0.0 else float("nan")) for k, v in step_norms.items()}

    global_step = 0

    for epoch in range(1, CFG.epochs + 1):
        epoch_loss = 0.0
        steps = len(train_loader)
        start = time.time()
        log_info(f"\n[TRAIN-QA] Epoch {epoch}/{CFG.epochs} - steps: {steps}")

        train_metric_state = _init_metric_state()

        with tqdm(
            total=steps,
            desc=f"Epoch {epoch}/{CFG.epochs}",
            dynamic_ncols=True,
            disable=not _log_allowed("info"),
        ) as pbar:
            for batch in train_loader:
                patches = batch["patches"].to(device)
                pad_mask = batch["pad_mask"].to(device)

                if not debug_batch_meta_logged:
                    metadata = batch.get("metadata", {})
                    subjects = metadata.get("subject", [])
                    activities = metadata.get("activity_name", [])
                    sensors = metadata.get("sensor_path", [])
                    raw_rows = metadata.get("raw_row", [])
                    for idx, (q, a) in enumerate(zip(batch["questions"], batch["answers"])):
                        subj = subjects[idx] if idx < len(subjects) else None
                        act = activities[idx] if idx < len(activities) else None
                        sensor_path = sensors[idx] if idx < len(sensors) else None
                        row = raw_rows[idx] if idx < len(raw_rows) else None
                        session_id = None
                        if isinstance(row, dict):
                            session_id = row.get("activity_index")
                        log_debug(
                            f"[DEBUG] Batch sample {idx}: subject={subj}, activity={act}, sensor_path={sensor_path}, activity_index={session_id}"
                        )
                    debug_batch_meta_logged = True

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    loss, qa_info = model(patches, pad_mask, batch["questions"], batch["answers"])

                if not torch.isfinite(loss):
                    log_warn("[WARN] Non-finite loss encountered; skipping batch")
                    scheduler.step()
                    pbar.update(1)
                    continue

                logits = qa_info["logits"]
                labels = qa_info["labels"]
                label_mask = qa_info["label_mask"]
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    mask = label_mask.to(dtype=torch.bool)
                    _update_metrics(train_metric_state, preds, labels, mask, tokenizer)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    step_norms = record_encoder_grad_norms()
                    if not encoder_grad_logged:
                        log_debug(
                            "[DEBUG] Encoder grad norms (first batch): "
                            + ", ".join(f"{k}={v:.4f}" for k, v in step_norms.items())
                        )
                        encoder_grad_logged = True
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    step_norms = record_encoder_grad_norms()
                    if not encoder_grad_logged:
                        log_debug(
                            "[DEBUG] Encoder grad norms (first batch): "
                            + ", ".join(f"{k}={v:.4f}" for k, v in step_norms.items())
                        )
                        encoder_grad_logged = True
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                    optimizer.step()

                scheduler.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                train_batch_history.append(loss_value)

                if len(train_batch_history) % CFG.loss_plot_every == 0:
                    plot_batch_loss(train_batch_history)

                pbar.set_postfix(loss=f"{loss_value:.6f}")
                pbar.update(1)

                generation_every = getattr(CFG, "generation_every", 0)
                if (
                    generation_every
                    and generation_every > 0
                    and _log_allowed("info")
                    and (global_step % generation_every == 0)
                ):
                    tokenizer = model.qa_head.tokenizer
                    max_samples = min(2, preds.size(0))
                    for sample_idx in range(max_samples):
                        mask_i = mask[sample_idx]
                        if mask_i.any():
                            pred_ids = preds[sample_idx][mask_i].detach().cpu().tolist()
                            gt_ids = labels[sample_idx][mask_i].detach().cpu().tolist()
                            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                            gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
                            question = batch["questions"][sample_idx]
                            metadata = batch.get("metadata") if isinstance(batch, dict) else None
                            timestamps_sample = None
                            if isinstance(metadata, dict):
                                timestamps_list = metadata.get("timestamps")
                                if isinstance(timestamps_list, (list, tuple)) and sample_idx < len(timestamps_list):
                                    timestamps_sample = timestamps_list[sample_idx]
                            save_generation_visual(
                                epoch,
                                global_step,
                                sample_idx,
                                question,
                                pred_text,
                                gt_text,
                                patches[sample_idx],
                                qa_info.get("small_features", torch.empty(0))[sample_idx]
                                if "small_features" in qa_info
                                else torch.empty(0),
                                qa_info["tokens"][sample_idx],
                                qa_info.get("fused_patches", torch.empty(0))[sample_idx]
                                if "fused_patches" in qa_info
                                else torch.empty(0),
                                qa_info["pad_mask"][sample_idx],
                                timestamps_sample,
                            )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                global_step += 1

        avg_train_loss = epoch_loss / max(1, steps)
        train_epoch_losses.append(avg_train_loss)
        train_metrics = _summarize_metrics(train_metric_state)
        train_epoch_em.append(train_metrics["exact_match"])
        train_epoch_f1.append(train_metrics["f1"])
        train_epoch_bleu.append(train_metrics["bleu4"])
        train_epoch_rouge.append(train_metrics["rougeL"])
        log_info(
            "[EPOCH-QA] {} train_loss={:.6f} train_exact_match={:.4f} train_f1={:.4f} "
            "train_bleu4={:.4f} train_rougeL={:.4f} train_token_acc={:.4f} time={:.1f}s".format(
                epoch,
                avg_train_loss,
                train_metrics["exact_match"],
                train_metrics["f1"],
                train_metrics["bleu4"],
                train_metrics["rougeL"],
                train_metrics["token_accuracy"],
                time.time() - start,
            )
        )

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_steps = len(val_loader)
        val_metric_state = _init_metric_state()
        with torch.no_grad():
            for batch in val_loader:
                patches = batch["patches"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                loss, qa_info = model(patches, pad_mask, batch["questions"], batch["answers"])
                val_loss_total += loss.item()
                logits = qa_info["logits"]
                labels = qa_info["labels"]
                label_mask = qa_info["label_mask"]
                preds = logits.argmax(dim=-1)
                mask = label_mask.to(dtype=torch.bool)
                _update_metrics(val_metric_state, preds, labels, mask, tokenizer)

        avg_val_loss = val_loss_total / max(1, val_steps)
        val_epoch_losses.append(avg_val_loss)
        val_metrics = _summarize_metrics(val_metric_state)
        val_epoch_em.append(val_metrics["exact_match"])
        val_epoch_f1.append(val_metrics["f1"])
        val_epoch_bleu.append(val_metrics["bleu4"])
        val_epoch_rouge.append(val_metrics["rougeL"])
        log_info(
            "[VAL-QA] {} val_loss={:.6f} val_exact_match={:.4f} val_f1={:.4f} "
            "val_bleu4={:.4f} val_rougeL={:.4f} val_token_acc={:.4f}".format(
                epoch,
                avg_val_loss,
                val_metrics["exact_match"],
                val_metrics["f1"],
                val_metrics["bleu4"],
                val_metrics["rougeL"],
                val_metrics["token_accuracy"],
            )
        )

        plot_loss_curves(train_epoch_losses, val_epoch_losses)
        plot_metric_curve(
            train_epoch_em,
            val_epoch_em,
            "ActionSense QA Exact Match",
            "Exact Match",
            EM_PLOT_PATH,
        )
        plot_metric_curve(
            train_epoch_f1,
            val_epoch_f1,
            "ActionSense QA F1",
            "F1",
            F1_PLOT_PATH,
        )
        plot_bleu_rouge_curve(
            train_epoch_bleu,
            val_epoch_bleu,
            train_epoch_rouge,
            val_epoch_rouge,
            BLEU_ROUGE_PLOT_PATH,
        )
        model.train()

        if CFG.checkpoint_every > 0 and (epoch % CFG.checkpoint_every == 0):
            save_joint_checkpoint(model.encoder, model.qa_head, epoch)

    plot_batch_loss(train_batch_history)
    plot_metric_curve(
        train_epoch_em,
        val_epoch_em,
        "ActionSense QA Exact Match",
        "Exact Match",
        EM_PLOT_PATH,
    )
    plot_metric_curve(
        train_epoch_f1,
        val_epoch_f1,
        "ActionSense QA F1",
        "F1",
        F1_PLOT_PATH,
    )
    plot_bleu_rouge_curve(
        train_epoch_bleu,
        val_epoch_bleu,
        train_epoch_rouge,
        val_epoch_rouge,
        BLEU_ROUGE_PLOT_PATH,
    )
    plot_grad_history(grad_history)


if __name__ == "__main__":
    train()
