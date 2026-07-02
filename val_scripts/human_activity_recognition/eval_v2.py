"""
Evaluation protocol v2 — consolidated, leakage-free scoring for HALO/TSFM and baselines.

Replaces the v1 open-set/closed-set/synonym-ontology machinery with:

  * ZS-XD: zero-shot cross-dataset classification against the TARGET dataset's own
    pre-registered label strings (benchmark_data/eval_v2/labels/{dataset}.json).
    Exact string match. No synonym groups anywhere in the scoring path.
  * Primary metric: macro-F1 (over classes present in ground truth) + balanced
    accuracy, per the ZSL evaluation canon (Xian et al., TPAMI 2018) and
    imbalance-aware ZSL-HAR practice.
  * Subject-disjoint splits for anything that trains on target data (few-shot),
    using the subject index stored in label_native.npy[:, 0, 1].
  * Subject-stratified bootstrap confidence intervals (windows within a subject
    are correlated; resampling windows would understate variance).
  * ConSE bridge (Norouzi et al., 2014) for closed-vocabulary baselines:
    convex combination of frozen-SBERT label embeddings weighted by the
    classifier's softmax, scored against the target vocabulary.

Ground-truth handling: label_native.npy codes ARE metadata activity_to_idx
values. We majority-vote the raw codes and map through idx_to_label. There is
deliberately NO offset arithmetic here (the v1 min-subtraction caused the
HARTH label bug).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_CONFIG_DIR = PROJECT_ROOT / "benchmark_data" / "eval_v2" / "labels"

# Pre-registered protocol constants (do not tune post-hoc)
CONSE_TOP_T = 10
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 3431
SOFT_POOL_TAU = 0.07  # matches training temperature


# =============================================================================
# Label configs & ground truth
# =============================================================================

def load_label_config(dataset: str) -> dict:
    """Load the pre-registered label config for a test dataset."""
    path = LABEL_CONFIG_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No pre-registered label config for '{dataset}' at {path}. "
            "Run benchmark_data/scripts/generate_eval_v2_labels.py first."
        )
    with open(path) as f:
        return json.load(f)


def window_ground_truth(
    labels_raw: np.ndarray,
    idx_to_label: Dict[int, str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Derive per-window ground truth names + subject ids from label_native.npy.

    Majority-votes the raw per-timestep activity codes and maps them through the
    dataset's own idx_to_label. No offset arithmetic — codes are activity_to_idx
    values by construction (see preprocess_tsfm_eval.py).

    Args:
        labels_raw: (N, W, 2) int array — [:, :, 0]=activity code, [:, :, 1]=subject.
        idx_to_label: authoritative code -> label-string mapping.

    Returns:
        gt_names: list of N' label strings (windows with unknown codes dropped)
        subjects: (N',) int subject id per window
        keep_idx: (N',) indices into the original N windows that were kept
    """
    act = labels_raw[:, :, 0].astype(np.int64)
    subj = labels_raw[:, 0, 1].astype(np.int64)

    # Majority vote tolerating possible -1 (unknown) codes: shift by +1 for bincount.
    shifted = act + 1
    window_codes = np.array(
        [np.bincount(row).argmax() - 1 for row in shifted], dtype=np.int64
    )

    keep = np.array([c in idx_to_label for c in window_codes], dtype=bool)
    keep_idx = np.nonzero(keep)[0]
    gt_names = [idx_to_label[int(c)] for c in window_codes[keep]]
    return gt_names, subj[keep], keep_idx


# =============================================================================
# Subject-disjoint splitting
# =============================================================================

def subject_disjoint_split(
    subjects: np.ndarray,
    fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split window indices into train/val/test with DISJOINT subject sets.

    Subjects are shuffled and partitioned by the given fractions (of subjects,
    not windows). Every split receives at least one subject; requires >= 3
    unique subjects.

    Returns three index arrays into `subjects`.
    """
    uniq = np.unique(subjects)
    if len(uniq) < 3:
        raise ValueError(
            f"subject_disjoint_split needs >=3 unique subjects, got {len(uniq)}"
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)

    n = len(perm)
    n_train = max(1, int(round(n * fracs[0])))
    n_val = max(1, int(round(n * fracs[1])))
    # Ensure test gets at least one subject
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_subj = set(perm[:n_train].tolist())
    val_subj = set(perm[n_train:n_train + n_val].tolist())
    test_subj = set(perm[n_train + n_val:].tolist())

    assert train_subj.isdisjoint(val_subj) and train_subj.isdisjoint(test_subj) \
        and val_subj.isdisjoint(test_subj), "subject splits must be disjoint"
    assert train_subj and val_subj and test_subj, "every split needs >=1 subject"

    in_train = np.isin(subjects, list(train_subj))
    in_val = np.isin(subjects, list(val_subj))
    in_test = np.isin(subjects, list(test_subj))
    return np.nonzero(in_train)[0], np.nonzero(in_val)[0], np.nonzero(in_test)[0]


def balanced_subsample_indices(
    indices: np.ndarray,
    gt_names: Sequence[str],
    rate: float,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Class-balanced subsample of `indices` down to ~rate of its size."""
    rng = np.random.RandomState(seed)
    names = np.asarray(gt_names)[indices]
    classes = np.unique(names)
    n_total = max(1, int(len(indices) * rate))
    n_per_class = max(1, n_total // len(classes))

    picked = []
    for c in classes:
        cls_idx = indices[names == c]
        cls_idx = rng.permutation(cls_idx)
        picked.extend(cls_idx[:n_per_class].tolist())
    picked = np.array(picked, dtype=np.int64)
    return rng.permutation(picked)


# =============================================================================
# Scoring: similarities -> predictions
# =============================================================================

def _as_2d_label_sims(sims: np.ndarray) -> np.ndarray:
    """Collapse multi-prototype similarity (N, L, K) -> (N, L) by max over K."""
    if sims.ndim == 3:
        return sims.max(axis=-1)
    return sims


def predict_from_similarity(
    sims: np.ndarray,
    candidates: Sequence[str],
) -> List[str]:
    """argmax over candidate labels. sims: (N, L)."""
    sims = _as_2d_label_sims(np.asarray(sims))
    idx = sims.argmax(axis=1)
    return [candidates[i] for i in idx]


def soft_pool_patch_scores(
    patch_sims: np.ndarray,
    patch_mask: np.ndarray,
    tau: float = SOFT_POOL_TAU,
) -> np.ndarray:
    """Soft logit-pooling of per-patch label scores into one segment score.

    score[k] = sum_t softmax(patch_sims[t] / tau)[k] over valid patches.
    Statistically stronger than hard per-patch majority voting (no vote
    fragmentation, no first-index tie bias).

    Args:
        patch_sims: (P, L) per-patch cosine similarities
        patch_mask: (P,) bool, True = valid patch
        tau: softmax temperature (pre-registered = training temperature)

    Returns:
        (L,) pooled segment scores
    """
    sims = np.asarray(patch_sims, dtype=np.float64)[np.asarray(patch_mask, dtype=bool)]
    if sims.size == 0:
        raise ValueError("no valid patches to pool")
    z = sims / tau
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    return p.sum(axis=0)


def segment_predictions(
    patch_sims: np.ndarray,
    patch_masks: np.ndarray,
    candidates: Sequence[str],
    mode: str = "soft",
    tau: float = SOFT_POOL_TAU,
) -> List[str]:
    """Segment-level predictions from per-patch similarities.

    Args:
        patch_sims: (N, P, L) per-patch similarities (padded)
        patch_masks: (N, P) bool valid-patch masks
        candidates: L label strings
        mode: 'soft' (pre-registered default) or 'vote' (legacy diagnostic)
    """
    preds = []
    for i in range(patch_sims.shape[0]):
        if mode == "soft":
            scores = soft_pool_patch_scores(patch_sims[i], patch_masks[i], tau)
        elif mode == "vote":
            valid = np.asarray(patch_masks[i], dtype=bool)
            votes = np.asarray(patch_sims[i])[valid].argmax(axis=1)
            scores = np.bincount(votes, minlength=len(candidates))
        else:
            raise ValueError(f"unknown pooling mode: {mode}")
        preds.append(candidates[int(np.argmax(scores))])
    return preds


# =============================================================================
# Metrics
# =============================================================================

def classification_metrics(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
) -> Dict[str, float]:
    """v2 metric set. Macro-F1 is computed over classes PRESENT IN GROUND TRUTH
    (classes with zero test windows — e.g. HARTH's cycling variants — are
    excluded rather than counted as automatic zeros)."""
    gt_classes = sorted(set(gt_names))
    return {
        "f1_macro": f1_score(gt_names, pred_names, labels=gt_classes,
                             average="macro", zero_division=0) * 100,
        "balanced_accuracy": balanced_accuracy_score(gt_names, pred_names) * 100,
        "accuracy": accuracy_score(gt_names, pred_names) * 100,
        "f1_weighted": f1_score(gt_names, pred_names, labels=gt_classes,
                                average="weighted", zero_division=0) * 100,
        "n_samples": len(gt_names),
        "n_gt_classes": len(gt_classes),
    }


def per_class_f1(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
) -> Dict[str, float]:
    gt_classes = sorted(set(gt_names))
    scores = f1_score(gt_names, pred_names, labels=gt_classes,
                      average=None, zero_division=0)
    return {c: float(s) * 100 for c, s in zip(gt_classes, scores)}


def subject_bootstrap_ci(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    subjects: np.ndarray,
    metric: str = "f1_macro",
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Subject-stratified bootstrap CI: resample SUBJECTS with replacement
    (windows within a subject are correlated — resampling windows would
    understate variance)."""
    gt = np.asarray(gt_names)
    pred = np.asarray(pred_names)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    subj_windows = {s: np.nonzero(subjects == s)[0] for s in uniq}

    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(B):
        sample_subj = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([subj_windows[s] for s in sample_subj])
        m = classification_metrics(gt[idx].tolist(), pred[idx].tolist())
        stats.append(m[metric])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        f"{metric}_ci_lo": float(lo),
        f"{metric}_ci_hi": float(hi),
        "bootstrap_B": B,
        "n_subjects": int(len(uniq)),
    }


# =============================================================================
# ConSE bridge for closed-vocabulary baselines (Norouzi et al., 2014)
# =============================================================================

_SBERT_CACHE: dict = {}


def get_sbert_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[Sequence[str]], np.ndarray]:
    """Frozen SBERT mean-pool encoder used by the ConSE bridge (same encoder
    for every bridged model). Labels are de-underscored before encoding."""
    if model_name not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    sbert = _SBERT_CACHE[model_name]

    def encode(labels: Sequence[str]) -> np.ndarray:
        texts = [l.replace("_", " ") for l in labels]
        return np.asarray(sbert.encode(texts, normalize_embeddings=True))

    return encode


def conse_embeddings(
    probs: np.ndarray,
    train_vocab_embs: np.ndarray,
    top_T: int = CONSE_TOP_T,
) -> np.ndarray:
    """ConSE semantic embedding: probability-weighted convex combination of the
    top-T training-label embeddings.

    Args:
        probs: (N, K) classifier softmax over its own training vocabulary
        train_vocab_embs: (K, D) L2-normalized embeddings of the training labels
        top_T: number of top classes to combine (pre-registered = 10)

    Returns:
        (N, D) L2-normalized semantic embeddings
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != train_vocab_embs.shape[0]:
        raise ValueError(
            f"probs {probs.shape} incompatible with vocab embeddings "
            f"{train_vocab_embs.shape}"
        )
    T = min(top_T, probs.shape[1])
    top_idx = np.argsort(-probs, axis=1)[:, :T]                      # (N, T)
    top_p = np.take_along_axis(probs, top_idx, axis=1)               # (N, T)
    denom = top_p.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    w = top_p / denom
    v = np.einsum("nt,ntd->nd", w, train_vocab_embs[top_idx])        # (N, D)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return v / norms


def conse_predict(
    probs: np.ndarray,
    train_vocab: Sequence[str],
    target_labels: Sequence[str],
    encode: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    top_T: int = CONSE_TOP_T,
) -> Tuple[List[str], Dict[str, object]]:
    """Full ConSE bridge: classifier softmax over train vocab -> predictions
    over the target dataset's label strings, plus reachability stats.

    Returns:
        pred_names: N predictions among target_labels
        info: {'reachable_classes', 'reachability', 'top_T'}
    """
    if encode is None:
        encode = get_sbert_encoder()
    train_embs = encode(train_vocab)
    target_embs = encode(target_labels)

    v = conse_embeddings(probs, train_embs, top_T=top_T)             # (N, D)
    sims = v @ target_embs.T                                         # (N, L)
    preds = [target_labels[i] for i in sims.argmax(axis=1)]

    # Reachability: a target class is reachable iff at least one single
    # training label maps nearest to it (T=1 sufficiency criterion).
    nn_of_train = (train_embs @ target_embs.T).argmax(axis=1)        # (K,)
    reachable = sorted({target_labels[i] for i in nn_of_train})
    info = {
        "reachable_classes": reachable,
        "reachability": len(reachable) / len(target_labels),
        "top_T": top_T,
    }
    return preds, info
