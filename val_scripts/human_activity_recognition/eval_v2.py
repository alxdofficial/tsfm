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
from sklearn.metrics import accuracy_score, f1_score, recall_score

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

    # Majority vote over VALID codes only. Negative codes (e.g. -1 = unknown /
    # dropped-during-conversion) never win a tie and are excluded from the vote;
    # a window that is all-negative gets code -1 and is dropped below. (Voting
    # over the shifted array would let -1 win ties at bincount index 0.)
    window_codes = np.empty(len(act), dtype=np.int64)
    for i in range(len(act)):
        valid = act[i][act[i] >= 0]
        window_codes[i] = int(np.bincount(valid).argmax()) if valid.size else -1

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
    # Floor the val allocation so the remainder falls to TEST — this hands test
    # >=2 subjects whenever the cohort is large enough (e.g. n=15 -> 12/1/2),
    # giving a non-degenerate bootstrap CI. Genuinely small cohorts (shoaib=10,
    # opportunity=4) still yield 1 test subject; that is flagged downstream.
    n_val = max(1, int(n * fracs[1]))
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
    return_counts: bool = False,
):
    """As-balanced-as-possible subsample of `indices` down to ~rate of its size.

    Uses water-filling: classes are filled scarce-first with an equal share of
    the remaining budget, and any deficit from a class that runs out of windows
    is redistributed to classes that still have spare capacity. This keeps the
    total at ~rate*N (a plain per-class cap silently under-samples and would
    make e.g. HARTH FS-10% both too small and imbalanced), while remaining as
    balanced as the data allows. Achieved per-class counts are returned when
    `return_counts=True` so the caller can record them.
    """
    rng = np.random.RandomState(seed)
    indices = np.asarray(indices)
    names = np.asarray(gt_names)[indices]
    classes, counts = np.unique(names, return_counts=True)
    avail = dict(zip(classes.tolist(), counts.tolist()))
    n_total = max(len(classes), int(len(indices) * rate))

    # Water-fill quotas, scarce class first.
    order = sorted(classes.tolist(), key=lambda c: avail[c])
    quota = {c: 0 for c in classes.tolist()}
    remaining = n_total
    k = len(order)
    for i, c in enumerate(order):
        share = remaining // (k - i)
        take = min(avail[c], share)
        quota[c] = take
        remaining -= take

    picked = []
    achieved = {}
    for c in classes.tolist():
        cls_idx = rng.permutation(indices[names == c])
        take = quota[c]
        picked.extend(cls_idx[:take].tolist())
        achieved[c] = int(take)
    picked = rng.permutation(np.array(picked, dtype=np.int64))
    if return_counts:
        return picked, achieved
    return picked


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

def macro_f1_classes(gt_names: Sequence[str], pred_names: Sequence[str]) -> List[str]:
    """The class set macro-F1 is averaged over: ground-truth classes UNION
    predicted classes. Union (sklearn's default) charges false positives that a
    model routes into candidate classes with zero test windows — e.g. HALO
    predicting HARTH's `cycling_sit` (a real L_D string with no test windows)
    is penalized, not silently exempt. GT-only would let those FPs escape;
    full-L_D would inject automatic F1=0 for never-relevant classes and
    over-penalize."""
    return sorted(set(gt_names) | set(pred_names))


def classification_metrics(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
    f1_classes: Sequence[str] = None,
    recall_classes: Sequence[str] = None,
) -> Dict[str, float]:
    """v2 metric set.

    - macro-F1 (primary): averaged over `f1_classes` = GT ∪ predicted classes.
    - balanced accuracy = macro recall over `recall_classes` = GT classes only
      (recall is undefined for a class with no true samples).
    `f1_classes`/`recall_classes` may be pinned by the caller (the bootstrap
    freezes them on the full sample so every replicate scores the SAME estimand).
    """
    gt_names = list(gt_names)
    pred_names = list(pred_names)
    if f1_classes is None:
        f1_classes = macro_f1_classes(gt_names, pred_names)
    if recall_classes is None:
        recall_classes = sorted(set(gt_names))
    return {
        "f1_macro": f1_score(gt_names, pred_names, labels=list(f1_classes),
                             average="macro", zero_division=0) * 100,
        "balanced_accuracy": recall_score(gt_names, pred_names, labels=list(recall_classes),
                                          average="macro", zero_division=0) * 100,
        "accuracy": accuracy_score(gt_names, pred_names) * 100,
        "f1_weighted": f1_score(gt_names, pred_names, labels=list(f1_classes),
                                average="weighted", zero_division=0) * 100,
        "n_samples": len(gt_names),
        "n_gt_classes": len(set(gt_names)),
        "n_scored_classes": len(f1_classes),
    }


def per_class_f1(
    gt_names: Sequence[str],
    pred_names: Sequence[str],
) -> Dict[str, float]:
    classes = macro_f1_classes(gt_names, pred_names)
    scores = f1_score(gt_names, pred_names, labels=classes,
                      average=None, zero_division=0)
    return {c: float(s) * 100 for c, s in zip(classes, scores)}


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
    understate variance).

    Two correctness guards from the M0 debug sweep:
      * The scoring class set is FROZEN once on the full sample and reused for
        every replicate. Re-deriving it per replicate (as an earlier version
        did) makes replicates that drop a subject-exclusive class average
        macro-F1 over fewer classes — a different estimand — so the interval
        need not bracket the point estimate.
      * With < 2 subjects a subject-bootstrap has no variance to resample; we
        return a NaN interval flagged `ci_degenerate` rather than a fake
        zero-width 95% CI.
    """
    gt = np.asarray(gt_names)
    pred = np.asarray(pred_names)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)

    if len(uniq) < 2:
        return {
            f"{metric}_ci_lo": float("nan"),
            f"{metric}_ci_hi": float("nan"),
            "bootstrap_B": 0,
            "n_subjects": int(len(uniq)),
            "ci_degenerate": True,
        }

    f1_classes = macro_f1_classes(gt.tolist(), pred.tolist())
    recall_classes = sorted(set(gt.tolist()))

    def score(g, p) -> float:
        if metric == "f1_macro":
            return f1_score(g, p, labels=f1_classes, average="macro", zero_division=0) * 100
        if metric == "balanced_accuracy":
            return recall_score(g, p, labels=recall_classes, average="macro", zero_division=0) * 100
        if metric == "accuracy":
            return accuracy_score(g, p) * 100
        raise ValueError(f"unsupported bootstrap metric: {metric}")

    subj_windows = {s: np.nonzero(subjects == s)[0] for s in uniq}
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(B):
        sample_subj = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([subj_windows[s] for s in sample_subj])
        stats.append(score(gt[idx].tolist(), pred[idx].tolist()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        f"{metric}_ci_lo": float(lo),
        f"{metric}_ci_hi": float(hi),
        "bootstrap_B": B,
        "n_subjects": int(len(uniq)),
        "ci_degenerate": False,
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
        info: reachability stats. `reachable_nn_lb` is a T=1 nearest-neighbour
              LOWER BOUND on which target classes the bridge can output (a
              training label maps nearest to them); the actual top-T convex
              combinations can also land on other classes, so `predicted_classes`
              (the classes actually hit on this data) is reported alongside.
    """
    if encode is None:
        encode = get_sbert_encoder()
    train_embs = encode(train_vocab)
    target_embs = encode(target_labels)

    v = conse_embeddings(probs, train_embs, top_T=top_T)             # (N, D)
    sims = v @ target_embs.T                                         # (N, L)
    preds = [target_labels[i] for i in sims.argmax(axis=1)]

    nn_of_train = (train_embs @ target_embs.T).argmax(axis=1)        # (K,)
    reachable_lb = sorted({target_labels[i] for i in nn_of_train})
    predicted = sorted(set(preds))
    info = {
        "reachable_nn_lb": reachable_lb,
        "reachability_lb": len(reachable_lb) / len(target_labels),
        "predicted_classes": predicted,
        "n_predicted_classes": len(predicted),
        "top_T": top_T,
    }
    return preds, info
