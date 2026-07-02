"""Unit tests for evaluation protocol v2 (val_scripts/.../eval_v2.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.eval_v2 import (
    balanced_subsample_indices,
    classification_metrics,
    conse_embeddings,
    conse_predict,
    per_class_f1,
    predict_from_similarity,
    segment_predictions,
    soft_pool_patch_scores,
    subject_bootstrap_ci,
    subject_disjoint_split,
    window_ground_truth,
)


# =============================================================================
# Ground truth (the HARTH-bug regression test)
# =============================================================================

class TestWindowGroundTruth:
    def test_non_contiguous_codes_no_offset_bug(self):
        """HARTH regression: codes {2..8, 11} must map through idx_to_label
        verbatim — never shifted by the minimum code."""
        idx_to_label = {2: "lying", 3: "running", 11: "walking"}
        # 3 windows of 5 timesteps: majority codes 2, 11, 3
        labels_raw = np.zeros((3, 5, 2), dtype=np.int64)
        labels_raw[0, :, 0] = [2, 2, 2, 3, 3]
        labels_raw[1, :, 0] = [11, 11, 11, 11, 2]
        labels_raw[2, :, 0] = [3, 3, 3, 2, 11]
        labels_raw[:, :, 1] = [[7]] * 3

        gt, subj, keep = window_ground_truth(labels_raw, idx_to_label)
        assert gt == ["lying", "walking", "running"]
        assert subj.tolist() == [7, 7, 7]
        assert keep.tolist() == [0, 1, 2]

    def test_unknown_codes_dropped(self):
        idx_to_label = {0: "a", 1: "b"}
        labels_raw = np.zeros((3, 4, 2), dtype=np.int64)
        labels_raw[0, :, 0] = [0, 0, 1, 0]
        labels_raw[1, :, 0] = [5, 5, 5, 5]     # unknown code
        labels_raw[2, :, 0] = [-1, -1, -1, 1]  # majority -1 (unknown)
        gt, subj, keep = window_ground_truth(labels_raw, idx_to_label)
        assert gt == ["a"]
        assert keep.tolist() == [0]

    def test_subject_extraction(self):
        idx_to_label = {0: "a"}
        labels_raw = np.zeros((2, 3, 2), dtype=np.int64)
        labels_raw[0, :, 1] = 4
        labels_raw[1, :, 1] = 9
        _, subj, _ = window_ground_truth(labels_raw, idx_to_label)
        assert subj.tolist() == [4, 9]


# =============================================================================
# Subject-disjoint splitting
# =============================================================================

class TestSubjectDisjointSplit:
    def test_disjoint_and_complete(self):
        subjects = np.repeat(np.arange(10), 20)  # 10 subjects x 20 windows
        tr, va, te = subject_disjoint_split(subjects, seed=0)
        s_tr, s_va, s_te = (set(subjects[i] for i in idx) for idx in (tr, va, te))
        assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)
        assert len(tr) + len(va) + len(te) == len(subjects)

    def test_small_subject_count(self):
        subjects = np.repeat(np.arange(4), 5)  # opportunity-like: 4 subjects
        tr, va, te = subject_disjoint_split(subjects, seed=0)
        for idx in (tr, va, te):
            assert len(idx) > 0

    def test_raises_below_three_subjects(self):
        with pytest.raises(ValueError):
            subject_disjoint_split(np.array([0, 0, 1, 1]))

    def test_deterministic(self):
        subjects = np.repeat(np.arange(8), 3)
        a = subject_disjoint_split(subjects, seed=42)
        b = subject_disjoint_split(subjects, seed=42)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


class TestBalancedSubsample:
    def test_balance_and_rate(self):
        gt = ["a"] * 100 + ["b"] * 100
        indices = np.arange(200)
        picked = balanced_subsample_indices(indices, gt, rate=0.1, seed=0)
        names = np.asarray(gt)[picked]
        counts = {c: int((names == c).sum()) for c in ("a", "b")}
        assert counts["a"] == counts["b"] == 10


# =============================================================================
# Pooling & prediction
# =============================================================================

class TestPooling:
    def test_soft_pool_beats_vote_fragmentation(self):
        """3 patches weakly prefer label 0 individually collectively strong;
        with votes split 2-1 the vote and soft answers can differ. Construct a
        case where fragmented hard votes pick label 1 but soft pooling
        recovers label 0."""
        candidates = ["zero", "one", "two"]
        # patch sims: two patches marginally prefer "one"; one strongly "zero"
        sims = np.array([
            [[0.50, 0.51, 0.10],
             [0.50, 0.51, 0.10],
             [0.95, 0.10, 0.10]],
        ])
        masks = np.ones((1, 3), dtype=bool)
        vote = segment_predictions(sims, masks, candidates, mode="vote")
        soft = segment_predictions(sims, masks, candidates, mode="soft", tau=0.07)
        assert vote == ["one"]     # 2 votes vs 1
        assert soft == ["zero"]    # strong evidence dominates soft pool

    def test_soft_pool_masks_padding(self):
        sims = np.array([[0.9, 0.1], [0.0, 0.99]])
        mask = np.array([True, False])  # second patch is padding
        scores = soft_pool_patch_scores(sims, mask, tau=0.07)
        assert scores.argmax() == 0

    def test_predict_from_similarity_multiprototype(self):
        sims = np.zeros((1, 2, 3))       # (N, L, K) multi-prototype
        sims[0, 1, 2] = 0.9
        assert predict_from_similarity(sims, ["a", "b"]) == ["b"]


# =============================================================================
# Metrics
# =============================================================================

class TestMetrics:
    def test_macro_f1_over_gt_classes_only(self):
        """Classes absent from GT (e.g. HARTH cycling variants) must not drag
        macro-F1 down as automatic zeros."""
        gt = ["walking", "sitting", "walking", "sitting"]
        pred = ["walking", "sitting", "walking", "sitting"]
        m = classification_metrics(gt, pred)
        assert m["f1_macro"] == pytest.approx(100.0)
        assert m["n_gt_classes"] == 2

    def test_pred_only_class_not_counted_as_gt_class(self):
        gt = ["a", "a", "b", "b"]
        pred = ["a", "c", "b", "b"]  # 'c' never in GT
        m = classification_metrics(gt, pred)
        # f1(a)=2*1*0.5/1.5=0.667, f1(b)=1.0 -> macro=0.833
        assert m["f1_macro"] == pytest.approx(83.333, abs=0.01)
        pc = per_class_f1(gt, pred)
        assert set(pc) == {"a", "b"}

    def test_balanced_accuracy_matches_macro_recall(self):
        gt = ["a"] * 90 + ["b"] * 10
        pred = ["a"] * 90 + ["a"] * 10  # never predicts b
        m = classification_metrics(gt, pred)
        assert m["balanced_accuracy"] == pytest.approx(50.0)
        assert m["accuracy"] == pytest.approx(90.0)

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.RandomState(0)
        subjects = np.repeat(np.arange(12), 30)
        gt = rng.choice(["a", "b"], size=len(subjects)).tolist()
        pred = [g if rng.rand() < 0.8 else ("b" if g == "a" else "a") for g in gt]
        point = classification_metrics(gt, pred)["f1_macro"]
        ci = subject_bootstrap_ci(gt, pred, subjects, B=200, seed=1)
        assert ci["f1_macro_ci_lo"] <= point <= ci["f1_macro_ci_hi"]
        assert ci["n_subjects"] == 12


# =============================================================================
# ConSE bridge
# =============================================================================

def fake_encoder(labels):
    """Deterministic orthogonal-ish embeddings; identical strings -> identical
    vectors (mimics frozen SBERT for protocol tests)."""
    base = {
        "walking": [1, 0, 0, 0],
        "walking_upstairs": [0.8, 0.6, 0, 0],
        "upstairs": [0.75, 0.66, 0, 0],
        "sitting": [0, 0, 1, 0],
        "car_step_in": [0, 0, 0, 1],
    }
    out = np.array([base[l] for l in labels], dtype=np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class TestConSE:
    def test_t1_reduces_to_argmax_bridge(self):
        train = ["walking", "walking_upstairs", "sitting"]
        target = ["upstairs", "sitting", "walking"]
        probs = np.array([[0.1, 0.85, 0.05]])
        preds, _ = conse_predict(probs, train, target, encode=fake_encoder, top_T=1)
        assert preds == ["upstairs"]  # argmax train label maps nearest to upstairs

    def test_identical_string_behaves_as_exact_match(self):
        train = ["walking", "sitting"]
        target = ["walking", "sitting"]
        probs = np.array([[0.99, 0.01], [0.02, 0.98]])
        preds, _ = conse_predict(probs, train, target, encode=fake_encoder, top_T=10)
        assert preds == ["walking", "sitting"]

    def test_uncertainty_preserved_vs_argmax(self):
        """A 0.55/0.45 split between 'walking' and 'walking_upstairs' should
        pull the ConSE vector between them; here the blend lands on 'upstairs'
        while a pure argmax (T=1) would say 'walking'."""
        train = ["walking", "walking_upstairs", "sitting"]
        target = ["upstairs", "sitting", "walking"]
        probs = np.array([[0.55, 0.45, 0.0]])
        pred_t1, _ = conse_predict(probs, train, target, encode=fake_encoder, top_T=1)
        pred_blend, _ = conse_predict(probs, train, target, encode=fake_encoder, top_T=10)
        assert pred_t1 == ["walking"]
        # blended vector: 0.55*walking + 0.45*walking_upstairs, then argmax cos
        v = conse_embeddings(probs, fake_encoder(train), top_T=10)
        sims = v @ fake_encoder(target).T
        assert pred_blend == [["upstairs", "sitting", "walking"][int(sims.argmax())]]

    def test_conse_embeddings_normalized(self):
        train = ["walking", "walking_upstairs", "sitting"]
        probs = np.array([[0.3, 0.4, 0.3], [1.0, 0.0, 0.0]])
        v = conse_embeddings(probs, fake_encoder(train))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-9)

    def test_reachability_reports_structural_zeros(self):
        train = ["walking", "walking_upstairs", "sitting"]
        target = ["walking", "upstairs", "sitting", "car_step_in"]
        probs = np.ones((1, 3)) / 3
        _, info = conse_predict(probs, train, target, encode=fake_encoder)
        assert "car_step_in" not in info["reachable_classes"]
        assert info["reachability"] == pytest.approx(3 / 4)

    def test_probs_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            conse_embeddings(np.ones((2, 5)), fake_encoder(["walking", "sitting"]))
