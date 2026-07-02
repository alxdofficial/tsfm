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

    def test_unknown_and_negative_codes(self):
        """Vote over VALID codes only: an unknown out-of-vocab code is dropped;
        an all-negative window is dropped; but -1 must never win a tie and steal
        a window that has a real label present (Finding 3)."""
        idx_to_label = {0: "a", 1: "b"}
        labels_raw = np.zeros((4, 4, 2), dtype=np.int64)
        labels_raw[0, :, 0] = [0, 0, 1, 0]      # -> "a"
        labels_raw[1, :, 0] = [5, 5, 5, 5]      # out-of-vocab code -> dropped
        labels_raw[2, :, 0] = [-1, -1, -1, -1]  # all unknown -> dropped
        labels_raw[3, :, 0] = [-1, -1, 1, 1]    # -1 must not win; valid vote -> "b"
        gt, subj, keep = window_ground_truth(labels_raw, idx_to_label)
        assert gt == ["a", "b"]
        assert keep.tolist() == [0, 3]

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

    def test_large_cohort_reserves_two_test_subjects(self):
        """Floored val allocation gives test >=2 subjects on a 15-subject cohort
        (realworld) so its bootstrap CI is non-degenerate."""
        subjects = np.repeat(np.arange(15), 10)
        tr, va, te = subject_disjoint_split(subjects, seed=3431)
        assert len(np.unique(subjects[te])) >= 2

    def test_every_subject_assigned_no_dropout(self):
        for n in (3, 4, 5, 10, 15, 24):
            subjects = np.repeat(np.arange(n), 4)
            tr, va, te = subject_disjoint_split(subjects, seed=1)
            covered = set(subjects[tr]) | set(subjects[va]) | set(subjects[te])
            assert covered == set(range(n))
            assert len(tr) + len(va) + len(te) == len(subjects)


class TestBalancedSubsample:
    def test_balance_and_rate(self):
        gt = ["a"] * 100 + ["b"] * 100
        indices = np.arange(200)
        picked = balanced_subsample_indices(indices, gt, rate=0.1, seed=0)
        names = np.asarray(gt)[picked]
        counts = {c: int((names == c).sum()) for c in ("a", "b")}
        assert counts["a"] == counts["b"] == 10

    def test_water_fills_deficit_to_hit_budget(self):
        """A scarce class must not shrink the total; its deficit flows to
        classes with spare capacity (HARTH FS-10% regression)."""
        gt = ["rare"] * 3 + ["common"] * 97   # 100 windows
        indices = np.arange(100)
        picked, counts = balanced_subsample_indices(
            indices, gt, rate=0.10, seed=0, return_counts=True)
        # budget ~10; rare capped at 3, remaining 7 go to common
        assert counts["rare"] == 3
        assert counts["common"] == 7
        assert len(picked) == 10

    def test_never_below_one_per_class(self):
        gt = ["a"] * 50 + ["b"] * 50
        picked, counts = balanced_subsample_indices(
            np.arange(100), gt, rate=0.02, seed=0, return_counts=True)
        assert counts["a"] >= 1 and counts["b"] >= 1


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

    def test_false_positive_into_candidate_class_is_charged(self):
        """A model that dumps a false positive into a candidate class with no
        GT windows must be penalized (Finding 8): macro-F1 averages over GT ∪
        pred, so the spurious 'c' contributes an F1=0 term."""
        gt = ["a", "a", "b", "b"]
        pred = ["a", "c", "b", "b"]  # 'c' is a candidate string, never in GT
        m = classification_metrics(gt, pred)
        # classes {a,b,c}: f1(a)=2*1*0.5/1.5=0.667, f1(b)=1.0, f1(c)=0 -> macro=0.556
        assert m["f1_macro"] == pytest.approx(55.556, abs=0.01)
        assert m["n_scored_classes"] == 3
        pc = per_class_f1(gt, pred)
        assert set(pc) == {"a", "b", "c"} and pc["c"] == 0.0

    def test_balanced_accuracy_ignores_pred_only_classes(self):
        """balanced accuracy = macro recall over TRUE classes only (a class with
        no true samples has undefined recall and must not enter the average)."""
        gt = ["a", "a", "b", "b"]
        pred = ["a", "c", "b", "b"]
        m = classification_metrics(gt, pred)
        # recall(a)=0.5, recall(b)=1.0 -> 0.75; 'c' excluded
        assert m["balanced_accuracy"] == pytest.approx(75.0)

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

    def test_bootstrap_frozen_labelset_brackets_with_subject_exclusive_class(self):
        """P1 regression: a class living in only one subject makes ~1/3 of
        resamples drop it. With a per-replicate label set the macro-F1 estimand
        drifts and the CI floats above the point estimate; with the frozen set
        the point estimate must lie inside the interval."""
        # subjects 0,1 have only {a,b}; subject 2 uniquely holds class 'c'
        subjects = np.array([0] * 30 + [1] * 30 + [2] * 30)
        gt = ["a", "b"] * 15 + ["a", "b"] * 15 + ["c"] * 30
        pred = list(gt)
        # inject some errors so F1 < 100 and has spread
        for i in range(0, 90, 7):
            pred[i] = "a" if gt[i] != "a" else "b"
        point = classification_metrics(gt, pred)["f1_macro"]
        ci = subject_bootstrap_ci(gt, pred, subjects, B=500, seed=1)
        assert not ci["ci_degenerate"]
        assert ci["f1_macro_ci_lo"] - 1e-9 <= point <= ci["f1_macro_ci_hi"] + 1e-9

    def test_bootstrap_single_subject_is_flagged_not_zero_width(self):
        gt = ["a", "b", "a", "b"]
        pred = ["a", "b", "b", "b"]
        subjects = np.array([5, 5, 5, 5])  # one subject
        ci = subject_bootstrap_ci(gt, pred, subjects, B=100)
        assert ci["ci_degenerate"] is True
        assert np.isnan(ci["f1_macro_ci_lo"]) and np.isnan(ci["f1_macro_ci_hi"])
        assert ci["bootstrap_B"] == 0

    def test_bootstrap_balanced_accuracy_metric(self):
        subjects = np.repeat(np.arange(6), 20)
        rng = np.random.RandomState(2)
        gt = rng.choice(["a", "b", "c"], size=120).tolist()
        pred = [g if rng.rand() < 0.7 else "a" for g in gt]
        ci = subject_bootstrap_ci(gt, pred, subjects, metric="balanced_accuracy", B=200)
        assert not ci["ci_degenerate"]
        assert ci["balanced_accuracy_ci_lo"] <= ci["balanced_accuracy_ci_hi"]


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
        # T=1 nearest-neighbour lower bound: no train label maps to car_step_in
        assert "car_step_in" not in info["reachable_nn_lb"]
        assert info["reachability_lb"] == pytest.approx(3 / 4)
        # predicted_classes reflects what the actual top-T path hit on this data
        assert set(info["predicted_classes"]).issubset(set(target))

    def test_probs_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            conse_embeddings(np.ones((2, 5)), fake_encoder(["walking", "sitting"]))
