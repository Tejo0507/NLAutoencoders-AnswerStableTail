"""The correctness probe, and the leak-free score the stopping sweep uses.

The probe is the baseline the primary research question is measured against,
so how its per-boundary score is produced matters as much as how accurate it
is. `score_baselines` originally fitted on the train split and then scored
*every* problem, including the train-split ones - and those problems are in
the stopping sweep too. The probe's safety/saving curve was therefore partly
in-sample, which inflates the baseline in exactly the direction that makes the
verbalised readout look worse for a reason that is not about the readout.

The fix is grouped out-of-fold scoring: every row is scored by a probe fitted
without that row's problem. These tests pin that, and pin the two properties
the rest of the study relies on - grouping by problem, and collapse to chance
under shuffled labels (F6).
"""

from __future__ import annotations

import numpy as np
import pytest

from nlaast.baselines.probe import (
    CorrectnessProbe,
    cross_validate,
    difference_in_means_direction,
    grouped_folds,
    permute_labels,
)
from nlaast.config import ProbeConfig

CFG = ProbeConfig(n_folds=4, seed=11, max_iter=500)


def separable_corpus(n_problems=24, per_problem=5, d=16, seed=0):
    """Boundaries grouped by problem, with a real signal in one direction.

    The label is a property of the *problem*, not of the boundary, which is the
    structure the real corpus has: every boundary of a correct trace is
    labelled correct. That is also what makes an ungrouped split dishonest.
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    X, y, groups = [], [], []
    for p in range(n_problems):
        label = p % 2
        centre = (2.0 if label else -2.0) * direction
        for _ in range(per_problem):
            X.append(centre + rng.standard_normal(d) * 0.6)
            y.append(label)
            groups.append(f"p{p:03d}")
    return np.asarray(X), np.asarray(y), groups


class TestGroupedFolds:
    def test_a_problem_lands_entirely_in_one_fold(self):
        _, _, groups = separable_corpus()
        folds = grouped_folds(groups, 4, seed=11)
        for idx in folds:
            # No problem may appear in two folds, or a boundary from a training
            # problem would be scored as held out.
            fold_groups = {groups[i] for i in idx}
            for other in folds:
                if other is idx:
                    continue
                assert not (fold_groups & {groups[i] for i in other})

    def test_assignment_does_not_depend_on_corpus_size(self):
        """Hashed, not shuffled: a pilot's folds survive the corpus growing."""
        _, _, small = separable_corpus(n_problems=8)
        _, _, big = separable_corpus(n_problems=24)
        f_small = grouped_folds(small, 4, seed=11)
        f_big = grouped_folds(big, 4, seed=11)

        def mapping(groups, folds):
            out = {}
            for k, idx in enumerate(folds):
                for i in idx:
                    out[groups[i]] = k
            return out

        m_small, m_big = mapping(small, f_small), mapping(big, f_big)
        for g in m_small:
            assert m_small[g] == m_big[g]


class TestOutOfFoldScores:
    def test_every_row_gets_an_out_of_fold_score(self):
        X, y, groups = separable_corpus()
        rep = cross_validate(X, y, groups, CFG)
        oof = np.asarray(rep.oof)
        assert oof.shape == (len(y),)
        assert np.isfinite(oof).all()

    def test_the_out_of_fold_score_is_not_the_in_sample_one(self):
        """If they were the same the leak would be invisible, so check they
        differ and that the in-sample fit really is the optimistic one."""
        from sklearn.metrics import roc_auc_score

        X, y, groups = separable_corpus(per_problem=4, d=48, seed=3)
        rep = cross_validate(X, y, groups, CFG)
        in_sample = CorrectnessProbe(CFG).fit(X, y).predict_proba(X)
        oof = np.asarray(rep.oof)
        assert not np.allclose(oof, in_sample)
        assert roc_auc_score(y, in_sample) >= roc_auc_score(y, oof) - 1e-9

    def test_oof_is_kept_out_of_the_serialised_report(self):
        X, y, groups = separable_corpus()
        rep = cross_validate(X, y, groups, CFG)
        assert "oof" not in rep.to_dict()
        assert rep.to_dict()["notes"]["n_scored_out_of_fold"] == len(y)

    def test_an_unusable_corpus_still_returns_an_oof_array(self):
        """One class only: no fold can fit, and the caller needs a shaped
        result rather than an exception or a shorter array."""
        X = np.random.default_rng(0).standard_normal((40, 8))
        y = np.zeros(40, dtype=int)
        groups = [f"p{i // 5}" for i in range(40)]
        rep = cross_validate(X, y, groups, CFG)
        assert len(rep.oof) == 40
        assert not np.isfinite(np.asarray(rep.oof)).any()
        assert "error" in rep.notes


class TestLeakageControl:
    """F6. The only convincing evidence a probe is not leaking - and the
    reason the obvious version of it does not work on this corpus."""

    def test_the_across_group_shuffle_collapses_a_real_signal_to_chance(self):
        X, y, groups = separable_corpus(n_problems=40, d=12, seed=7)
        real = cross_validate(X, y, groups, CFG)
        shuffled = cross_validate(X, y, groups, CFG, shuffle_labels=True,
                                  shuffle_scope="across_groups")
        assert real.auc > 0.8
        assert abs(shuffled.auc - 0.5) < 0.25, shuffled.auc

    def test_the_within_group_shuffle_is_degenerate_when_a_problem_has_one_label(self):
        """Why the default scope changed.

        In the real corpus a trace's boundaries usually share one correctness
        label, so permuting inside the problem changes nothing: the "shuffled"
        probe scores exactly as the unshuffled one, which looks like total
        leakage and is in fact no control at all. The report has to say that
        the shuffle moved no labels.
        """
        X, y, groups = separable_corpus(n_problems=40, d=12, seed=7)
        shuffled = cross_validate(X, y, groups, CFG, shuffle_labels=True,
                                  shuffle_scope="within_group")
        info = shuffled.to_dict()["notes"]["shuffle"]
        assert info["n_labels_changed"] == 0
        assert shuffled.auc == pytest.approx(cross_validate(X, y, groups, CFG).auc)

    def test_the_within_group_shuffle_does_work_when_labels_vary_inside_a_problem(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 8))
        groups = [f"p{i // 5}" for i in range(200)]
        y = (X[:, 0] > 0).astype(int)  # varies within every problem
        shuffled = cross_validate(X, y, groups, CFG, shuffle_labels=True,
                                  shuffle_scope="within_group")
        assert shuffled.to_dict()["notes"]["shuffle"]["n_labels_changed"] > 0

    def test_shuffling_preserves_the_class_balance(self):
        X, y, groups = separable_corpus()
        for scope in ("across_groups", "within_group"):
            shuffled = cross_validate(X, y, groups, CFG, shuffle_labels=True,
                                      shuffle_scope=scope)
            assert shuffled.positive_rate == pytest.approx(float(y.mean()))

    def test_across_group_shuffle_keeps_group_sizes_and_moves_labels(self):
        X, y, groups = separable_corpus(n_problems=12, per_problem=4)
        permuted, info = permute_labels(y, groups, "across_groups", seed=3)
        assert permuted.shape == y.shape
        assert info["n_groups"] == 12
        assert np.sum(permuted != y) > 0

    def test_an_unknown_scope_is_refused(self):
        _, y, groups = separable_corpus(n_problems=4, per_problem=2)
        with pytest.raises(ValueError, match="unknown shuffle scope"):
            permute_labels(y, groups, "sideways", seed=1)


class TestDirection:
    def test_the_probe_direction_lives_in_activation_space(self):
        X, y, groups = separable_corpus()
        probe = CorrectnessProbe(CFG).fit(X, y)
        d = probe.direction
        assert d.shape == (X.shape[1],)
        assert np.linalg.norm(d) == pytest.approx(1.0)

    def test_difference_in_means_points_from_neg_to_pos(self):
        rng = np.random.default_rng(0)
        offset = np.zeros(6)
        offset[2] = 5.0
        neg = rng.standard_normal((30, 6))
        pos = neg + offset
        d = difference_in_means_direction(pos, neg)
        assert np.argmax(np.abs(d)) == 2
        assert d[2] > 0
        assert np.linalg.norm(d) == pytest.approx(1.0)

    def test_difference_in_means_refuses_an_empty_group(self):
        with pytest.raises(ValueError):
            difference_in_means_direction(np.zeros((0, 4)), np.ones((3, 4)))
