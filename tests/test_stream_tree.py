"""Batch tree contracts and an independent exhaustive split reference."""

from collections import Counter
from math import log2

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier


def _impurity(labels, criterion):
    probabilities = [count / len(labels) for count in Counter(labels).values()]
    if criterion == "gini":
        return 1 - sum(p**2 for p in probabilities)
    return -sum(p * log2(p) for p in probabilities)


def _brute_force_root(X, y, criterion):
    """Score each candidate from fresh subsets and frequency counts."""
    parent = _impurity(y, criterion)
    for feature, values in enumerate(X.T):
        candidates = []
        missing = np.isnan(values)
        observed = sorted(set(values[~missing]))
        for threshold in observed[:-1]:
            for missing_left in [False, True] if missing.any() else [False]:
                left = (values <= threshold) | (missing & missing_left)
                candidates.append((threshold, missing_left, left))
        if observed and missing.any():
            candidates.append((np.inf, False, ~missing))
        for threshold, missing_left, left in candidates:
            score = (
                left.sum() * _impurity(y[left], criterion)
                + (~left).sum() * _impurity(y[~left], criterion)
            ) / len(y)
            yield parent - score, feature, threshold, missing_left


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_vectorised_gains_agree_with_exhaustive_splits(criterion):
    rng = np.random.default_rng(860)
    # Duplicate feature values, three classes, constant columns and missing
    # values exercise the split boundaries and both missing-value routes.
    for _ in range(50):
        X = rng.integers(-2, 3, size=(13, 4)).astype(float)
        X[:, -1] = 0
        X[rng.random(X.shape) < 0.25] = np.nan
        y = rng.integers(0, 3, size=len(X))
        model = DecisionTreeClassifier(max_depth=1, criterion=criterion).fit(X, y)
        candidates = list(_brute_force_root(X, y, criterion))
        optimum = max((candidate[0] for candidate in candidates), default=0.0)
        root = model._tree.nodes[0]
        if optimum <= 1e-12:
            assert len(model._tree.nodes) == 1
            continue
        gain, feature, threshold, missing_left = next(
            candidate for candidate in candidates if candidate[0] >= optimum - 1e-12
        )
        assert root.split.gain == pytest.approx(gain, abs=1e-12)
        assert (root.split.feature, root.split.threshold, root.split.missing_left) == (
            feature, threshold, missing_left
        )
        left = np.where(np.isnan(X[:, feature]), missing_left, X[:, feature] <= threshold)
        for selected in [left, ~left]:
            expected = np.array([(y[selected] == label).mean() for label in model.classes_])
            np.testing.assert_allclose(
                model.predict_proba(X[selected]), np.tile(expected, (selected.sum(), 1))
            )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_twelve_student_course_example(criterion):
    # The four repeated rows and the exceptional fifth row come from the
    # Week 10 classification-tree walkthrough.
    X = np.array([
        [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0],
        [1, 0, 1], [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0],
    ])
    y = np.array([1] * 5 + [0] * 7)
    model = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(X, y)
    split = model._tree.nodes[0].split
    assert (split.feature, split.threshold) == (0, 0)
    if criterion == "gini":
        # Parent has 5 pass / 7 fail. Children are 0/5 and 5/2.
        assert split.gain == pytest.approx(35 / 72 - 5 / 21)
    np.testing.assert_array_equal(model.predict(X), [1] * 4 + [0] * 8)


def test_multiclass_string_probabilities_follow_sorted_labels():
    model = DecisionTreeClassifier(max_depth=1).fit(
        [[0], [0], [1], [1], [1]], ["wet", "fog", "sun", "sun", "sun"]
    )
    assert model.n_features_in_ == 1
    np.testing.assert_array_equal(model.classes_, ["fog", "sun", "wet"])
    assert not model.classes_.flags.writeable
    np.testing.assert_array_equal(model.predict([[0], [1]]), ["fog", "sun"])
    np.testing.assert_allclose(model.predict_proba([[0], [1]]), [[0.5, 0, 0.5], [0, 1, 0]])


@pytest.mark.parametrize("missing_label,side", [(0, True), (1, False)])
def test_learns_which_child_receives_missing_values(missing_label, side):
    tree = DecisionTreeClassifier(max_depth=1).fit([[0], [2], [np.nan]], [0, 1, missing_label])
    assert tree._tree.nodes[0].split.missing_left is side
    np.testing.assert_array_equal(tree.predict([[0], [2], [np.nan]]), [0, 1, missing_label])


def test_constant_observed_feature_can_split_on_missingness():
    tree = DecisionTreeClassifier().fit([[2], [2], [np.nan]], [0, 0, 1])
    np.testing.assert_array_equal(tree.predict([[-10], [9], [np.nan]]), [0, 0, 1])


def test_equal_gains_prefer_lower_feature_and_threshold():
    tree = DecisionTreeClassifier(max_depth=1).fit([[0, 0], [1, 1], [2, 2]], [0, 1, 0])
    split = tree._tree.nodes[0].split
    assert (split.feature, split.threshold, split.missing_left) == (0, 0, False)
    assert tree.predict([[np.nan, 0]]).item() == 0


def test_equal_missing_routes_favour_right():
    model = DecisionTreeClassifier(max_depth=1).fit(
        [[0], [0], [1], [1], [np.nan], [np.nan]], [0, 0, 1, 1, 0, 1]
    )
    assert model._tree.nodes[0].split.missing_left is False
    assert model.predict([[np.nan]]).item() == 1


def test_entropy_roundoff_does_not_split_balanced_groups():
    # Each feature value contains one row of each class, so every possible
    # child keeps the exact same class proportions as the parent.
    X = np.repeat(np.arange(5_000), 2).reshape(-1, 1)
    y = np.tile([0, 1], 5_000)
    tree = DecisionTreeClassifier(max_depth=1, criterion="entropy").fit(X, y)
    assert len(tree._tree.nodes) == 1
    np.testing.assert_allclose(tree.predict_proba([[0], [4_999]]), [[0.5, 0.5]] * 2)


@pytest.mark.parametrize("kwargs,X,y", [
    ({"max_depth": 0}, [[0], [1]], [0, 1]),
    ({"min_samples_split": 3}, [[0], [1]], [0, 1]),
    ({}, [[0], [1]], [2, 2]),
    ({}, [[np.nan], [np.nan]], [0, 1]),
    ({}, [[8], [8]], [0, 1]),
    ({}, [[0], [0], [1], [1]], [0, 1, 0, 1]),
])
def test_stopping_rules_produce_one_leaf(kwargs, X, y):
    tree = DecisionTreeClassifier(**kwargs).fit(X, y)
    assert len(tree._tree.nodes) == 1
    np.testing.assert_array_equal(tree.predict(X), np.full(len(X), min(y)))


def test_exact_boundaries_for_large_integers_and_mixed_query_dtype():
    maximum = np.iinfo(np.uint64).max
    tree = DecisionTreeClassifier().fit(np.array([[maximum - 1], [maximum]], dtype=np.uint64),
                                        [0, 1])
    np.testing.assert_array_equal(tree.predict(np.array([[maximum - 1], [maximum]], np.uint64)),
                                  [0, 1])
    assert tree.predict([[float(2**64)]]).item() == 1


@pytest.mark.parametrize("first,second", [
    (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2),
    (np.nextafter(np.finfo(float).max, 0.0), np.finfo(float).max),
])
def test_adjacent_float_values_remain_separable(first, second):
    X = [[first], [second]]
    np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, [0, 1]).predict(X), [0, 1])


@pytest.mark.parametrize("max_features", [1, 0.25, "sqrt", "log2"])
def test_feature_sampling_is_repeatable_without_using_global_rng(max_features):
    X = np.tile(np.arange(8)[:, None], (1, 8))
    y = X[:, 0] >= 4
    before = np.random.get_state()
    model = DecisionTreeClassifier(max_features=max_features, random_state=4).fit(X, y)
    feature = model._tree.nodes[0].split.feature
    assert feature > 0
    assert model.fit(X, y)._tree.nodes[0].split.feature == feature
    np.testing.assert_array_equal(model.predict(X), y)
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[0] == before[0] and after[2:] == before[2:]


def test_refit_replaces_classes_and_failed_fit_preserves_previous_tree():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    assert model.fit([[0, 0], [1, 0]], ["wet", "sun"]) is model
    with pytest.raises(ValueError, match="infinite"):
        model.fit([[np.inf]], ["bad"])
    np.testing.assert_array_equal(model.classes_, ["sun", "wet"])
    np.testing.assert_array_equal(model.predict([[0, 0], [1, 0]]), ["wet", "sun"])
    assert model.predict(np.empty((0, 2))).shape == (0,)
    assert model.predict_proba(np.empty((0, 2))).shape == (0, 2)
    with pytest.raises(ValueError, match="feature"):
        model.predict([[0]])


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_unfitted_prediction(method):
    with pytest.raises(NotFittedError):
        getattr(DecisionTreeClassifier(), method)([[0]])


def test_fit_accepts_strided_read_only_inputs_and_retains_no_training_data():
    backing = np.array([[0, 9], [2, 9], [3, 9]])
    X = backing[:, ::2]
    y = np.array([0, 1, 1])
    X.setflags(write=False)
    model = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(backing, [[0, 9], [2, 9], [3, 9]])
    np.testing.assert_array_equal(y, [0, 1, 1])
    backing[:, 0] = 0
    y[:] = 0
    np.testing.assert_array_equal(model.predict([[0], [2], [3]]), [0, 1, 1])


@pytest.mark.parametrize("X,y,error,match", [
    ([0], [0], ValueError, "shape"),
    ([[0], [1]], [0], ValueError, "same length"),
    ([[0]], [[0]], ValueError, "1 dimension"),
    (np.empty((0, 1)), [], ValueError, "at least one"),
    (np.empty((1, 0)), [1], ValueError, "at least one"),
    ([["0"]], [0], TypeError, "numeric"),
    ([[0]], [np.nan], ValueError, "NaN"),
    ([[0], [1]], np.array(["a", 1], dtype=object), TypeError, "comparable"),
])
def test_invalid_training_data(X, y, error, match):
    with pytest.raises(error, match=match):
        DecisionTreeClassifier().fit(X, y)


@pytest.mark.parametrize("kwargs,error", [
    ({"max_depth": -1}, ValueError),
    ({"max_depth": True}, TypeError),
    ({"min_samples_split": 1}, ValueError),
    ({"max_features": 2}, ValueError),
    ({"max_features": 0.0}, ValueError),
    ({"max_features": "all"}, ValueError),
    ({"criterion": "invalid"}, ValueError),
    ({"buffer_size": 0}, ValueError),
    ({"random_state": -1}, ValueError),
])
def test_invalid_configuration_is_rejected_at_fit(kwargs, error):
    with pytest.raises(error, match=next(iter(kwargs))):
        DecisionTreeClassifier(**kwargs).fit([[0], [1]], [0, 1])
