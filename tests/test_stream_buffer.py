"""Public buffering behaviour of the streaming tree."""

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError
from numcompute_stream import tree
from numcompute_stream.tree import DecisionTreeClassifier


def test_exact_threshold_trains_and_clears_the_buffer():
    model = DecisionTreeClassifier(buffer_size=5)
    assert model.partial_fit([[0], [1], [2]], [0, 0, 0]) is model
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (3, 0, 0)
    with pytest.raises(NotFittedError):
        model.predict([[0]])
    with pytest.raises(NotFittedError):
        model.predict_proba([[0]])

    model.partial_fit([[3], [4]], [1, 1])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 1, 5)
    np.testing.assert_array_equal(model.predict([[0], [4]]), [0, 1])


@pytest.mark.parametrize("trigger_size", [4, 13])
def test_overshoot_trains_once_with_every_row(trigger_size):
    model = DecisionTreeClassifier(buffer_size=5, max_depth=0)
    model.partial_fit(np.zeros((3, 1)), [0, 0, 0])
    model.partial_fit(np.zeros((trigger_size, 1)), np.ones(trigger_size, dtype=int))

    total = trigger_size + 3
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 1, total)
    np.testing.assert_allclose(model.predict_proba([[0]]), [[3 / total, trigger_size / total]])


def test_successive_retrains_use_only_pending_data():
    model = DecisionTreeClassifier(buffer_size=4)
    model.partial_fit([[0], [0], [0], [0]], [0, 0, 0, 1])
    first = model.predict_proba([[0]])
    np.testing.assert_allclose(first, [[0.75, 0.25]])

    model.partial_fit([[0], [0]], [2, 2])
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict([[0]]), [0])
    np.testing.assert_array_equal(model.predict_proba([[0]]), first)
    assert model.pending_rows_ == 2

    model.partial_fit([[0], [0]], [2, 2])
    np.testing.assert_array_equal(model.classes_, [2])
    np.testing.assert_array_equal(model.predict([[0]]), [2])
    np.testing.assert_array_equal(model.predict_proba([[0]]), [[1]])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 2, 4)


def test_retrain_matches_batch_fit_with_different_chunk_boundaries():
    X = np.array([[0, 1], [1, 1], [2, 0], [np.nan, 1], [4, 1], [5, 0]])
    y = np.array(["wet", "fog", "sun", "fog", "sun", "sun"])
    settings = {"buffer_size": 6, "criterion": "entropy", "max_features": 1, "random_state": 4}
    batch = DecisionTreeClassifier(**settings).fit(X, y)
    for split in [1, 2, 4]:
        streamed = DecisionTreeClassifier(**settings)
        streamed.partial_fit(X[:split], y[:split]).partial_fit(X[split:], y[split:])
        np.testing.assert_array_equal(streamed.classes_, batch.classes_)
        np.testing.assert_array_equal(streamed.predict(X), batch.predict(X))
        np.testing.assert_array_equal(streamed.predict_proba(X), batch.predict_proba(X))


def test_pending_data_is_independent_of_caller_changes():
    backing = np.array([[0, 99], [2, 99], [3, 99]])
    X = backing[:, ::2]
    X.setflags(write=False)
    y = np.array([0, 1, 1])
    model = DecisionTreeClassifier(buffer_size=4).partial_fit(X, y)
    backing[:] = 77
    y[:] = 9
    model.partial_fit([[4]], [1])
    np.testing.assert_array_equal(model.predict([[0], [2], [3], [4]]), [0, 1, 1, 1])


def test_empty_chunks_are_no_ops_before_and_after_fit():
    model = DecisionTreeClassifier(buffer_size=2)
    shape = np.empty((0, 1))
    model.partial_fit(shape, [])
    assert (model.pending_rows_, model.n_retrains_) == (0, 0)
    model.partial_fit([[0]], [0])
    model.partial_fit(shape, [])
    assert (model.pending_rows_, model.n_retrains_) == (1, 0)
    model.partial_fit([[1]], [1])
    model.partial_fit(shape, [])
    assert (model.pending_rows_, model.n_retrains_) == (0, 1)
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])


def test_many_small_chunks_form_one_batch():
    model = DecisionTreeClassifier(max_depth=0, buffer_size=100)
    for i in range(100):
        model.partial_fit([[i]], [i >= 50])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 1, 100)
    np.testing.assert_allclose(model.predict_proba([[0]]), [[0.5, 0.5]])


@pytest.mark.parametrize("X,y,error,match", [
    ([[1, 2]], [1], ValueError, "features"),
    ([[np.inf]], [1], ValueError, "infinite"),
    ([[1]], [], ValueError, "same length"),
    ([[1]], [np.nan], ValueError, "NaN"),
    ([[1]], ["text"], TypeError, "kind"),
])
def test_invalid_chunk_leaves_pending_rows_for_a_valid_retry(X, y, error, match):
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0]], [0])
    with pytest.raises(error, match=match):
        model.partial_fit(X, y)
    assert (model.pending_rows_, model.n_retrains_) == (1, 0)
    model.partial_fit([[1]], [1])
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])


def test_fitted_feature_count_is_checked_when_buffer_is_empty():
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="fitted with 1 features"):
        model.partial_fit([[2, 3]], [1])
    assert (model.pending_rows_, model.n_retrains_) == (0, 1)


def test_fit_resets_pending_data_only_after_success():
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 1])
    model.partial_fit([[2]], [9])
    with pytest.raises(ValueError, match="infinite"):
        model.fit([[np.inf]], [8])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (1, 1, 2)
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])

    model.fit([[0, 0]], [7])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 0, 0)
    model.partial_fit([[0, 1]], [6])
    assert model.pending_rows_ == 1
    assert model.predict([[0, 1]]).item() == 7


def test_failed_retrain_preserves_active_model_and_retry_does_not_duplicate(monkeypatch):
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 1])
    model.partial_fit([[1]], [2])

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    with monkeypatch.context() as patch:
        patch.setattr(tree, "_build_nodes", fail)
        with pytest.raises(RuntimeError, match="simulated training failure"):
            model.partial_fit([[2]], [3])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (1, 1, 2)
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])

    model.partial_fit([[2]], [3])
    assert (model.pending_rows_, model.n_retrains_, model.last_retrain_size_) == (0, 2, 2)
    np.testing.assert_array_equal(model.predict([[1], [2]]), [2, 3])


def test_mixed_feature_dtypes_accept_exact_conversion_and_nan():
    model = DecisionTreeClassifier(buffer_size=3)
    model.partial_fit([[0], [1]], [0, 0]).partial_fit([[np.nan]], [1])
    np.testing.assert_array_equal(model.predict([[0], [np.nan]]), [0, 1])


@pytest.mark.parametrize("integer_first", [True, False])
def test_lossy_feature_promotion_is_rejected_before_admission(integer_first):
    integer = np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64)
    floating = np.array([[0.0]])
    first, second = (integer, floating) if integer_first else (floating, integer)
    model = DecisionTreeClassifier(buffer_size=3).partial_fit(first, [0])
    with pytest.raises(ValueError, match="precision"):
        model.partial_fit(second, [1])
    assert model.pending_rows_ == 1
    model.partial_fit(np.repeat(first, 2, axis=0), [0, 0])
    assert (model.pending_rows_, model.last_retrain_size_) == (0, 3)


def test_mixed_numeric_label_dtypes_preserve_large_integer_classes():
    model = DecisionTreeClassifier(buffer_size=2)
    model.partial_fit([[0]], [float(2**53)])
    model.partial_fit([[1]], [2**53 + 1])
    assert len(model.classes_) == 2
    assert model.predict([[0], [1]]).tolist() == [float(2**53), 2**53 + 1]


def test_string_chunks_keep_the_full_label():
    model = DecisionTreeClassifier(buffer_size=2)
    model.partial_fit([[0]], ["dry"])
    model.partial_fit([[1]], ["heavy rain"])
    assert model.predict([[0], [1]]).tolist() == ["dry", "heavy rain"]
