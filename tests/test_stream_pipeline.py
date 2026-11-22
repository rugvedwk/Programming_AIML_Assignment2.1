"""Integration checks for raw buffering and consistent model snapshots."""

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError, ThresholdClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_retrain_replays_every_raw_row_with_one_scaler():
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=3))])
    first = np.array([[0.0], [10.0]])
    y_first = np.array([0, 1])
    assert pipe.partial_fit(first, y_first) is pipe
    assert (pipe.pending_rows_, pipe.n_retrains_, pipe.fitted_) == (2, 0, False)
    with pytest.raises(NotFittedError):
        pipe.predict([[0]])

    # The buffer owns its observations. The second chunk overshoots the
    # threshold so all four original rows must go through one transform.
    first[:] = 99
    y_first[:] = 1
    pipe.partial_fit([[10], [10]], [1, 1])
    assert (pipe.pending_rows_, pipe.last_retrain_size_, pipe.n_retrains_) == (0, 4, 1)
    active = pipe.named_steps
    assert active["tree"].pending_rows_ == 0
    assert active["tree"].n_retrains_ == 0
    np.testing.assert_array_equal(active["scale"].n_samples_seen_, [4])
    raw = [[0], [10], [10], [10]]
    expected_scale = StandardScaler().fit(raw)
    expected_tree = DecisionTreeClassifier().fit(expected_scale.transform(raw), [0, 1, 1, 1])
    query = [[0], [1], [10]]
    np.testing.assert_allclose(pipe.transform(query), expected_scale.transform(query))
    np.testing.assert_array_equal(
        pipe.predict(query), expected_tree.predict(expected_scale.transform(query)),
    )
    np.testing.assert_array_equal(pipe.predict_proba(query), [[1, 0], [0, 1], [0, 1]])


def test_scale_shift_preserves_predictions_until_the_next_retrain():
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=3))])
    pipe.partial_fit([[0], [10], [10]], ["low", "high", "high"])
    probe = [[0], [10]]
    features = pipe.transform(probe)
    predictions = pipe.predict(probe)
    probabilities = pipe.predict_proba(probe)
    classes = pipe.classes_

    pipe.partial_fit([[1000]], ["new"])
    assert (pipe.pending_rows_, pipe.n_retrains_) == (1, 1)
    np.testing.assert_array_equal(pipe.transform(probe), features)
    np.testing.assert_array_equal(pipe.predict(probe), predictions)
    np.testing.assert_array_equal(pipe.predict_proba(probe), probabilities)
    np.testing.assert_array_equal(pipe.classes_, classes)

    pipe.partial_fit([[1001], [1002]], ["new", "new"])
    assert (pipe.pending_rows_, pipe.n_retrains_) == (0, 2)
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [3023 / 6])
    np.testing.assert_array_equal(pipe.predict(probe), ["new", "new"])
    np.testing.assert_array_equal(pipe.predict_proba(probe), [[1], [1]])


def test_category_growth_updates_model_width_only_at_retrain():
    pipe = Pipeline([("encode", OneHotEncoder()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([["red"], ["blue"]], [0, 1])
    expected = pipe.predict_proba([["red"], ["blue"], ["green"]])
    pipe.partial_fit([["green"]], [2])
    np.testing.assert_array_equal(pipe.transform([["green"]]), [[0, 0]])
    np.testing.assert_array_equal(pipe.predict_proba([["red"], ["blue"], ["green"]]), expected)
    assert pipe.named_steps["tree"].n_features_in_ == 2

    pipe.partial_fit([["red"]], [0])
    assert pipe.named_steps["tree"].n_features_in_ == 3
    assert pipe.named_steps["encode"].categories_ == ((0, "red"), (0, "blue"), (0, "green"))
    np.testing.assert_array_equal(pipe.transform([["green"]]), [[0, 0, 1]])
    np.testing.assert_array_equal(pipe.predict([["red"], ["green"]]), [0, 2])


def test_all_missing_numeric_chunk_followed_by_text_stays_missing_at_retrain():
    # Plain promotion to Unicode turned the first chunk's NaN into "nan",
    # which the strict encoder then rejected as an unseen category on replay.
    pipe = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="error")),
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ])
    pipe.partial_fit([[np.nan]], ["missing"])
    pipe.partial_fit([["rain"]], ["observed"])
    query = np.array([[np.nan], ["rain"]], dtype=object)
    np.testing.assert_array_equal(pipe.transform(query), np.eye(2))
    np.testing.assert_array_equal(pipe.predict(query), ["missing", "observed"])


def test_multiple_stages_fit_in_order_and_explicit_fit_resets_the_stream():
    pipe = Pipeline([
        ("impute", Imputer()), ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=3)),
    ])
    pipe.partial_fit([[0], [1], [2]], [0, 0, 1])
    pipe.partial_fit([[1000]], [1])
    X = [[1], [np.nan], [3], [7]]
    y = [0, 0, 1, 1]
    assert pipe.fit(X, y) is pipe
    imputed = Imputer().fit_transform(X)
    transformed = StandardScaler().fit_transform(imputed)
    np.testing.assert_allclose(pipe.transform(X), transformed)
    expected = DecisionTreeClassifier().fit(transformed, y).predict(transformed)
    np.testing.assert_array_equal(pipe.predict(X), expected)
    assert (pipe.pending_rows_, pipe.n_retrains_, pipe.last_retrain_size_) == (0, 0, 0)
    pipe.partial_fit([[8], [9], [10]], [1, 1, 1])
    active = pipe.named_steps
    np.testing.assert_array_equal(active["impute"].n_samples_seen_, [6])
    np.testing.assert_array_equal(active["scale"].n_samples_seen_, [7])


def test_failed_retrain_keeps_the_active_pair_and_drops_the_failed_trial(monkeypatch):
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0], [10]], [0, 1])
    pipe.partial_fit([[100]], [2])
    query = [[0], [10]]
    expected_features = pipe.transform(query)
    expected_prediction = pipe.predict(query)

    def fail_fit(self, X, y):
        raise RuntimeError("training interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(DecisionTreeClassifier, "fit", fail_fit)
        with pytest.raises(RuntimeError, match="training interrupted"):
            pipe.partial_fit([[200]], [3])

    assert (pipe.pending_rows_, pipe.n_retrains_, pipe.last_retrain_size_) == (1, 1, 2)
    np.testing.assert_array_equal(pipe.transform(query), expected_features)
    np.testing.assert_array_equal(pipe.predict(query), expected_prediction)
    pipe.partial_fit([[200]], [3])
    np.testing.assert_array_equal(pipe.named_steps["scale"].n_samples_seen_, [4])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [77.5])
    np.testing.assert_array_equal(pipe.predict([[100], [200]]), [2, 3])


def test_named_steps_cannot_mutate_the_live_pair():
    scale = StandardScaler()
    model = DecisionTreeClassifier(buffer_size=2)
    pipe = Pipeline([("scale", scale), ("tree", model)]).partial_fit([[0], [1]], [0, 1])
    assert not hasattr(scale, "n_features_in_")
    assert not hasattr(model, "classes_")
    active = pipe.named_steps
    active["scale"].fit([[200], [300]])
    active["tree"].fit([[0]], [9])
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [0, 1])


def test_bad_chunk_and_fit_leave_previous_state_available():
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0], [1]], [0, 1])
    pipe.partial_fit([[10]], [1])
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit([[2, 3]], [1])
    with pytest.raises(TypeError, match="kind"):
        pipe.partial_fit([[2]], ["wrong label kind"])
    with pytest.raises(TypeError, match="numeric"):
        pipe.fit([["text"]], [0])
    assert pipe.pending_rows_ == 1
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [0, 1])
    pipe.partial_fit([[20]], [1])
    np.testing.assert_array_equal(pipe.named_steps["scale"].n_samples_seen_, [4])


def test_classifier_only_pipeline_agrees_with_standalone_streaming_tree():
    pipe = Pipeline([("tree", DecisionTreeClassifier())], buffer_size=3)
    tree = DecisionTreeClassifier(buffer_size=3)
    assert not pipe.partial_fit(np.empty((0, 3)), []).fitted_
    for X, y in [([[0]], [0]), ([[1], [2]], [1, 1]), ([[10]], [2])]:
        pipe.partial_fit(X, y)
        tree.partial_fit(X, y)
        assert pipe.pending_rows_ == tree.pending_rows_
        assert pipe.n_retrains_ == tree.n_retrains_
        if pipe.fitted_:
            np.testing.assert_array_equal(pipe.predict(X), tree.predict(X))
    empty = np.empty((0, 1))
    assert pipe.predict(empty).shape == (0,)
    assert pipe.predict_proba(empty).shape == (0, 2)


def test_pipeline_can_retrain_a_fit_predict_classifier_with_explicit_threshold():
    pipe = Pipeline([("baseline", ThresholdClassifier())], buffer_size=3)
    pipe.partial_fit([[0]], [0]).partial_fit([[1], [9]], [0, 1])
    assert pipe.n_retrains_ == 1
    np.testing.assert_array_equal(pipe.predict([[0], [10]]), [0, 1])
    with pytest.raises(AttributeError, match="predict_proba"):
        pipe.predict_proba([[0]])


class _KeywordTransformer:
    def fit(self, X):
        return self

    def partial_fit(self, X, *, y):
        self.last_labels = y.tolist()
        return self

    def transform(self, X):
        return X


class _DropsRows(_KeywordTransformer):
    def transform(self, X):
        return X[1:]


def test_transformer_protocol_accepts_keyword_only_targets():
    pipe = Pipeline([("stage", _KeywordTransformer()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0]], [0]).partial_fit([[1]], [1])
    assert pipe.named_steps["stage"].last_labels == [1]
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [0, 1])


def test_shape_changing_transform_is_rejected_before_deployment():
    pipe = Pipeline([("drops", _DropsRows()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0]], [0])
    with pytest.raises(ValueError, match="drops.*2 rows"):
        pipe.partial_fit([[1]], [1])
    assert not pipe.fitted_
    assert pipe.pending_rows_ == 1


@pytest.mark.parametrize("steps,kwargs,error,match", [
    ([], {}, ValueError, "non-empty"),
    ([(" ", DecisionTreeClassifier())], {}, ValueError, "nonempty"),
    ([("step", Imputer()), ("step", DecisionTreeClassifier())], {}, ValueError, "unique"),
    ([("step", object()), ("tree", DecisionTreeClassifier())], {}, TypeError, "fit"),
    ([("tree", StandardScaler())], {}, TypeError, "predict"),
    ([("tree", DecisionTreeClassifier())], {"buffer_size": 0}, ValueError, "buffer_size"),
])
def test_invalid_pipeline_configuration(steps, kwargs, error, match):
    with pytest.raises(error, match=match):
        Pipeline(steps, **kwargs)
