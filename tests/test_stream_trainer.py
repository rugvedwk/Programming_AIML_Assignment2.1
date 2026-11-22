"""Pre-update scoring, logs and failure behaviour."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def test_training_chunk_is_scored_with_the_previous_snapshot():
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=0, buffer_size=2)),
    ])
    trainer = StreamTrainer(pipeline, pos_label="yes", window_size=3)
    first = trainer.fit_chunk([[0]], ["no"])
    assert not first.retrained and first.pending_rows == 1 and first.model_array_bytes > 0
    assert trainer.score_chunk([[1]], ["no"]) is None
    second = trainer.fit_chunk([[1]], ["no"])
    assert second.retrained and second.retrain_seconds > 0 and second.pending_rows == 0
    assert second.scored_samples == 0
    assert np.isnan(second.chunk_accuracy) and np.isnan(second.cumulative_accuracy)

    X, y = [[100], [200]], ["yes", "yes"]
    preview = trainer.score_chunk(X, y)
    assert preview.accuracy == 0
    np.testing.assert_array_equal(preview.y_pred, ["no", "no"])
    np.testing.assert_array_equal(preview.y_score, [0, 0])
    assert len(trainer.logs) == 2 and trainer.result()["count"] == 0

    third = trainer.fit_chunk(X, y)
    assert third.retrained and third.scored_samples == 2 and third.rows_seen == 4
    assert third.chunk_accuracy == third.cumulative_accuracy == third.rolling_accuracy == 0
    np.testing.assert_array_equal(trainer.latest_score.y_pred, ["no", "no"])
    np.testing.assert_array_equal(pipeline.predict(X), y)
    assert trainer.result()["count"] == 2

    fourth = trainer.fit_chunk([[300], [400]], ["yes", "no"])
    assert fourth.chunk_accuracy == 0.5 and fourth.cumulative_accuracy == 0.25
    assert fourth.rolling_accuracy == pytest.approx(1 / 3)
    assert fourth.cumulative_auc == pytest.approx(1 / 6)
    assert fourth.rolling_auc == pytest.approx(1 / 4)
    assert trainer.result(rolling=True)["count"] == 3
    assert [log.chunk for log in trainer.logs] == [1, 2, 3, 4]


def test_auc_uses_positive_probability_column_instead_of_hard_labels():
    X = [[0], [1], [2], [3], [4], [5]]
    model = DecisionTreeClassifier(max_depth=1, buffer_size=100).fit(X, [0, 0, 1, 0, 1, 1])
    trainer = StreamTrainer(model, pos_label=0)
    expected = model.predict_proba(X)[:, 0]
    previews = trainer.score_chunk(X, [0, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(previews.y_score, expected)
    assert np.any((expected > 0) & (expected < 1))
    log = trainer.fit_chunk(X, [0, 0, 1, 0, 1, 1])
    assert not log.retrained and log.retrain_seconds == 0 and log.pending_rows == 6
    assert trainer.result()["auc_count"] == 6


def test_failed_update_does_not_record_prior_score(monkeypatch):
    model = DecisionTreeClassifier(buffer_size=2)
    trainer = StreamTrainer(model)
    trainer.fit_chunk([[0], [1]], [0, 1])
    trainer.fit_chunk([[0]], [0])
    expected = trainer.latest_score

    def fail(*args):
        raise RuntimeError("training interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(model, "partial_fit", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            trainer.fit_chunk([[1]], [1])
    assert trainer.result()["count"] == 1
    assert trainer.latest_score is expected
    assert len(trainer.logs) == 2 and model.pending_rows_ == 1
    assert trainer.fit_chunk([[1]], [1]).scored_samples == 2


def test_disabled_auc_and_empty_chunks():
    model = DecisionTreeClassifier(buffer_size=1)
    trainer = StreamTrainer(model, track_auc=False)
    assert trainer.fit_chunk(np.empty((0, 1)), []) is None
    assert not trainer.logs
    trainer.fit_chunk([[1]], [1])
    assert trainer.score_chunk([[1]], [1]).y_score is None
    trainer.fit_chunk([[1]], [1])
    assert trainer.result()["auc_count"] == 0
    with pytest.raises(ValueError, match="window_size"):
        trainer.result(rolling=True)


def test_invalid_input_is_rejected_before_unfitted_training():
    model = DecisionTreeClassifier(buffer_size=4)
    trainer = StreamTrainer(model)
    with pytest.raises(ValueError, match="same length"):
        trainer.fit_chunk([[1], [2]], [1])
    with pytest.raises(ValueError, match="infinite"):
        trainer.fit_chunk([[np.inf]], [1])
    assert not trainer.logs and model.pending_rows_ == 0
