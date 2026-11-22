"""Streaming classification metrics against independent batch references."""

import numpy as np
import pytest

from numcompute.metrics import accuracy, confusion_matrix, f1, precision, recall
from numcompute_stream.metrics import StreamingMetrics


def _pairwise_auc(truth, scores, pos_label):
    if len(set(truth)) != 2 or pos_label not in truth:
        return np.nan
    positive = scores[truth == pos_label]
    negative = scores[truth != pos_label]
    pairs = positive[:, None] - negative
    return np.mean((pairs > 0) + 0.5 * (pairs == 0))


@pytest.mark.parametrize("average", ["binary", "macro", "micro"])
def test_cumulative_metrics_match_batch_reference(average):
    # The third label appears only in the second chunk.
    truth = np.array([0, 1, 1, 0, 2, 2])
    pred = np.array([0, 1, 0, 1, 2, 1])
    metrics = StreamingMetrics(average=average)
    assert metrics.update(truth[:4], pred[:4]) is metrics
    metrics.update(truth[4:], pred[4:])
    report = metrics.result()
    np.testing.assert_array_equal(report["labels"], [0, 1, 2])
    np.testing.assert_array_equal(report["confusion_matrix"], confusion_matrix(truth, pred))
    assert report["count"] == 6
    assert report["accuracy"] == accuracy(truth, pred)
    assert report["error_rate"] == 1 - report["accuracy"]
    # The batch reference accepts two labels in binary mode. Collapse the
    # other two classes to their one-vs-rest label for that comparison.
    ref_truth, ref_pred = (truth == 1, pred == 1) if average == "binary" else (truth, pred)
    for name, ref in [("precision", precision), ("recall", recall), ("f1", f1)]:
        assert report[name] == pytest.approx(ref(ref_truth, ref_pred, average=average))


@pytest.mark.parametrize("window_size", [1, 5, 12])
def test_rolling_metrics_match_a_row_window(window_size):
    rng = np.random.default_rng(27)
    truth = rng.choice(["wet", "dry"], size=31)
    pred = rng.choice(["wet", "dry"], size=31)
    scores = rng.choice([0.0, 0.5, 1.0], size=31)  # ties across chunks
    metrics = StreamingMetrics(average="macro", pos_label="wet", window_size=window_size)
    for start, end in zip([0, 7, 8, 9, 16, 25], [7, 8, 9, 16, 25, 31]):
        metrics.update(truth[start:end], pred[start:end], y_score=scores[start:end])
        expected = slice(max(0, end - window_size), end)
        report = metrics.result()
        np.testing.assert_array_equal(report["confusion_matrix"],
                                      confusion_matrix(truth[expected], pred[expected]))
        assert report["count"] == report["auc_count"] == min(end, window_size)
        assert report["accuracy"] == accuracy(truth[expected], pred[expected])
        expected_auc = _pairwise_auc(truth[expected], scores[expected], "wet")
        np.testing.assert_allclose(report["auc"], expected_auc)


def test_rolling_window_evicts_unscored_rows_and_obsolete_labels():
    metrics = StreamingMetrics(window_size=2)
    metrics.update([2, 1], [2, 2], y_score=[0.2, 0.8])
    metrics.update([0], [0])
    metrics.update([0], [1], y_score=[0.4])
    report = metrics.result()
    np.testing.assert_array_equal(report["labels"], [0, 1])
    np.testing.assert_array_equal(report["confusion_matrix"], [[1, 1], [0, 0]])
    assert report["count"] == 2 and report["auc_count"] == 1
    assert np.isnan(report["auc"])


@pytest.mark.parametrize("scores,auc", [
    ([0.1, 0.9, 0.2, 0.8], 1.0),
    ([0.9, 0.1, 0.8, 0.2], 0.0),
    ([0.5] * 4, 0.5),
    ([0.1, 0.5, 0.5, 0.9], 0.875),
])
def test_exact_auc_accumulates_score_order_and_ties(scores, auc):
    metrics = StreamingMetrics()
    metrics.update([0, 1], [0, 0], y_score=scores[:2])
    metrics.update([0, 1], [0, 0], y_score=scores[2:])
    assert metrics.result()["auc"] == pytest.approx(auc)


def test_one_class_and_multiclass_auc_are_undefined_without_warnings():
    metrics = StreamingMetrics(average="macro")
    metrics.update([1, 1], [0, 0], y_score=[0.0, 1.0])
    assert np.isnan(metrics.result()["auc"])
    metrics.update([0, 2], [0, 0], y_score=[0.0, 0.0])
    assert np.isnan(metrics.result()["auc"])
    assert metrics.result()["auc_count"] == 4


def test_wide_numeric_labels_are_not_rounded_together():
    a = 2**53
    metrics = StreamingMetrics(pos_label=a + 1)
    metrics.update([float(a)], [a + 1], y_score=[0.9])
    metrics.update([a + 1], [a + 1], y_score=[1.0])
    report = metrics.result()
    assert report["labels"].tolist() == [a, a + 1]
    np.testing.assert_array_equal(report["confusion_matrix"], [[0, 1], [0, 1]])
    assert report["precision"] == 0.5 and report["recall"] == 1 and report["auc"] == 1


def test_reset_clears_history_and_outputs_do_not_alias_state():
    labels = np.array([1, 0])
    scores = np.array([1.0, 0.0])
    metrics = StreamingMetrics(window_size=4).update(labels, labels, y_score=scores)
    labels[:] = 9
    scores[:] = 0
    report = metrics.result()
    report["labels"][:] = -1
    report["confusion_matrix"][:] = -1
    metrics.update([1], [1], y_score=[1.0])
    assert metrics.result()["auc"] == 1.0
    np.testing.assert_array_equal(metrics.result()["confusion_matrix"], [[1, 0], [0, 2]])
    assert metrics.reset() is metrics
    reset = metrics.update([], []).result()
    assert reset["count"] == reset["auc_count"] == 0
    assert reset["precision"] == reset["recall"] == reset["f1"] == 0
    assert np.isnan(reset["accuracy"]) and np.isnan(reset["auc"])
    assert reset["confusion_matrix"].shape == (0, 0)


@pytest.mark.parametrize("truth,pred,score,exc", [
    ([0], [], None, ValueError),
    ([0], [0], [[1.0]], ValueError),
    ([0], [0], [np.inf], ValueError),
    ([0], [0], [np.nan], ValueError),
    ([0], ["0"], None, TypeError),
    (["a"], ["a"], None, TypeError),
])
def test_invalid_update_leaves_metrics_intact(truth, pred, score, exc):
    metrics = StreamingMetrics().update([0, 1], [1, 1], y_score=[0.1, 0.7])
    with pytest.raises(exc):
        metrics.update(truth, pred, y_score=score)
    report = metrics.result()
    assert report["count"] == 2 and report["auc"] == 1.0
    np.testing.assert_array_equal(report["confusion_matrix"], [[0, 1], [0, 1]])
