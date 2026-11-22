"""Incremental confusion counts and exact binary ROC AUC."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from numcompute.utils import check_array, check_labels, check_same_length

from ._buffer import _join_labels
from ._validation import _discrete_kind, _integer

__all__ = ["StreamingMetrics"]


def _native_label(value):
    return value.item() if isinstance(value, np.generic) else value


def _positive_mask(labels, pos_label):
    """Compare Python scalars to preserve wide integer labels."""
    return np.fromiter(
        (_native_label(label) == pos_label for label in labels),
        dtype=bool, count=len(labels),
    )


@dataclass(frozen=True)
class _Batch:
    truth: np.ndarray
    predicted: np.ndarray
    scores: np.ndarray | None
    kind: str | None

    def slice(self, rows, *, copy=False):
        truth, predicted = self.truth[rows], self.predicted[rows]
        scores = None if self.scores is None else self.scores[rows]
        if copy:
            truth, predicted = truth.copy(), predicted.copy()
            if scores is not None:
                scores = scores.copy()
        return _Batch(truth, predicted, scores, self.kind)


def _make_batch(y_true, y_pred, y_score):
    """Validate one batch and align labels without lossy numeric casts."""
    truth, predicted = check_labels(y_true), check_labels(y_pred)
    check_same_length(truth, predicted)
    kind = _discrete_kind(truth) if len(truth) else None
    if kind is not None and kind != _discrete_kind(predicted):
        raise TypeError("true and predicted labels must use the same label kind")
    if y_score is not None:
        y_score = check_array(y_score, dtype=np.float64, allow_nan=False, allow_inf=False)
        if y_score.ndim != 1:
            raise ValueError("y_score must have shape (n_samples,)")
        check_same_length(truth, y_score)
    if kind is not None:
        labels = _join_labels((truth, predicted))
        if labels.dtype.kind == "O":
            labels = np.fromiter(
                (_native_label(value) for value in labels), dtype=object, count=len(labels),
            )
        truth, predicted = np.split(labels, 2)
    return _Batch(truth, predicted, y_score, kind)


def _change_counts(mapping, key, amount):
    new_count = mapping.get(key, 0) + amount
    if new_count:
        mapping[key] = new_count
    else:
        del mapping[key]


def _divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator != 0)


class StreamingMetrics:
    """Collect classification metrics from batches of truth and predictions.

    average is "binary" (positive label vs all others), "macro" (unweighted
    mean over observed labels) or "micro" (pooled counts). pos_label is a
    scalar label and defaults to 1. window_size is a positive number of scored
    rows to retain, or None for cumulative metrics.

    Confusion counts use observed label pairs. Exact AUC uses grouped counts
    by score and true label. Without a window its storage grows with the
    number of distinct score/label pairs. With a window, raw rows and the
    dictionaries are bounded by window_size. Scores are optional and only
    scored rows with a supplied continuous score contribute to AUC.
    """

    def __init__(self, *, average="binary", pos_label=1, window_size=None):
        if not isinstance(average, str) or average not in ("binary", "macro", "micro"):
            raise ValueError("average must be 'binary', 'macro' or 'micro'")
        if window_size is not None:
            window_size = _integer(window_size, "window_size", 1)
        label = np.asarray(pos_label)
        if label.ndim:
            raise TypeError("pos_label must be a scalar label")
        self._pos_label = _native_label(check_labels(label.reshape(1))[0])
        self._average = average
        self._window_size = window_size
        self.reset()

    def reset(self):
        """Clear accumulated counts and window rows, retaining configuration."""
        self._pairs = {}
        self._score_groups = {}
        self._history = deque()
        self._kind = None
        self._count = 0
        self._auc_count = 0
        return self

    def _validate_batch(self, batch):
        if batch.kind is not None and self._kind not in (None, batch.kind):
            raise TypeError("label kind must match previous updates")

    def _accumulate(self, batch, sign):
        rows = len(batch.truth)
        labels, codes = np.unique(
            np.concatenate((batch.truth, batch.predicted)), return_inverse=True,
        )
        labels = labels.tolist()
        label_count = len(labels)
        truth, predicted = codes[:rows], codes[rows:]
        pairs, counts = np.unique(truth * label_count + predicted, return_counts=True)
        for pair, count in zip(pairs.tolist(), counts.tolist()):
            actual, guessed = divmod(pair, label_count)
            _change_counts(self._pairs, (labels[actual], labels[guessed]), count * sign)
        if batch.scores is not None:
            scores, score_codes = np.unique(batch.scores, return_inverse=True)
            groups, counts = np.unique(score_codes * label_count + truth, return_counts=True)
            for group, count in zip(groups.tolist(), counts.tolist()):
                score, actual = divmod(group, label_count)
                _change_counts(self._score_groups, (float(scores[score]), labels[actual]),
                               count * sign)
            self._auc_count += sign * rows
        self._count += sign * rows

    def _update_batch(self, batch):
        if not len(batch.truth):
            return
        window = self._window_size
        if window is not None:
            if len(batch.truth) >= window:
                # Every older row has expired. Own only the arriving suffix.
                self.reset()
                batch = batch.slice(slice(-window, None), copy=True)
            else:
                batch = batch.slice(slice(None), copy=True)
                excess = self._count + len(batch.truth) - window
                while excess > 0:
                    oldest = self._history.popleft()
                    dropped = min(excess, len(oldest.truth))
                    self._accumulate(oldest.slice(slice(dropped)), -1)
                    if dropped < len(oldest.truth):
                        # Use views for small evictions and compact once the
                        # original allocation is more than twice as large.
                        remaining = len(oldest.truth) - dropped
                        backing = oldest.truth.base
                        slots = len(backing) if backing is not None else len(oldest.truth)
                        self._history.appendleft(
                            oldest.slice(slice(dropped, None), copy=2 * remaining < slots)
                        )
                    excess -= dropped
            self._history.append(batch)
        self._accumulate(batch, 1)
        self._kind = batch.kind

    def update(self, y_true_chunk, y_pred_chunk, *, y_score=None):
        """Add length-matched 1-D arrays and return self.

        y_score is an optional finite 1-D score vector converted to float64.
        Larger values favour pos_label. Empty valid inputs do nothing. Errors
        leave state intact. Inputs are never modified. The work is dominated
        by sorting the incoming labels, pairs and scores plus any evicted
        rolling rows. Counts are updated once per distinct pair or score group.
        """
        batch = _make_batch(y_true_chunk, y_pred_chunk, y_score)
        self._validate_batch(batch)
        self._update_batch(batch)
        return self

    def _confusion(self):
        labels = sorted({label for pair in self._pairs for label in pair})
        index = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for (actual, guessed), count in self._pairs.items():
            matrix[index[actual], index[guessed]] = count
        return np.array(labels, dtype=object), matrix

    def _auc(self):
        if len({label for _, label in self._score_groups}) != 2:
            return float("nan")
        groups = sorted(self._score_groups, key=lambda pair: pair[0])
        positives = np.array([label == self._pos_label for _, label in groups], dtype=bool)
        counts = np.array([self._score_groups[group] for group in groups], dtype=np.int64)
        n_pos = counts[positives].sum()
        n_neg = counts[~positives].sum()
        if not n_pos or not n_neg:
            return float("nan")
        scores = np.array([score for score, _ in groups])
        starts = np.r_[0, 1 + np.flatnonzero(scores[1:] != scores[:-1])]
        pos = np.add.reduceat(np.where(positives, counts, 0), starts)
        neg = np.add.reduceat(np.where(positives, 0, counts), starts)
        # Count concordant pairs in increasing score order and grant half
        # credit for ties. Normalise before multiplying to avoid overflow.
        return float(np.sum((pos / n_pos) * ((np.cumsum(neg) - 0.5 * neg) / n_neg)))

    def result(self):
        """Return fresh count, accuracy, precision, recall, F1 and AUC values.

        The dict includes count, accuracy, error_rate, precision, recall, f1,
        auc, auc_count, labels and confusion_matrix. Confusion rows are true
        labels and columns are predictions in sorted labels order. Undefined
        precision, recall and F1 use zero. Empty accuracy and undefined AUC use
        NaN. AUC requires exactly two true classes among supplied scores,
        including pos_label. Returned arrays are independent copies.

        For K observed labels and G score/label groups, the dense confusion
        result needs O(K²) space and the AUC query sorts O(G) groups.
        """
        labels, matrix = self._confusion()
        true_positive = matrix.diagonal()
        actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
        correct = int(true_positive.sum())
        accuracy = correct / self._count if self._count else float("nan")
        precision = _divide(true_positive, predicted)
        recall = _divide(true_positive, actual)
        f1 = 2 * _divide(true_positive, actual + predicted)
        if self._average == "micro":
            p = r = f = accuracy if self._count else 0.0
        elif self._average == "macro":
            p, r, f = (precision.mean(), recall.mean(), f1.mean()) if len(labels) else (0, 0, 0)
        else:
            positive = _positive_mask(labels, self._pos_label)
            p, r, f = precision[positive].sum(), recall[positive].sum(), f1[positive].sum()
        return {
            "count": self._count, "accuracy": accuracy, "error_rate": 1 - accuracy,
            "precision": float(p), "recall": float(r), "f1": float(f),
            "auc": self._auc(), "auc_count": self._auc_count,
            "labels": labels, "confusion_matrix": matrix,
        }
