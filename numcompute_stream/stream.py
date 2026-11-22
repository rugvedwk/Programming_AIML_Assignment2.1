"""Score-then-train orchestration and per-chunk logs."""

from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from time import perf_counter

import numpy as np

from numcompute.pipeline import NotFittedError
from numcompute.utils import check_array, check_categories, check_labels, check_same_length

from .metrics import StreamingMetrics, _make_batch, _positive_mask

__all__ = ["ChunkLog", "ChunkScore", "StreamTrainer"]


@dataclass(frozen=True)
class ChunkScore:
    """Truth, predicted labels, optional positive scores and accuracy for one chunk."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    accuracy: float


@dataclass(frozen=True)
class ChunkLog:
    """Metrics, timing and model state after processing one nonempty chunk."""

    chunk: int
    rows: int
    rows_seen: int
    scored_samples: int
    chunk_accuracy: float
    cumulative_accuracy: float
    rolling_accuracy: float
    cumulative_auc: float
    rolling_auc: float
    pending_rows: int
    retrained: bool
    retrain_seconds: float
    model_array_bytes: int


def _array_bytes(value, seen):
    """Estimate retained NumPy array payloads, counting shared storage once."""
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        if isinstance(value.base, np.ndarray):
            return _array_bytes(value.base, seen)
        return value.nbytes
    if isinstance(value, dict):
        return sum(_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, deque)):
        return sum(_array_bytes(item, seen) for item in value)
    if not isinstance(value, type):
        if is_dataclass(value):
            return sum(_array_bytes(getattr(value, field.name), seen) for field in fields(value))
        if hasattr(value, "__dict__"):
            return _array_bytes(vars(value), seen)
    return 0


class StreamTrainer:
    """Evaluate a buffered classifier or pipeline before updating on each chunk.

    model implements predict and partial_fit, exposes pending_rows_ and
    n_retrains_, and raises NotFittedError before inference is available.
    average and pos_label configure StreamingMetrics. window_size optionally
    enables a second accumulator for the most recent scored rows. When
    track_auc is True (default), model also implements predict_proba and
    exposes classes_. A missing positive class contributes score zero.

    score_chunk previews predictions with no trainer state change. fit_chunk
    is the sole operation that updates metrics and logs. A chunk used to
    train the first tree has no pre-update prediction and no accuracy.
    The trainer uses the supplied model instance.
    """

    def __init__(self, model, *, average="binary", pos_label=1, window_size=None, track_auc=True):
        for method in ("predict", "partial_fit"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model must implement {method}()")
        for name in ("pending_rows_", "n_retrains_"):
            if not hasattr(model, name):
                raise TypeError(f"model must expose {name}")
        if not isinstance(track_auc, (bool, np.bool_)):
            raise TypeError("track_auc must be boolean")
        if track_auc and not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must implement predict_proba() when track_auc=True")

        self.model = model
        self._cumulative = StreamingMetrics(average=average, pos_label=pos_label)
        self._rolling = (None if window_size is None else
                         StreamingMetrics(average=average, pos_label=pos_label,
                                          window_size=window_size))
        self._track_auc = track_auc
        self._logs = []
        self._rows_seen = 0
        self.latest_score = None

    @property
    def logs(self):
        """Return an ordered tuple of immutable ChunkLog entries."""
        return tuple(self._logs)

    def result(self, *, rolling=False):
        """Return cumulative metrics or the configured rolling-window metrics."""
        if rolling and self._rolling is None:
            raise ValueError("rolling metrics require a window_size")
        return (self._rolling if rolling else self._cumulative).result()

    @staticmethod
    def _arrays(X, y):
        X, y = check_categories(X), check_labels(y)
        check_same_length(X, y)
        if not X.shape[1]:
            raise ValueError("X must have at least one feature")
        return X, y

    def _predict_batch(self, X, y):
        if not len(y):
            return None
        try:
            predicted = self.model.predict(X)
        except NotFittedError:
            return None

        scores = None
        if self._track_auc:
            classes = check_labels(self.model.classes_)
            probabilities = check_array(
                self.model.predict_proba(X), ensure_2d=True, allow_nan=False, allow_inf=False,
            )
            if not len(classes) or probabilities.shape != (len(y), len(classes)):
                raise ValueError("predict_proba must return shape (n_samples, n_classes)")
            if ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("predict_proba must return probabilities in [0, 1]")
            positive = _positive_mask(classes, self._cumulative._pos_label)
            scores = probabilities[:, positive].sum(axis=1)
        return _make_batch(y, predicted, scores)

    @staticmethod
    def _as_score(batch):
        if batch is None:
            return None
        return ChunkScore(batch.truth.copy(), batch.predicted.copy(),
                          None if batch.scores is None else batch.scores.copy(),
                          float(np.mean(batch.truth == batch.predicted)))

    def score_chunk(self, X, y):
        """Preview a ChunkScore for a raw matrix and a 1-D label vector.

        Returns None for an empty chunk or before inference is available.
        Unsupported types, invalid shapes or prediction failures raise.
        The call neither trains the model nor records a metric update.
        Cost is model inference plus validation and O(n_samples) output.
        """
        X, y = self._arrays(X, y)
        return self._as_score(self._predict_batch(X, y))

    def fit_chunk(self, X, y):
        """Score then learn from a chunk and return its ChunkLog.

        X has shape (n_samples, n_features) and y has shape (n_samples,).
        Empty chunks return None without logging. Input or training errors
        propagate. Metrics and log entries are recorded after successful
        training. Warm-up predictions and undefined metrics use NaN in logs.

        retrain_seconds is elapsed partial_fit time on triggering calls,
        including preprocessing and buffering. model_array_bytes is the
        retained NumPy payload estimate after the call, including model
        arrays and pending rows. It excludes Python overhead, temporary
        training allocations, metrics and logs. Work includes inference,
        model update, metric accumulation/queries and a model-state traversal.
        """
        X, y = self._arrays(X, y)
        if not len(y):
            return None
        batch = self._predict_batch(X, y)
        if batch is not None:
            self._cumulative._validate_batch(batch)
            if self._rolling is not None:
                self._rolling._validate_batch(batch)
        score = self._as_score(batch)

        previous_retrains = self.model.n_retrains_
        start = perf_counter()
        self.model.partial_fit(X, y)
        duration = perf_counter() - start
        retrained = self.model.n_retrains_ > previous_retrains

        if batch is not None:
            self._cumulative._update_batch(batch)
            if self._rolling is not None:
                self._rolling._update_batch(batch)
        self.latest_score = score
        self._rows_seen += len(y)
        cumulative = self.result()
        rolling = self.result(rolling=True) if self._rolling is not None else {}
        log = ChunkLog(
            chunk=len(self._logs) + 1, rows=len(y), rows_seen=self._rows_seen,
            scored_samples=cumulative["count"],
            chunk_accuracy=score.accuracy if score is not None else float("nan"),
            cumulative_accuracy=cumulative["accuracy"],
            rolling_accuracy=rolling.get("accuracy", float("nan")),
            cumulative_auc=cumulative["auc"], rolling_auc=rolling.get("auc", float("nan")),
            pending_rows=self.model.pending_rows_, retrained=retrained,
            retrain_seconds=duration if retrained else 0.0,
            model_array_bytes=_array_bytes(self.model, set()),
        )
        self._logs.append(log)
        return log
