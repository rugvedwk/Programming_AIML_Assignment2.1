"""Shared whole-chunk buffering for periodic retraining."""

import numpy as np

from ._validation import _discrete_kind, _wide_integer


def _check_feature_promotion(X, dtype):
    if not _wide_integer(X.dtype, dtype):
        return
    converted = X.astype(dtype)
    # The upper bound is exclusive. Some platforms saturate on a cast back,
    # which could hide the rounding of the largest integer to this bound.
    upper = 1 << (np.iinfo(X.dtype).bits - (X.dtype.kind == "i"))
    if (converted >= dtype.type(upper)).any() or not np.array_equal(
        converted.astype(X.dtype), X
    ):
        raise ValueError("mixing these feature dtypes would lose integer precision")


def _join_labels(chunks):
    dtype = np.result_type(*(chunk.dtype for chunk in chunks))
    if any(_wide_integer(chunk.dtype, dtype) for chunk in chunks):
        dtype = np.dtype(object)
    return np.concatenate(chunks, dtype=dtype)


class _TrainingBuffer:
    """Own pending arrays and hand one assembled batch to a fitting callback."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._chunks = []
        self.rows = 0
        self._dtype = None
        self._kind = None

    def submit(self, X, y, threshold, train):
        """Accept validated arrays and return the fitted row count, or zero."""
        if self._chunks and X.shape[1] != self._chunks[0][0].shape[1]:
            expected = self._chunks[0][0].shape[1]
            raise ValueError(f"X has {X.shape[1]} features but the buffer expects {expected}")
        if not len(X):
            return 0
        kind = _discrete_kind(y)
        if self._kind is not None and kind != self._kind:
            raise TypeError("y label kind must match the pending chunks")
        dtype = X.dtype
        if self._dtype is not None:
            dtype = np.result_type(self._dtype, dtype)
            if dtype != self._dtype:
                for previous_X, _ in self._chunks:
                    _check_feature_promotion(previous_X, dtype)
            _check_feature_promotion(X, dtype)

        total = self.rows + len(X)
        if total < threshold:
            self._chunks.append((X.copy(), y.copy()))
            self.rows = total
            self._dtype = dtype
            self._kind = kind
            return 0

        # Assemble a trial batch. Keep previous pending rows until training
        # succeeds so retrying a failed call cannot duplicate its chunk.
        if self._chunks:
            features, labels = zip(*self._chunks)
            X = np.concatenate((*features, X), dtype=dtype)
            y = _join_labels((*labels, y))
        train(X, y)
        self.clear()
        return total
