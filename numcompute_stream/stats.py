"""Column-wise statistics for cumulative streams and bounded row windows."""

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np

from numcompute.pipeline import NotFittedError
from numcompute.preprocessing import _column_moments
from numcompute.stats import quantile as _batch_quantile
from numcompute.utils import check_array

from ._validation import _numeric_matrix

__all__ = ["StreamingStats", "update_stats"]

@dataclass(frozen=True)
class _Moments:
    count: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_batch(cls, X):
        mean, std, count = _column_moments(X)
        return cls(count, mean, std)

    def merge(self, other):
        count = self.count + other.count
        first = self.count / np.maximum(count, 1)
        second = other.count / np.maximum(count, 1)
        cross = np.sqrt(first * second)
        # Chan's population-variance formula has two within-batch terms and
        # one between-batch term. Combine their square roots to keep a finite
        # standard deviation when the variance itself overflows float64.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            delta = other.mean - self.mean
            mean = self.mean + delta * second
            # With opposite signs, the rounded difference can erase the
            # residual even if it does not overflow (for example ±1e16).
            same_sign = np.signbit(self.mean) == np.signbit(other.mean)
            mean = np.where(same_sign, mean, self.mean * first + other.mean * second)
            spread = np.where(
                np.isfinite(delta), delta * cross, other.mean * cross - self.mean * cross,
            )
            std = np.hypot(np.hypot(self.std * np.sqrt(first),
                                    other.std * np.sqrt(second)), spread)
        mean = np.clip(mean, np.minimum(self.mean, other.mean), np.maximum(self.mean, other.mean))
        mean = np.where(self.count == 0, other.mean, np.where(other.count == 0, self.mean, mean))
        # The population standard deviation cannot exceed the largest finite
        # float when every observation is finite. Clip roundoff at that bound.
        return _Moments(count, mean, np.minimum(std, np.finfo(np.float64).max))

    def variance(self):
        with np.errstate(over="ignore", under="ignore"):
            return self.std**2


def _histogram(X, edges):
    n_bins = len(edges) - 1
    cells = np.searchsorted(edges, X, side="right") - 1
    cells[X == edges[-1]] = n_bins - 1
    valid = (cells >= 0) & (cells < n_bins)
    keys = cells + np.arange(X.shape[1]) * n_bins
    return np.bincount(keys[valid], minlength=X.shape[1] * n_bins).reshape(-1, n_bins)


class StreamingStats:
    """Summarise a row stream of real matrices with a fixed feature count.

    Parameters
    ----------
    bins : array_like, shape (n_bins + 1,), optional
        Finite, strictly increasing edges shared by all features.
        Histogram queries require edges.
    window_size : int or None, default None
        Use the last window_size rows for all results, including rows with NaN.
        None uses the entire stream.
    track_quantiles : bool, default True
        Enable exact quantiles by retaining rows. Disable this for bounded
        cumulative moments and histograms. A rolling window always owns its rows.

    Notes
    -----
    Axis 0 is the sample axis. Columns omit NaNs independently and reject
    infinity. Calculations and stored rows use float64, which can round large
    integers. Undefined moment and quantile results are NaN.

    In cumulative mode, retained state is O(d + d*b) for d features and b
    histogram bins, plus O(n*d) if exact quantiles are enabled. Updates compute
    moments and histogram counts per chunk. Rolling mode stores O(w*d) values
    and evaluates moment and histogram queries on demand. Moments are cached
    until the next update. Exact quantiles sort the retained values on demand.
    """

    def __init__(self, *, bins=None, window_size=None, track_quantiles=True):
        if window_size is not None:
            if isinstance(window_size, (bool, np.bool_)) or not isinstance(
                window_size, (int, np.integer)
            ):
                raise TypeError("window_size must be a positive integer or None")
            if window_size < 1:
                raise ValueError("window_size must be positive")
            window_size = int(window_size)
        if not isinstance(track_quantiles, (bool, np.bool_)):
            raise TypeError("track_quantiles must be boolean")
        if bins is not None:
            bins = check_array(bins, allow_nan=False, allow_inf=False, dtype=np.float64)
            if bins.ndim != 1 or len(bins) < 2 or np.any(bins[1:] <= bins[:-1]):
                raise ValueError("bins must be a vector of at least two increasing edges")
            bins = bins.copy()
        self._window_size = window_size
        self._track_quantiles = track_quantiles
        self._edges = bins
        self._history = deque()
        self._stored_rows = 0
        self._n_features = None
        self._moments = None
        self._counts = None

    def _record(self, X):
        window = self._window_size
        if window is not None and len(X) >= window:
            self._history = deque([X[-window:].copy()])
            self._stored_rows = window
            return
        self._history.append(X.copy())
        self._stored_rows += len(X)
        if window is None:
            return
        excess = self._stored_rows - window
        while excess > 0:
            oldest = self._history.popleft()
            dropped = min(len(oldest), excess)
            if dropped < len(oldest):
                remaining = oldest[dropped:]
                # Compact occasionally to bound retained storage without
                # copying most of the window on every single-row update.
                backing = oldest if oldest.base is None else oldest.base
                if remaining.nbytes * 2 < backing.nbytes:
                    remaining = remaining.copy()
                self._history.appendleft(remaining)
            excess -= dropped
            self._stored_rows -= dropped

    def update_stats(self, X_chunk):
        """Update from a matrix of shape (chunk_rows, n_features) and return self.

        Empty chunks are no-ops after validation. Feature width stays fixed
        after the first nonempty chunk. This method preserves caller arrays.
        Raises TypeError for unsupported dtypes or ValueError for a bad shape,
        width or infinity. For m incoming rows, cumulative updates take
        O(m*d) time for moments and O(m*d*log(b+1) + d*b) with binning.
        Rolling updates copy admitted rows and evict old chunks or a prefix.
        """
        X = _numeric_matrix(X_chunk)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"X has {X.shape[1]} features, expected {self._n_features}")
        if not len(X):
            return self
        if self._window_size is None:
            batch = _Moments.from_batch(X)
            self._moments = batch if self._moments is None else self._moments.merge(batch)
            if self._edges is not None:
                counts = _histogram(X, self._edges)
                self._counts = counts if self._counts is None else self._counts + counts
        else:
            self._moments = None
        if self._track_quantiles or self._window_size is not None:
            self._record(X)
        self._n_features = X.shape[1]
        return self

    update = update_stats

    def _require_data(self):
        if self._n_features is None:
            raise NotFittedError("StreamingStats needs a nonempty chunk before queries")

    def _rows(self):
        if len(self._history) == 1:
            return self._history[0]
        return np.concatenate(tuple(self._history), axis=0)

    def _summary(self):
        self._require_data()
        if self._moments is None:
            self._moments = _Moments.from_batch(self._rows())
        return self._moments

    @property
    def retained_rows(self):
        """Number of retained observations available to queries, in constant time."""
        return self._stored_rows

    @property
    def count(self):
        """Non-NaN counts with shape (n_features,), returned as a copy."""
        return self._summary().count.copy()

    @property
    def mean(self):
        """Means with shape (n_features,), using NaN for unobserved columns."""
        summary = self._summary()
        return np.where(summary.count, summary.mean, np.nan)

    @property
    def std(self):
        """Population standard deviations with shape (n_features,)."""
        summary = self._summary()
        return np.where(summary.count, summary.std, np.nan)

    @property
    def variance(self):
        """Population variances with shape (n_features,).

        Empty columns return NaN. Unrepresentable positive variances return
        infinity without a warning.
        """
        summary = self._summary()
        return np.where(summary.count, summary.variance(), np.nan)

    def quantile(self, q, *, interpolation="linear"):
        """Return exact column quantiles for probabilities in [0, 1].

        Scalar q returns shape (n_features,) and a vector of size k returns
        (k, n_features). Uses NumCompute's five interpolation methods and
        returns NaN for all-missing columns. Raises NotFittedError before
        observing a chunk, ValueError when tracking is disabled or q and the
        interpolation mode are invalid, and TypeError for unsupported types.
        For r retained rows, time is O(d*r*log(r+1) + k*d), with
        O((r+k)*d) scratch space.
        """
        self._require_data()
        if not self._track_quantiles:
            raise ValueError("quantile tracking is disabled")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Percentile is undefined for an empty or all-NaN slice",
                category=RuntimeWarning,
            )
            return _batch_quantile(self._rows(), q, axis=0, interpolation=interpolation)

    def histogram(self):
        """Return copies of counts (n_features, n_bins) and the bin edges.

        Bins are left-closed except that the final bin also includes its right
        edge. Values outside the edges and NaNs do not count. Raises ValueError
        without configured bins and NotFittedError before the first chunk.
        Cumulative queries copy O(d*b) values. Rolling queries bin the window.
        """
        self._require_data()
        if self._edges is None:
            raise ValueError("histogram requires configured bins")
        counts = (self._counts.copy() if self._window_size is None
                  else _histogram(self._rows(), self._edges))
        return counts, self._edges.copy()


def update_stats(X_chunk, *, stats=None):
    """Update a supplied StreamingStats, or create one with the default settings.

    X_chunk is a real matrix of shape (chunk_rows, n_features). Returns the
    accumulator for the next call. Shape, NaN and cost rules are those of
    StreamingStats.update_stats. Raises TypeError for an invalid accumulator.
    """
    if stats is None:
        stats = StreamingStats()
    elif not isinstance(stats, StreamingStats):
        raise TypeError("stats must be a StreamingStats instance")
    return stats.update_stats(X_chunk)
